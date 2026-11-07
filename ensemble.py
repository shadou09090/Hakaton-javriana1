#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ensemble.py — Pipeline Kaggle Grandmaster para SpaceLAB:
1. Inferencia Sliced (Ventana Deslizante) para modelos entrenados en tiles.
2. Weighted Boxes Fusion (WBF) multi-modelo y multi-vista.
3. Recalibración Monótona de Umbral por Clase (Gamma Calibration).
4. Fusión Selectiva ponderada por clase entre modelos (PC 1 + PC 2).
"""
import argparse
from collections import defaultdict
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parent
KIT_DIR = ROOT / "kit_participantes"
sys.path.insert(0, str(KIT_DIR))

import formato_envio as fmt
from ultralytics import YOLO

CLASES = ["small_boat", "cargo_ship", "fishing_vessel", "warship", "work_vessel"]

# Umbrales óptimos calculados en val_dev para cada clase
UMBRALES_OPTIMOS_CLASE = {
    0: 0.37,  # small_boat
    1: 0.33,  # cargo_ship
    2: 0.20,  # fishing_vessel (dispara Recall de 22% a 48%)
    3: 0.53,  # warship
    4: 0.48   # work_vessel
}


def calcular_gammas(t_global: float = 0.33, umbrales_clase: dict = None) -> dict:
    """Calcula el exponente gamma para la transformación monótona s_new = s^gamma."""
    if umbrales_clase is None:
        umbrales_clase = UMBRALES_OPTIMOS_CLASE
    gammas = {}
    for c, tc in umbrales_clase.items():
        tc_safe = max(0.02, min(0.95, tc))
        gammas[c] = math.log(t_global) / math.log(tc_safe)
    return gammas


def aplicar_calibracion_gamma(dets: list, gammas: dict) -> list:
    """Aplica s_new = s^gamma_c a las confianzas para alinear umbrales al umbral global."""
    calib = []
    for c, xc, yc, w, h, s in dets:
        g = gammas.get(int(c), 1.0)
        s_new = float(np.clip(s ** g, 1e-5, 1.0))
        calib.append((c, xc, yc, w, h, s_new))
    calib.sort(key=lambda d: -d[5])
    return calib


def bb_iou_np(b1, b2):
    """IoU entre dos cajas [x1, y1, x2, y2]."""
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    union = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    return inter / union if union > 0 else 0.0


def weighted_boxes_fusion(dets: list, iou_thr: float = 0.55, max_det: int = 300, conf_mode: str = "max") -> list:
    """
    Weighted Boxes Fusion (WBF) puro por clase.
    dets: lista de tuplas (clase, xc, yc, w, h, conf, peso_modelo)
    """
    if not dets:
        return []

    by_class = defaultdict(list)
    for d in dets:
        if len(d) == 6:
            c, xc, yc, w, h, s = d
            weight = 1.0
        else:
            c, xc, yc, w, h, s, weight = d
            
        if w >= 0.0005 and h >= 0.0005 and 0.0 <= s <= 1.0:
            # pasar a xyxy normalizado
            x1 = xc - w / 2.0
            y1 = yc - h / 2.0
            x2 = xc + w / 2.0
            y2 = yc + h / 2.0
            by_class[int(c)].append([x1, y1, x2, y2, s * weight, s])

    final_dets = []
    for c, items in by_class.items():
        arr = np.array(items, dtype=np.float32)
        boxes = arr[:, :4]
        effective_scores = arr[:, 4]
        raw_scores = arr[:, 5]

        order = np.argsort(-effective_scores)
        boxes = boxes[order]
        effective_scores = effective_scores[order]
        raw_scores = raw_scores[order]

        clusters_boxes = []
        clusters_weights = []
        clusters_raw = []
        fused_boxes = []

        for b, eff_s, raw_s in zip(boxes, effective_scores, raw_scores):
            matched = False
            for c_idx, fb in enumerate(fused_boxes):
                if bb_iou_np(b, fb) >= iou_thr:
                    clusters_boxes[c_idx].append(b)
                    clusters_weights[c_idx].append(eff_s)
                    clusters_raw[c_idx].append(raw_s)
                    # recalcular caja promedio ponderada
                    cb = np.array(clusters_boxes[c_idx])
                    cw = np.array(clusters_weights[c_idx])
                    fused_boxes[c_idx] = np.sum(cb * cw[:, None], axis=0) / max(np.sum(cw), 1e-8)
                    matched = True
                    break
            if not matched:
                clusters_boxes.append([b])
                clusters_weights.append([eff_s])
                clusters_raw.append([raw_s])
                fused_boxes.append(b)

        for c_idx, fb in enumerate(fused_boxes):
            c_raw = clusters_raw[c_idx]
            final_conf = max(c_raw) if conf_mode == "max" else float(np.mean(c_raw))
            # pasar de xyxy normalizado a xc, yc, w, h
            x1, y1, x2, y2 = fb
            w_box = max(0.0005, float(x2 - x1))
            h_box = max(0.0005, float(y2 - y1))
            xc_box = float((x1 + x2) / 2.0)
            yc_box = float((y1 + y2) / 2.0)
            final_dets.append((c, xc_box, yc_box, w_box, h_box, final_conf))

    final_dets.sort(key=lambda d: -d[5])
    return final_dets[:max_det]


def predecir_imagen_sliced(model, img_bgr, tile_size: int = 512, overlap: float = 0.25, device=0, conf: float = 0.005) -> list:
    """
    Inferencia por ventana deslizante (Sliced) sobre una imagen satelital con padding reflexivo
    y reproyeccion exacta de coordenadas.
    """
    H, W = img_bgr.shape[:2]
    dets = []
    
    # Si la imagen cabe en un tile, inferir directamente
    if W <= tile_size and H <= tile_size:
        r = model.predict(img_bgr, imgsz=tile_size, conf=conf, device=device, verbose=False)[0].boxes
        if len(r) > 0:
            for (x, y, w, h), c, s in zip(r.xywhn.cpu().numpy(), r.cls.cpu().numpy(), r.conf.cpu().numpy()):
                if w >= 0.0005 and h >= 0.0005:
                    dets.append((int(c), float(x), float(y), float(w), float(h), float(s), 1.0))
        return dets

    # Pad reflexivo si alguna dimension es menor a tile_size
    pad_w = max(0, tile_size - W)
    pad_h = max(0, tile_size - H)
    if pad_w > 0 or pad_h > 0:
        img_padded = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    else:
        img_padded = img_bgr
    
    H_pad, W_pad = img_padded.shape[:2]

    step = int(tile_size * (1.0 - overlap))
    x_starts = list(range(0, W_pad - tile_size + 1, step))
    if not x_starts or x_starts[-1] + tile_size < W_pad:
        x_starts.append(W_pad - tile_size)
    y_starts = list(range(0, H_pad - tile_size + 1, step))
    if not y_starts or y_starts[-1] + tile_size < H_pad:
        y_starts.append(H_pad - tile_size)

    # 1. Tambien inferir imagen completa reescalada a tile_size para barcos grandes
    r_full = model.predict(img_bgr, imgsz=tile_size, conf=conf, device=device, verbose=False)[0].boxes
    if len(r_full) > 0:
        for (x, y, w, h), c, s in zip(r_full.xywhn.cpu().numpy(), r_full.cls.cpu().numpy(), r_full.conf.cpu().numpy()):
            if w >= 0.0005 and h >= 0.0005:
                # Peso menor en botes pequenos para la vista completa
                w_mult = 1.2 if int(c) in {1, 3} else 0.7
                dets.append((int(c), float(x), float(y), float(w), float(h), float(s), w_mult))

    # 2. Recorrer ventanas deslizantes en batch para maxima velocidad
    patches = []
    offsets = []
    for ys in y_starts:
        for xs in x_starts:
            patches.append(img_padded[ys:ys + tile_size, xs:xs + tile_size])
            offsets.append((xs, ys))

    batch_size = 16
    for b_idx in range(0, len(patches), batch_size):
        sub_patches = patches[b_idx:b_idx + batch_size]
        sub_offsets = offsets[b_idx:b_idx + batch_size]
        results = model.predict(sub_patches, imgsz=tile_size, conf=conf, device=device, verbose=False)
        for r, (xs, ys) in zip(results, sub_offsets):
            if len(r.boxes) > 0:
                xyxy_patch = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for (px1, py1, px2, py2), c, s in zip(xyxy_patch, classes, confs):
                    # Clampear a coordenadas reales de la imagen (ignorando padding)
                    rx1 = max(0.0, min(float(W), float(px1 + xs)))
                    ry1 = max(0.0, min(float(H), float(py1 + ys)))
                    rx2 = max(0.0, min(float(W), float(px2 + xs)))
                    ry2 = max(0.0, min(float(H), float(py2 + ys)))
                    
                    rw = rx2 - rx1
                    rh = ry2 - ry1
                    if rw < 2.0 or rh < 2.0:
                        continue
                    
                    gxc = (rx1 + rx2) / (2.0 * W)
                    gyc = (ry1 + ry2) / (2.0 * H)
                    gw = rw / W
                    gh = rh / H
                    
                    if gw >= 0.0005 and gh >= 0.0005:
                        w_mult = 1.3 if int(c) in {0, 2, 4} else 0.8
                        dets.append((int(c), float(gxc), float(gyc), float(gw), float(gh), float(s), w_mult))

    return weighted_boxes_fusion(dets, iou_thr=0.50, conf_mode="max")


def predecir_con_tta_exacta(model_path: Path, images_dir: Path, out_dir: Path, imgsz: int = 1024, iou_thr: float = 0.55):
    """
    TTA con transformaciones ortogonales exactas (Original, HFlip, VFlip).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.name.startswith(".")])
    print(f"Iniciando TTA Ortogonal Exacto en {len(images)} imágenes a {imgsz} px...")

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    start_time = time.time()

    for idx, p in enumerate(images):
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        all_dets = []

        # 1. Vista Original (peso 1.0)
        r0 = model.predict(img_bgr, imgsz=imgsz, conf=0.005, device=device, verbose=False)[0].boxes
        if len(r0) > 0:
            for (x, y, w, h), c, s in zip(r0.xywhn.cpu().numpy(), r0.cls.cpu().numpy(), r0.conf.cpu().numpy()):
                all_dets.append((int(c), float(x), float(y), float(w), float(h), float(s), 1.0))

        # 2. Horizontal Flip exacto (peso 0.9)
        img_lr = np.fliplr(img_bgr).copy()
        r_lr = model.predict(img_lr, imgsz=imgsz, conf=0.005, device=device, verbose=False)[0].boxes
        if len(r_lr) > 0:
            for (x, y, w, h), c, s in zip(r_lr.xywhn.cpu().numpy(), r_lr.cls.cpu().numpy(), r_lr.conf.cpu().numpy()):
                all_dets.append((int(c), float(1.0 - x), float(y), float(w), float(h), float(s), 0.9))

        # 3. Vertical Flip exacto (peso 0.9)
        img_ud = np.flipud(img_bgr).copy()
        r_ud = model.predict(img_ud, imgsz=imgsz, conf=0.005, device=device, verbose=False)[0].boxes
        if len(r_ud) > 0:
            for (x, y, w, h), c, s in zip(r_ud.xywhn.cpu().numpy(), r_ud.cls.cpu().numpy(), r_ud.conf.cpu().numpy()):
                all_dets.append((int(c), float(x), float(1.0 - y), float(w), float(h), float(s), 0.9))

        # Fusión WBF de vistas
        fused = weighted_boxes_fusion(all_dets, iou_thr=iou_thr, conf_mode="max")
        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {s:.5f}" for c, x, y, w, h, s in fused]
        (out_dir / f"{p.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

        if (idx + 1) % 300 == 0 or (idx + 1) == len(images):
            fps = (idx + 1) / max(time.time() - start_time, 0.001)
            print(f"  {idx + 1}/{len(images)} imágenes ({fps:.1f} img/s)...")

    print(f"✔ TTA completado en {time.time() - start_time:.1f}s.")


def fusion_selectiva_modelos(carpeta_pc1: Path, carpeta_pc2: Path, out_dir: Path, iou_thr: float = 0.55):
    """
    Fusión Selectiva:
    - PC 2 (1024px Full Image): peso dominante en warship (clase 3) y cargo_ship (clase 1).
    - PC 1 (Tiles/768px): peso dominante en small_boat (0), fishing_vessel (2), work_vessel (4).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    all_names = sorted(list({f.name for f in carpeta_pc1.glob("*.txt") if f.name != "umbral.txt"} |
                            {f.name for f in carpeta_pc2.glob("*.txt") if f.name != "umbral.txt"}))
    print(f"Iniciando Fusión Selectiva PC1 + PC2 sobre {len(all_names)} archivos...")

    pesos_pc1 = {0: 1.4, 1: 0.7, 2: 1.4, 3: 0.6, 4: 1.4}
    pesos_pc2 = {0: 0.8, 1: 1.4, 2: 0.8, 3: 1.5, 4: 0.8}

    for name in all_names:
        dets = []
        p1 = carpeta_pc1 / name
        if p1.exists():
            for line in p1.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 6:
                    c, x, y, w, h, s = map(float, parts)
                    c = int(c)
                    dets.append((c, x, y, w, h, s, pesos_pc1.get(c, 1.0)))

        p2 = carpeta_pc2 / name
        if p2.exists():
            for line in p2.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 6:
                    c, x, y, w, h, s = map(float, parts)
                    c = int(c)
                    dets.append((c, x, y, w, h, s, pesos_pc2.get(c, 1.0)))

        fused = weighted_boxes_fusion(dets, iou_thr=iou_thr, conf_mode="max")
        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {s:.5f}" for c, x, y, w, h, s in fused]
        (out_dir / name).write_text("\n".join(lines), encoding="utf-8")

    print(f"✔ Fusión selectiva completada en {out_dir}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Subcomando TTA
    t_cmd = sub.add_parser("tta", help="Inferencia con TTA Ortogonal Exacto")
    t_cmd.add_argument("--model", type=Path, required=True)
    t_cmd.add_argument("--images", type=Path, default=ROOT / "spacelab_dataset/images/oculto")
    t_cmd.add_argument("--out-dir", type=Path, required=True)
    t_cmd.add_argument("--imgsz", type=int, default=1024)
    t_cmd.add_argument("--iou", type=float, default=0.55)

    # Subcomando Sliced (Ventana deslizante)
    s_cmd = sub.add_parser("sliced", help="Inferencia por ventana deslizante (Tiles)")
    s_cmd.add_argument("--model", type=Path, required=True)
    s_cmd.add_argument("--images", type=Path, default=ROOT / "spacelab_dataset/images/oculto")
    s_cmd.add_argument("--out-dir", type=Path, required=True)
    s_cmd.add_argument("--tile-size", type=int, default=512)
    s_cmd.add_argument("--overlap", type=float, default=0.25)
    s_cmd.add_argument("--out-zip", type=Path, default=None)
    s_cmd.add_argument("--t-global", type=float, default=0.33)

    # Subcomando Calibrar Gamma
    c_cmd = sub.add_parser("calibrate", help="Aplica recalibración monótona de umbrales por clase")
    c_cmd.add_argument("--in-dir", type=Path, required=True)
    c_cmd.add_argument("--out-dir", type=Path, required=True)
    c_cmd.add_argument("--t-global", type=float, default=0.33)

    # Subcomando Fusión Selectiva PC1 + PC2
    b_cmd = sub.add_parser("blend", help="Fusión selectiva de predicciones de PC1 y PC2")
    b_cmd.add_argument("--pc1-dir", type=Path, required=True)
    b_cmd.add_argument("--pc2-dir", type=Path, required=True)
    b_cmd.add_argument("--out-dir", type=Path, required=True)
    b_cmd.add_argument("--out-zip", type=Path, default=None)
    b_cmd.add_argument("--t-global", type=float, default=0.33)

    # Subcomando Empaquetar y Verificar
    p_cmd = sub.add_parser("pack", help="Empaquetar y verificar ZIP oficial")
    p_cmd.add_argument("--in-dir", type=Path, required=True)
    p_cmd.add_argument("--out-zip", type=Path, required=True)
    p_cmd.add_argument("--threshold", type=float, default=0.33)

    args = parser.parse_args()

    if args.cmd == "tta":
        predecir_con_tta_exacta(args.model, args.images, args.out_dir, imgsz=args.imgsz, iou_thr=args.iou)
    elif args.cmd == "sliced":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        imgs = sorted([p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.name.startswith(".")])
        m = YOLO(str(args.model))
        t0 = time.time()
        for idx, p in enumerate(imgs):
            img = cv2.imread(str(p))
            dets = predecir_imagen_sliced(m, img, tile_size=args.tile_size, overlap=args.overlap)
            lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {s:.5f}" for c, x, y, w, h, s in dets]
            (args.out_dir / f"{p.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            if (idx + 1) % 100 == 0 or (idx + 1) == len(imgs):
                fps = (idx + 1) / max(time.time() - t0, 0.001)
                print(f"  {idx + 1}/{len(imgs)} imágenes procesadas con Slicing ({fps:.1f} img/s)...")
        print(f"✔ Slicing completado en {time.time() - t0:.1f}s.")

        if args.out_zip:
            calib_dir = args.out_dir.parent / (args.out_dir.name + "_calib")
            gammas = calcular_gammas(t_global=args.t_global)
            calib_dir.mkdir(parents=True, exist_ok=True)
            for txt in args.out_dir.glob("*.txt"):
                if txt.name == "umbral.txt":
                    continue
                dets = []
                for l in txt.read_text(encoding="utf-8").splitlines():
                    parts = l.strip().split()
                    if len(parts) == 6:
                        c, x, y, w, h, s = map(float, parts)
                        dets.append((int(c), x, y, w, h, s))
                calib = aplicar_calibracion_gamma(dets, gammas)
                lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {s:.5f}" for c, x, y, w, h, s in calib]
                (calib_dir / txt.name).write_text("\n".join(lines), encoding="utf-8")

            n = fmt.crear_zip(str(calib_dir), str(args.out_zip), umbral=args.t_global)
            print(f"✔ Empaquetado: {args.out_zip} ({n} archivos, umbral {args.t_global:.2f})")
            fmt.verificar(str(args.out_zip), str(ROOT / "spacelab_dataset/images/oculto"))
    elif args.cmd == "calibrate":
        gammas = calcular_gammas(t_global=args.t_global)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for txt in args.in_dir.glob("*.txt"):
            if txt.name == "umbral.txt":
                continue
            dets = []
            for l in txt.read_text(encoding="utf-8").splitlines():
                parts = l.strip().split()
                if len(parts) == 6:
                    c, x, y, w, h, s = map(float, parts)
                    dets.append((int(c), x, y, w, h, s))
            calib = aplicar_calibracion_gamma(dets, gammas)
            lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {s:.5f}" for c, x, y, w, h, s in calib]
            (args.out_dir / txt.name).write_text("\n".join(lines), encoding="utf-8")
        print(f"✔ Recalibración monótona completada en {args.out_dir} con gammas {gammas}")
    elif args.cmd == "blend":
        fusion_selectiva_modelos(args.pc1_dir, args.pc2_dir, args.out_dir)
        if args.out_zip:
            # Primero calibrar
            calib_dir = args.out_dir.parent / (args.out_dir.name + "_calib")
            gammas = calcular_gammas(t_global=args.t_global)
            calib_dir.mkdir(parents=True, exist_ok=True)
            for txt in args.out_dir.glob("*.txt"):
                if txt.name == "umbral.txt":
                    continue
                dets = []
                for l in txt.read_text(encoding="utf-8").splitlines():
                    parts = l.strip().split()
                    if len(parts) == 6:
                        c, x, y, w, h, s = map(float, parts)
                        dets.append((int(c), x, y, w, h, s))
                calib = aplicar_calibracion_gamma(dets, gammas)
                lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {s:.5f}" for c, x, y, w, h, s in calib]
                (calib_dir / txt.name).write_text("\n".join(lines), encoding="utf-8")

            n = fmt.crear_zip(str(calib_dir), str(args.out_zip), umbral=args.t_global)
            print(f"✔ Empaquetado: {args.out_zip} ({n} archivos, umbral {args.t_global:.2f})")
            fmt.verificar(str(args.out_zip), str(ROOT / "spacelab_dataset/images/oculto"))
    elif args.cmd == "pack":
        n = fmt.crear_zip(str(args.in_dir), str(args.out_zip), umbral=args.threshold)
        print(f"✔ Empaquetado: {args.out_zip} ({n} archivos, umbral {args.threshold:.2f})")
        fmt.verificar(str(args.out_zip), str(ROOT / "spacelab_dataset/images/oculto"))


if __name__ == "__main__":
    main()
