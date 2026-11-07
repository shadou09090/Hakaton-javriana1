#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fusion_pc1_pc2.py — Fusión WBF Optimizada entre PC 1 (Sliced Tiles) y PC 2 (1024px Nativo)
Evita duplicación de cajas bajando iou_thr a 0.38 y aplicando WBF robusto.
"""
from collections import defaultdict
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent
KIT_DIR = ROOT / "kit_participantes"
sys.path.insert(0, str(KIT_DIR))
import formato_envio as fmt

CLASES = ["small_boat", "cargo_ship", "fishing_vessel", "warship", "work_vessel"]

UMBRALES_OPTIMOS_CLASE = {
    0: 0.37,  # small_boat
    1: 0.33,  # cargo_ship
    2: 0.20,  # fishing_vessel
    3: 0.53,  # warship
    4: 0.48   # work_vessel
}


def calcular_gammas(t_global: float = 0.33) -> dict:
    gammas = {}
    for c, tc in UMBRALES_OPTIMOS_CLASE.items():
        tc_safe = max(0.02, min(0.95, tc))
        gammas[c] = math.log(t_global) / math.log(tc_safe)
    return gammas


def bb_iou_np(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    union = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    return inter / union if union > 0 else 0.0


def wbf_anti_duplicados(dets: list, iou_thr: float = 0.38, max_det: int = 300) -> list:
    """WBF con umbral de agrupamiento más amplio (0.38) para fusionar cajas sin duplicar falsos positivos."""
    if not dets:
        return []

    by_class = defaultdict(list)
    for c, xc, yc, w, h, s, weight in dets:
        if w >= 0.0005 and h >= 0.0005 and 0.0 <= s <= 1.0:
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
            # Si ambos modelos la vieron, mayor confianza; si solo uno, confianza moderada
            if len(c_raw) >= 2:
                final_conf = max(c_raw)
            else:
                final_conf = c_raw[0] * 0.95  # penalización ligera a detecciones aisladas de un solo modelo

            x1, y1, x2, y2 = fb
            w_box = max(0.0005, float(x2 - x1))
            h_box = max(0.0005, float(y2 - y1))
            xc_box = float((x1 + x2) / 2.0)
            yc_box = float((y1 + y2) / 2.0)
            final_dets.append((c, xc_box, yc_box, w_box, h_box, final_conf))

    final_dets.sort(key=lambda d: -d[5])
    return final_dets[:max_det]


def procesar_fusion(pc1_dir: Path, pc2_dir: Path, out_zip: Path, t_global: float = 0.33):
    gammas = calcular_gammas(t_global)
    all_names = sorted(list({f.name for f in pc1_dir.glob("*.txt") if f.name != "umbral.txt"} |
                            {f.name for f in pc2_dir.glob("*.txt") if f.name != "umbral.txt"}))
    print(f"Fusionando predicciones de PC1 ({pc1_dir.name}) y PC2 ({pc2_dir.name}) sobre {len(all_names)} imágenes...")

    pesos_pc1 = {0: 1.4, 1: 0.7, 2: 1.4, 3: 0.6, 4: 1.4}
    pesos_pc2 = {0: 0.8, 1: 1.4, 2: 0.8, 3: 1.5, 4: 0.8}

    tmp_out = ROOT / "runs/fusion_pc1_pc2_temp"
    tmp_out.mkdir(parents=True, exist_ok=True)

    total_cajas = 0
    for name in all_names:
        dets = []
        p1 = pc1_dir / name
        if p1.exists():
            for line in p1.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 6:
                    c, x, y, w, h, s = map(float, parts)
                    c = int(c)
                    dets.append((c, x, y, w, h, s, pesos_pc1.get(c, 1.0)))

        p2 = pc2_dir / name
        if p2.exists():
            for line in p2.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 6:
                    c, x, y, w, h, s = map(float, parts)
                    c = int(c)
                    dets.append((c, x, y, w, h, s, pesos_pc2.get(c, 1.0)))

        # WBF anti-duplicados
        fused = wbf_anti_duplicados(dets, iou_thr=0.38)

        # Calibración Gamma
        lines = []
        for c, x, y, w, h, s in fused:
            g = gammas.get(c, 1.0)
            s_calib = float(np.clip(s ** g, 0.005, 1.0))
            lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {s_calib:.5f}")
        
        total_cajas += len(lines)
        (tmp_out / name).write_text("\n".join(lines), encoding="utf-8")

    print(f"Total cajas fusionadas calibradas: {total_cajas} (promedio {total_cajas/len(all_names):.1f} cajas/img)")
    n = fmt.crear_zip(str(tmp_out), str(out_zip), umbral=t_global)
    print(f"✔ Empaquetado: {out_zip} ({n} archivos, umbral {t_global:.2f})")
    fmt.verificar(str(out_zip), str(ROOT / "spacelab_dataset/images/oculto"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pc1-dir", type=Path, default=ROOT / "runs/E2_sliced_oculto")
    parser.add_argument("--pc2-zip", type=Path, default=ROOT / "predicciones_pc2.zip")
    parser.add_argument("--pc2-dir", type=Path, default=ROOT / "runs/E1b_1024_oculto")
    parser.add_argument("--out-zip", type=Path, default=ROOT / "envio_spacelab_final_pc1_pc2.zip")
    parser.add_argument("--t-global", type=float, default=0.33)
    args = parser.parse_args()

    if args.pc2_zip.exists() and not args.pc2_dir.exists():
        print(f"Extrayendo {args.pc2_zip} en {args.pc2_dir}...")
        args.pc2_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.pc2_zip, "r") as z:
            z.extractall(args.pc2_dir)

    if not args.pc2_dir.exists():
        print(f"ERROR: No se encontró la carpeta de PC 2: {args.pc2_dir}")
        sys.exit(1)

    procesar_fusion(args.pc1_dir, args.pc2_dir, args.out_zip, t_global=args.t_global)

