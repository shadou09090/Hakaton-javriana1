#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generar_envio.py — Inferencia sobre imágenes ocultas y empaquetado del envío oficial SpaceLAB.
"""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
KIT_DIR = ROOT / "kit_participantes"
sys.path.insert(0, str(KIT_DIR))

import formato_envio as fmt
from ultralytics import YOLO
import torch


def predecir_oculto(model_path: Path, images_dir: Path, out_dir: Path, imgsz: int = 768, batch: int = 16, conf: float = 0.001):
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.name.startswith(".")])
    print(f"Total imágenes en conjunto oculto: {len(images)}")
    
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Cargando modelo: {model_path} en dispositivo: {device}")
    model = YOLO(str(model_path))
    
    start_time = time.time()
    n_processed = 0
    total_detections = 0
    
    for i in range(0, len(images), batch):
        batch_paths = images[i:i + batch]
        results = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=imgsz,
            conf=conf,
            iou=0.6,
            max_det=300,
            device=device,
            verbose=False
        )
        for p, r in zip(batch_paths, results):
            boxes = r.boxes
            if len(boxes) > 0:
                coords = boxes.xywhn.tolist()
                classes = boxes.cls.tolist()
                confs = boxes.conf.tolist()
                lines = [
                    f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {s:.5f}"
                    for (x, y, w, h), c, s in zip(coords, classes, confs)
                    if w >= 0.0005 and h >= 0.0005
                ]
                total_detections += len(lines)
            else:
                lines = []
            
            txt_path = out_dir / f"{p.stem}.txt"
            txt_path.write_text("\n".join(lines), encoding="utf-8")
            n_processed += 1
            
        if n_processed % 300 == 0 or n_processed == len(images):
            elapsed = time.time() - start_time
            fps = n_processed / max(elapsed, 0.001)
            print(f"  Procesadas {n_processed}/{len(images)} imágenes ({fps:.1f} img/s)...")
            
    total_time = time.time() - start_time
    print(f"Inferencia completada en {total_time:.1f}s ({n_processed} archivos generados, {total_detections} detecciones totales).")
    return n_processed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/E0_A_768_seed42/weights/best.pt")
    parser.add_argument("--images", type=Path, default=ROOT / "spacelab_dataset/images/oculto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/E0_A_768_seed42/pred_oculto")
    parser.add_argument("--output-zip", type=Path, default=ROOT / "runs/E0_A_768_seed42/envio_spacelab.zip")
    parser.add_argument("--threshold", type=float, default=0.36)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    if not args.weights.exists():
        sys.exit(f"Error: no se encontró el modelo en {args.weights}")
    if not args.images.exists():
        sys.exit(f"Error: no se encontró la carpeta de imágenes en {args.images}")

    print("=== PASO 1: Inferencia sobre imágenes de oculto ===")
    predecir_oculto(args.weights, args.images, args.output_dir, imgsz=args.imgsz, batch=args.batch)

    print("\n=== PASO 2: Empaquetando ZIP oficial con umbral ===")
    n_zip = fmt.crear_zip(str(args.output_dir), str(args.output_zip), umbral=args.threshold)
    print(f"✔ Creado: {args.output_zip} con {n_zip} archivos + umbral.txt ({args.threshold:.2f})")

    print("\n=== PASO 3: Verificación oficial del ZIP ===")
    es_valido = fmt.verificar(str(args.output_zip), str(args.images))
    if not es_valido:
        sys.exit("Error: el archivo generado no pasó la verificación oficial.")
        
    print(f"\n🚀 ¡ÉXITO! El archivo listo para subir al Leaderboard es:\n{args.output_zip.resolve()}")


if __name__ == "__main__":
    main()

