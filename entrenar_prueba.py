"""Entrenar y evaluar YOLO sobre validación de desarrollo con el evaluador oficial."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT/"runs/.ultralytics"))


def guardar(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"derived/development/data.yaml")
    parser.add_argument("--kit-root", type=Path, default=ROOT/"kit_participantes")
    parser.add_argument("--output-root", type=Path, default=ROOT/"runs")
    parser.add_argument("--name", default="E0_A_768_seed42")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    import torch
    import yaml
    import psutil
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible. Revisar el entorno antes de entrenar.")
    torch.set_num_threads(min(8, max(1, (os.cpu_count() or 4)//2)))
    torch.backends.cudnn.benchmark = False
    output = args.output_root.resolve()/args.name
    if output.exists() and not args.resume:
        raise FileExistsError(f"{output} ya existe. Usa otro nombre o --resume.")
    output.mkdir(parents=True, exist_ok=True)
    began = time.time()
    status = dict(phase="initializing", pid=os.getpid(), started=time.time(), epoch=0,
                  epochs=args.epochs, name=args.name, config=vars(args),
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  python=sys.version, workers=args.workers)
    def update(phase, **kw):
        status.update(phase=phase, updated=time.time(), elapsed_seconds=time.time()-began, **kw)
        guardar(output/"status.json", status)

    try:
        test = torch.randn((512, 512), device="cuda")
        _ = test @ test
        torch.cuda.synchronize()
        del test
        manifest = args.data.resolve().parent/"manifest.json"
        status["split_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (output/"requirements-lock.txt").write_text(packages, encoding="utf-8")
        model = YOLO(str(args.resume) if args.resume else args.model)
        weight_path = Path(model.ckpt_path)
        status["initial_weights_sha256"] = hashlib.sha256(weight_path.read_bytes()).hexdigest()
        update("training")

        def epoch_done(trainer):
            update("training", epoch=trainer.epoch+1, epochs=trainer.epochs,
                   metrics={k: float(v) for k, v in trainer.metrics.items()},
                   gpu_peak_gb=torch.cuda.max_memory_allocated()/1024**3,
                   ram_used_gb=psutil.virtual_memory().used/1024**3)

        last_heartbeat = [0.0]
        def heartbeat(trainer):
            if time.time()-last_heartbeat[0] > 20:
                last_heartbeat[0] = time.time()
                update("training", epoch=trainer.epoch,
                       gpu_peak_gb=torch.cuda.max_memory_allocated()/1024**3)

        model.add_callback("on_fit_epoch_end", epoch_done)
        model.add_callback("on_train_batch_end", heartbeat)
        if args.resume:
            model.train(resume=True)
        else:
            model.train(data=str(args.data.resolve()), device=0, epochs=args.epochs,
                        imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                        amp=True, cache=False, seed=42, deterministic=True,
                        project=str(args.output_root.resolve()), name=args.name, exist_ok=True,
                        patience=8, cos_lr=True, close_mosaic=5 if args.epochs > 5 else 0,
                        fliplr=0.5, flipud=0.5, mosaic=0.5, mixup=0.0,
                        save=True, save_period=5, plots=True)
        best = Path(model.trainer.best)
        if not best.exists():
            raise RuntimeError("No se produjo el checkpoint best.pt")
        update("evaluating", best=str(best))
        data = yaml.safe_load(args.data.read_text(encoding="utf-8"))
        base = Path(data["path"])
        val = Path(data["val"])
        if not val.is_absolute():
            val = base/val
        images = [Path(p) for p in val.read_text(encoding="utf-8").splitlines()] if val.suffix == ".txt" else sorted(val.glob("*.jpg"))
        sys.path.insert(0, str(args.kit_root.resolve()))
        import evaluar as ev
        import formato_envio as fmt
        pred = output/"pred_val"
        pred.mkdir(exist_ok=True)
        model = YOLO(str(best))
        for start in range(0, len(images), args.batch):
            part = images[start:start+args.batch]
            results = model.predict([str(p) for p in part], imgsz=args.imgsz,
                                    conf=0.001, max_det=300, device=0, verbose=False)
            for path, result in zip(part, results):
                boxes = result.boxes
                dets = [(int(c), *xy, score) for c, xy, score in zip(
                    boxes.cls.tolist(), boxes.xyxy.tolist(), boxes.conf.tolist())]
                fmt.escribir_txt(str(pred/(path.stem+".txt")), dets, result.orig_shape[1], result.orig_shape[0])
            update("evaluating", evaluated=min(start+args.batch, len(images)), validation_images=len(images))
        label_dirs = {p.parent.parent.parent/"labels"/p.parent.name for p in images}
        gt = {}
        for folder in label_dirs:
            gt.update(ev.leer_gt(str(folder)))
        ids = {p.stem for p in images}
        gt = {key: value for key, value in gt.items() if key in ids}
        if set(gt) != ids:
            raise ValueError("No coinciden GT e imágenes de validación")
        envio = ev.leer_envio(str(pred))
        threshold, _ = ev.barrer_umbral(ev.preparar(gt, envio))
        metrics = ev.evaluar(gt, envio, ev.leer_meta(None, gt), umbral=threshold)
        guardar(output/"metricas_val.json", metrics)
        fmt.crear_zip(str(pred), str(output/"predicciones_val.zip"), umbral=threshold)
        update("completed", score=metrics["total"]["score"], threshold=threshold)
        print(json.dumps(metrics["total"], indent=2), flush=True)
    except Exception:
        update("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
