"""Ajuste final acotado por reloj; solo train_dev, conservando todos los respaldos."""
from pathlib import Path
from datetime import datetime, timezone
import os
import sys
import time
import json
import hashlib

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/.ultralytics'))


def main():
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    import torch
    from ultralytics import YOLO
    from optimizar_hora_extra import save_json
    torch.set_num_threads(6);torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available()
    out=ROOT/'runs/E3_final_pc2_refine_1024'
    assert not out.exists(), 'Use a new run name; never overwrite a training run'
    out.mkdir(parents=True)
    cutoff=datetime(2026,11,7,22,17,0,tzinfo=timezone.utc).timestamp()
    start=time.time();last=[0.0]
    source=ROOT/'best_pc2_1024.pt'
    state={'phase':'initializing','pid':os.getpid(),'started':start,'cutoff_utc':'2026-11-07T22:17:00Z',
           'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'train_split':'train_dev','val_split':'val_dev'}
    def status(trainer=None,phase='training'):
        state.update(phase=phase,updated=time.time(),elapsed=time.time()-start)
        if trainer is not None:
            state.update(epoch=trainer.epoch+1,epochs=trainer.epochs,
                         metrics={k:float(v) for k,v in trainer.metrics.items()},
                         gpu_peak_gb=torch.cuda.max_memory_allocated()/1024**3)
        save_json(out/'status.json',state)
    def tick(trainer):
        if time.time()>=cutoff:trainer.stop=True
        if time.time()-last[0]>25:
            status(trainer);last[0]=time.time()
    def epoch_done(trainer):
        status(trainer);print('EPOCH_COMPLETE',json.dumps(state),flush=True)
    try:
        model=YOLO(str(source))
        model.add_callback('on_train_batch_end',tick)
        model.add_callback('on_fit_epoch_end',epoch_done)
        model.train(data=str(ROOT/'derived/development/data.yaml'),device=0,
                    project=str(ROOT/'runs'),name=out.name,exist_ok=True,
                    epochs=6,time=None,imgsz=1024,batch=8,workers=4,amp=True,cache=False,
                    optimizer='AdamW',lr0=.0002,lrf=.15,cos_lr=True,weight_decay=.0005,
                    warmup_epochs=.25,warmup_bias_lr=.0002,
                    box=8.5,cls=.7,dfl=1.5,seed=43,deterministic=False,
                    mosaic=0,mixup=0,close_mosaic=0,degrees=0,scale=.25,translate=.05,
                    fliplr=.5,flipud=.5,hsv_h=.015,hsv_s=.5,hsv_v=.3,
                    save=True,save_period=1,plots=False,patience=10)
        state['best']=str(model.trainer.best);state['last']=str(model.trainer.last)
        assert Path(state['best']).exists()
        status(phase='completed');print('TRAINING_COMPLETED',json.dumps(state),flush=True)
    except Exception as e:
        state['error']=repr(e);status(phase='failed');raise


if __name__=='__main__':main()
