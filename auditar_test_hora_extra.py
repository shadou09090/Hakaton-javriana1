"""Auditoria del test local etiquetado (801), con configuraciones ya congeladas.

No calibra ni selecciona hiperparametros con estas etiquetas.
"""
from pathlib import Path
import json
import sys
import time
import numpy as np
from optimizar_hora_extra import ROOT, ev, infer_tta, fuse_cache, images, save_json, write_preds, strict_check
from comparar_fusion_hora_extra import combine, load, write_one


def main():
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    import cv2
    import torch
    from ultralytics import YOLO
    from ensemble import predecir_imagen_sliced
    from generar_envio import predecir_oculto
    torch.set_num_threads(6);cv2.setNumThreads(2)
    work=ROOT/'runs/auditoria_test801';work.mkdir(parents=True,exist_ok=True)
    files=images(ROOT/'spacelab_dataset/images/test',801)
    gt=ev.leer_gt(str(ROOT/'spacelab_dataset/labels/test'))
    assert set(gt)=={p.stem for p in files}
    fusion_dir=ROOT/'runs/hora_extra_fusion'
    config=json.loads((fusion_dir/'selected.json').read_text(encoding='utf-8'))
    p2cal=json.loads((ROOT/'runs/hora_extra_pc2/selected_calibration.json').read_text(encoding='utf-8'))
    p1cal=json.loads((fusion_dir/'pc1_tta_calibration.json').read_text(encoding='utf-8'))
    # Persist exactly what is under audit before seeing any test metrics.
    save_json(work/'frozen_configurations.json',{'fusion':config,'pc2':p2cal,'pc1':p1cal,
              'leaderboard_user_report':{'score':.4655,'map50_95':.4131,'f1_macro':.5877,'rank':7},
              'test_used_for_calibration':False})
    models={'pc1':ROOT/'runs/E0_A_768_seed42/weights/best.pt','pc2':ROOT/'best_pc2_1024.pt'}
    for key,model in models.items():
        infer_tta(model,files,work/(key+'_views'),work/'status.json')
        mode=p1cal['mode'] if key=='pc1' else p2cal['wbf_conf_mode']
        write_preds(work/(key+'_tta'),fuse_cache(work/(key+'_views'),mode))
    full=work/'pc1_full'
    if not full.exists() or len(list(full.glob('*.txt')))!=801:
        predecir_oculto(models['pc1'],ROOT/'spacelab_dataset/images/test',full,imgsz=1024,batch=16,conf=.005)
    sliced=work/'pc1_tiles';sliced.mkdir(parents=True,exist_ok=True)
    model=YOLO(str(ROOT/'runs/E2_A_tiles512_seed42/weights/best.pt'));start=time.time()
    for i,p in enumerate(files):
        if not (sliced/(p.stem+'.txt')).exists():
            im=cv2.imread(str(p));assert im is not None,p
            det=predecir_imagen_sliced(model,im,tile_size=512,overlap=.25)
            write_one(sliced,p.stem,np.asarray([d[:6] for d in det],float).reshape(-1,6))
        if (i+1)%100==0 or i+1==len(files):print('SLICED TEST',i+1,len(files),'seconds',round(time.time()-start,1),flush=True)
    del model;torch.cuda.empty_cache()
    inputs={'pc1_tta':load(work/'pc1_tta'),'pc2_tta':load(work/'pc2_tta'),
            'pc1_full':load(full),'pc1_tiles':load(sliced)}
    frozen_comparison=json.loads((fusion_dir/'comparison.json').read_text(encoding='utf-8'))
    variants=[r for r in frozen_comparison if r['config']['name'] in ['pc1_full','pc1_tta','pc2_tta']]+[config]
    results={}
    for v in variants:
        name=v['config']['name'];raw=combine(inputs,v['config'])
        out=work/(name+'_calibrated');write_preds(out,raw,v['calibration']['gammas'])
        # Historical full-image TXT files round coordinates to 6 decimals, so
        # border corners can be outside by 0.5e-6. Preserve them for a faithful
        # audit with the official reader; do not change the frozen predictions.
        env=ev.leer_envio(str(out))
        assert not env.avisos and set(env.preds)==set(gt)
        assert all(len(a)<=300 and np.isfinite(a).all() for a in env.preds.values())
        res=ev.evaluar(gt,env,umbral=v['calibration']['global_threshold'])
        results[name]={'val_dev':v['calibration']['after']['total'],'test801':res['total']}
        save_json(work/'results.json',results)
        print('AUDIT',name,json.dumps(res['total'],ensure_ascii=False),flush=True)
    save_json(work/'status.json',{'phase':'completed','test_images':801,'calibration_on_test':False,'results':results})
    lines=['# Auditoría independiente del test local (801 imágenes)','',
           'Configuraciones congeladas antes de evaluar. No se recalibró ni se eligieron parámetros sobre test.','',
           '| Configuración | Score val_dev | Score test801 | mAP test801 | F1 test801 |','|---|---:|---:|---:|---:|']
    for name,r in results.items():
        m=r['test801'];lines.append(f"| {name} | {r['val_dev']['score']:.6f} | {m['score']:.6f} | {m['map50_95']:.6f} | {m['f1_macro']:.6f} |")
    lines+=['','El usuario confirmó para `envio_mejor_validado_hora_extra.zip` en el leaderboard: score 0.4655, mAP 0.4131, F1 0.5877, puesto 7.','',
            '| Clase (combinación) | AP val_dev | AP test801 | Recall test801 | F1 test801 |','|---|---:|---:|---:|---:|']
    r=results[config['config']['name']]
    for name,m in r['test801']['por_clase'].items():
        lines.append(f"| {name} | {r['val_dev']['por_clase'][name]['ap50_95']:.4f} | {m['ap50_95']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
    (ROOT/'AUDITORIA_TEST801.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('COMPLETED',ROOT/'AUDITORIA_TEST801.md',flush=True)


if __name__=='__main__':main()
