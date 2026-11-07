"""Compara PC1, PC2 y mezclas en val_dev; exporta solo la mejor opcion medida."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from optimizar_hora_extra import ROOT, ev, fmt, wbf, calibrate, save_json, write_preds, strict_check, images, infer_tta, fuse_cache


def tta_pc1():
    work=ROOT/'runs/hora_extra_fusion'
    model=ROOT/'runs/E0_A_768_seed42/weights/best.pt'
    gt=ev.leer_gt(str(ROOT/'derived/development/labels/val_dev'))
    infer_tta(model,images(ROOT/'derived/development/images/val_dev',889),work/'pc1_views_val',work/'tta_status.json')
    candidates=[]
    for mode in ['support','mean','max']:
        pred=fuse_cache(work/'pc1_views_val',mode);cal=calibrate(pred,gt)
        candidates.append((cal['after']['total']['score'],mode,cal))
        print('PC1 TTA',mode,cal['after']['total']['score'],flush=True)
    _,mode,cal=max(candidates,key=lambda x:x[0])
    save_json(work/'pc1_tta_calibration.json',{'mode':mode,'calibration':cal})
    write_preds(work/'pc1_tta_val',fuse_cache(work/'pc1_views_val',mode))
    infer_tta(model,images(ROOT/'spacelab_dataset/images/oculto',1743),work/'pc1_views_oculto',work/'tta_status.json')
    write_preds(work/'pc1_tta_oculto',fuse_cache(work/'pc1_views_oculto',mode))


def sliced_val():
    import cv2
    import torch
    from ultralytics import YOLO
    from ensemble import predecir_imagen_sliced
    torch.set_num_threads(6);cv2.setNumThreads(2)
    out=ROOT/'runs/hora_extra_fusion/E2_sliced_val';out.mkdir(parents=True,exist_ok=True)
    files=images(ROOT/'derived/development/images/val_dev',889)
    model=YOLO(str(ROOT/'runs/E2_A_tiles512_seed42/weights/best.pt'))
    start=time.time()
    for i,p in enumerate(files):
        if (out/(p.stem+'.txt')).exists(): continue
        im=cv2.imread(str(p))
        assert im is not None,p
        dets=predecir_imagen_sliced(model,im,tile_size=512,overlap=.25)
        # The existing helper returns an extra model weight for single-tile images.
        a=np.asarray([d[:6] for d in dets],dtype=float).reshape(-1,6)
        write_one(out,p.stem,a)
        if (i+1)%100==0 or i+1==len(files): print('SLICED',i+1,len(files),'seconds',round(time.time()-start,1),flush=True)


def write_one(folder,name,arr):
    (folder/(name+'.txt')).write_text('\n'.join(f'{int(d[0])} '+' '.join(f'{v:.12g}' for v in d[1:]) for d in arr),encoding='utf-8')


def load(path):
    env=ev.leer_envio(str(path));assert not env.avisos,env.avisos
    return {k:a[a[:,5]>=.01] for k,a in env.preds.items()}


def combine(inputs,config):
    if config['kind']=='single': return inputs[config['sources'][0]]
    if config['kind']=='class_select':
        cache={}
        for cfg in config['class_configs']:
            key=json.dumps(cfg,sort_keys=True)
            if key not in cache:cache[key]=combine(inputs,cfg)
        out={}
        for name in next(iter(cache.values())):
            pieces=[]
            for c,cfg in enumerate(config['class_configs']):
                a=cache[json.dumps(cfg,sort_keys=True)][name];pieces.append(a[a[:,0]==c])
            arr=np.concatenate(pieces)
            out[name]=arr[np.argsort(-arr[:,5],kind='stable')[:300]]
        return out
    names=set(inputs[config['sources'][0]])
    assert all(set(inputs[s])==names for s in config['sources'])
    out={}
    for name in sorted(names):
        rows=[]
        for view,s in enumerate(config['sources']):
            a=inputs[s][name]
            rows.append(np.column_stack([a,np.full(len(a),view)]))
        out[name]=wbf(np.concatenate(rows),n_models=len(rows),iou_thr=config['iou'],mode=config['mode'],weights=config['weights'])
    return out


def search():
    work=ROOT/'runs/hora_extra_fusion';work.mkdir(parents=True,exist_ok=True)
    gt=ev.leer_gt(str(ROOT/'derived/development/labels/val_dev'))
    paths={'pc1_full':ROOT/'runs/E1a_infer1024_val','pc2_tta':ROOT/'preds_pc2_val'}
    sliced=work/'E2_sliced_val'
    if sliced.exists() and len(list(sliced.glob('*.txt')))==889: paths['pc1_tiles']=sliced
    tta=work/'pc1_tta_val'
    if tta.exists() and len(list(tta.glob('*.txt')))==889: paths['pc1_tta']=tta
    inputs={k:load(v) for k,v in paths.items()}
    assert all(set(x)==set(gt) for x in inputs.values())
    candidates=[{'name':k,'kind':'single','sources':[k]} for k in inputs]
    pairs=[['pc1_full','pc2_tta']]
    if 'pc1_tiles' in inputs: pairs += [['pc1_tiles','pc2_tta'],['pc1_full','pc1_tiles']]
    if 'pc1_tta' in inputs: pairs += [['pc1_tta','pc2_tta']]
    if 'pc1_tta' in inputs and 'pc1_tiles' in inputs: pairs += [['pc1_tta','pc1_tiles']]
    for pair in pairs:
        for iou in [.50,.60,.70]:
            for mode in ['support','max']:
                weights=[1,1]
                candidates.append({'name':'_'.join(pair)+f'_{iou}_{mode}','kind':'wbf','sources':pair,'iou':iou,'mode':mode,'weights':weights})
    if 'pc1_tiles' in inputs:
        for mode in ['support','max']:
            sources=['pc1_tta' if 'pc1_tta' in inputs else 'pc1_full','pc2_tta','pc1_tiles']
            candidates.append({'name':'three_'+mode,'kind':'wbf','sources':sources,'iou':.60,'mode':mode,'weights':[1,1,1]})
    reports=[];best=None;best_preds=None
    for cfg in candidates:
        pred=combine(inputs,cfg);cal=calibrate(pred,gt)
        report={'config':cfg,'calibration':cal};reports.append(report)
        score=cal['after']['total']['score']
        print('FUSION',cfg['name'],'score',round(score,6),'mAP',round(cal['after']['total']['map50_95'],6),flush=True)
        save_json(work/'comparison.json',reports)
        if best is None or score>best['calibration']['after']['total']['score']:
            best=report;best_preds=pred
    # Each class contributes equally to the official metric. Try measured class
    # specialization, then reevaluate the assembled <=300 predictions end to end.
    class_configs=[]
    for name in ev.CLASES:
        def class_score(r):
            m=r['calibration']['after']['total']['por_clase'][name]
            return .7*m['ap50_95']+.3*m['f1']
        class_configs.append(max(reports,key=class_score)['config'])
    cfg={'name':'seleccion_por_clase','kind':'class_select','class_configs':class_configs,
         'sources':sorted({s for c in class_configs for s in c['sources']})}
    pred=combine(inputs,cfg);cal=calibrate(pred,gt);report={'config':cfg,'calibration':cal}
    reports.append(report);save_json(work/'comparison.json',reports)
    print('FUSION',cfg['name'],'score',round(cal['after']['total']['score'],6),flush=True)
    if cal['after']['total']['score']>best['calibration']['after']['total']['score']:
        best=report;best_preds=pred
    save_json(work/'selected.json',best)
    write_preds(work/'selected_val_raw',best_preds)
    write_preds(work/'selected_val_calibrated',best_preds,best['calibration']['gammas'])
    check=ev.evaluar(gt,strict_check(work/'selected_val_calibrated',set(gt)),umbral=.33)
    save_json(work/'metricas_val_exportadas.json',check)
    print('BEST',best['config'],check['total'],flush=True)
    # Frozen validation choice is now applied to the hidden images.
    test_paths={'pc1_full':ROOT/'runs/E1a_1024_oculto','pc2_tta':ROOT/'preds_pc2_test','pc1_tiles':ROOT/'runs/E2_sliced_oculto',
                'pc1_tta':work/'pc1_tta_oculto'}
    needed=best['config']['sources'];test_inputs={k:load(test_paths[k]) for k in needed}
    expected={p.stem for p in images(ROOT/'spacelab_dataset/images/oculto',1743)}
    assert all(set(d)==expected for d in test_inputs.values())
    out=combine(test_inputs,best['config'])
    write_preds(work/'selected_oculto_raw',out)
    write_preds(work/'selected_oculto_calibrated',out,best['calibration']['gammas'])
    target=ROOT/'envio_mejor_validado_hora_extra.zip'
    fmt.crear_zip(str(work/'selected_oculto_calibrated'),str(target),.33)
    strict_check(target,expected)
    assert fmt.verificar(str(target),str(ROOT/'spacelab_dataset/images/oculto'))
    print('READY',target,flush=True)


if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('action',choices=['sliced-val','tta-pc1','search']);a=p.parse_args()
    {'sliced-val':sliced_val,'tta-pc1':tta_pc1,'search':search}[a.action]()
