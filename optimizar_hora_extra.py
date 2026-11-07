"""Inferencia reproducible PC2: TTA exacta, WBF y calibracion solo en val_dev.

Los nombres preds_pc2_test corresponden a images/oculto (1743 imagenes).
No se leen las etiquetas del test local ni se modifica ningun entrenamiento.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'kit_participantes'))
import evaluar as ev
import formato_envio as fmt


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, path)


def images(folder, expected):
    result = sorted(p for p in folder.iterdir() if p.suffix.lower() in fmt.EXT_IMG)
    assert len(result) == expected, (folder, len(result), expected)
    assert len({p.stem for p in result}) == expected
    return result


def undo_flip(rows, view):
    rows = rows.copy()
    if view == 1:
        rows[:, 1] = 1 - rows[:, 1]
    if view == 2:
        rows[:, 2] = 1 - rows[:, 2]
    return rows


def wbf(rows, n_models=3, iou_thr=.60, mode='support', weights=None):
    """Rows: class,xc,yc,w,h,confidence,view. Best-IoU clustering per class.

    Coordinates use confidence*model weight. support penalizes boxes seen by
    fewer distinct views; max and mean are validation ablations.
    """
    if not len(rows):
        return np.zeros((0, 6))
    weights = np.ones(n_models) if weights is None else np.asarray(weights)
    result = []
    for c in range(5):
        part = rows[rows[:, 0] == c]
        if not len(part):
            continue
        eff = part[:, 5] * weights[part[:, 6].astype(int)]
        order = np.argsort(-eff, kind='stable')
        centers = np.zeros((len(part), 4)); sums = np.zeros((len(part), 4))
        sums_w = np.zeros(len(part)); counts = np.zeros(len(part))
        raw_sum = np.zeros(len(part)); model_sum = np.zeros(len(part))
        maxima = np.zeros(len(part)); supports = [set() for _ in part]
        size = 0
        for ix in order:
            d = part[ix]; box = np.array([d[1]-d[3]/2,d[2]-d[4]/2,d[1]+d[3]/2,d[2]+d[4]/2])
            box = np.clip(box, 0, 1)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            match = -1
            if size:
                b = centers[:size]
                wh = np.maximum(0, np.minimum(b[:, 2:], box[2:])-np.maximum(b[:, :2], box[:2]))
                inter = wh[:, 0]*wh[:, 1]
                union = (b[:,2]-b[:,0])*(b[:,3]-b[:,1]) + (box[2]-box[0])*(box[3]-box[1])-inter
                ious = inter/np.maximum(union, 1e-12)
                best = int(ious.argmax())
                if ious[best] >= iou_thr:
                    match = best
            if match < 0:
                match = size; size += 1
            mw = weights[int(d[6])]
            sums[match] += box*eff[ix]; sums_w[match] += eff[ix]
            centers[match] = sums[match]/sums_w[match]
            counts[match] += 1; raw_sum[match] += d[5]*mw; model_sum[match] += mw
            maxima[match] = max(maxima[match], d[5]); supports[match].add(int(d[6]))
        for k in range(size):
            x1,y1,x2,y2 = centers[k]
            s = maxima[k] if mode == 'max' else raw_sum[k]/model_sum[k]
            if mode == 'support':
                s *= sum(weights[v] for v in supports[k])/sum(weights)
            if s >= .01 and min(x2-x1, y2-y1) >= .0005:
                result.append([c,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1,s])
    arr = np.asarray(result, dtype=np.float64).reshape(-1, 6)
    return arr[np.argsort(-arr[:, 5], kind='stable')[:300]]


def write_preds(folder, preds, gammas=None):
    folder.mkdir(parents=True, exist_ok=True)
    assert not ({p.stem for p in folder.glob('*.txt')} - set(preds)), 'Unexpected old files'
    for name, arr in preds.items():
        rows = arr.copy()
        if gammas is not None and len(rows):
            rows[:,5] = rows[:,5] ** np.asarray(gammas)[rows[:,0].astype(int)]
        # Keep the same <=300 candidates before/after gamma: class AP ranking is preserved.
        text = '\n'.join(f'{int(d[0])} ' + ' '.join(f'{v:.12g}' for v in d[1:]) for d in rows)
        (folder / (name+'.txt')).write_text(text, encoding='utf-8')


def calibrate(preds, gt, global_t=.33):
    assert set(preds) == set(gt)
    pc = ev.preparar(gt, ev.Envio(preds=preds))
    before = ev.metricas(pc, global_t)
    grid = np.round(np.arange(.01,.911,.01), 2)
    gammas, thresholds, low_recall = [], [], []
    for c, k in enumerate(pc):
        n_gt = sum(k.n_gt_img.values())
        order = np.argsort(-k.conf, kind='stable')
        conf = k.conf[order]; tp = np.r_[0,np.cumsum(k.tp[0,order])]
        low = before['por_clase'][ev.CLASES[c]]['recall'] < .5
        candidates = grid[grid <= global_t] if low else grid
        n = np.searchsorted(-conf, -candidates, side='right')
        f1 = 2*tp[n]/np.maximum(n+n_gt,1)
        # Prefer a threshold close to uncalibrated .33 in exact ties.
        best = np.flatnonzero(f1 >= f1.max()-1e-12)
        idx = best[np.argmin(abs(candidates[best]-global_t))]
        # Map equality slightly above the global boundary to survive TXT rounding.
        threshold = float(candidates[idx]); gamma = math.log(global_t+1e-9)/math.log(threshold)
        assert not low or gamma <= 1+1e-12
        thresholds.append(threshold); gammas.append(gamma); low_recall.append(low)
    transformed = {}
    for name, arr in preds.items():
        a = arr.copy()
        if len(a): a[:,5] **= np.asarray(gammas)[a[:,0].astype(int)]
        transformed[name] = a
    after = ev.evaluar(gt, ev.Envio(preds=transformed), umbral=global_t)
    # FP64 powers can merge nearly identical confidence ties; verify the actual metric.
    assert abs(after['total']['map50_95']-before['map50_95']) < 1e-5
    assert after['total']['score'] >= before['score']-1e-9
    return {'global_threshold':global_t,'class_thresholds':thresholds,'gammas':gammas,
            'low_recall_at_033':low_recall,'low_recall_definition':'recall < 0.50 at threshold 0.33',
            'before':before,'after':after,'map_delta':after['total']['map50_95']-before['map50_95'],
            'optimization':'class thresholds 0.01..0.91 step 0.01; fixed candidates; val_dev only'}


def infer_tta(model_path, files, cache_dir, status_path):
    import cv2
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(6); cv2.setNumThreads(2)
    assert torch.cuda.is_available(), 'CUDA required'
    torch.backends.cudnn.benchmark = False
    model = YOLO(str(model_path))
    assert list(model.names.values()) == ev.CLASES, model.names
    ckpt = model.ckpt or {}; ta = ckpt.get('train_args',{})
    print('MODEL', {k:ta.get(k) for k in ['name','epochs','imgsz','data','model']}, 'epoch',ckpt.get('epoch'),flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, p in enumerate(files):
        out = cache_dir/(p.stem+'.npz')
        if out.exists(): continue
        im = cv2.imread(str(p))
        if im is None: raise RuntimeError(f'Unreadable image: {p}')
        views = [im, cv2.flip(im,1), cv2.flip(im,0)]
        rr = model.predict(views, imgsz=1024, conf=.01, iou=.7, max_det=300,
                           device=0,quantize=16,verbose=False,rect=True,augment=False)
        rows = []
        for v, r in enumerate(rr):
            b = r.boxes
            a = np.column_stack([b.cls.cpu().numpy(), b.xywhn.cpu().numpy(), b.conf.cpu().numpy()]).astype(np.float64)
            a = undo_flip(a,v)
            rows.append(np.column_stack([a,np.full(len(a),v)]))
        np.savez_compressed(out, rows=np.concatenate(rows))
        if (i+1)%50 == 0 or i+1 == len(files):
            rate=(i+1)/max(time.time()-t0,.001)
            print(f'TTA {cache_dir.name} {i+1}/{len(files)} {rate:.2f} images/s',flush=True)
            save_json(status_path,{'phase':'tta','split':cache_dir.name,'done':i+1,'total':len(files),'seconds':time.time()-t0})
    del model
    torch.cuda.empty_cache()


def fuse_cache(cache_dir, mode):
    return {p.stem:wbf(np.load(p)['rows'],mode=mode) for p in sorted(cache_dir.glob('*.npz'))}


def strict_check(path, expected, floor=None):
    env = ev.leer_envio(str(path))
    assert not env.avisos, env.avisos
    assert set(env.preds) == expected, (len(env.preds),len(expected))
    for name, a in env.preds.items():
        assert len(a)<=300 and np.isfinite(a).all(), name
        if len(a):
            assert np.all(a[:,0]==a[:,0].astype(int)) and np.all((a[:,0]>=0)&(a[:,0]<5)),name
            assert np.all((a[:,1:6]>=0)&(a[:,1:6]<=1)),name
            assert np.all(a[:,3:5]>=.0005-1e-10),name
            assert np.all(a[:,1:3]-a[:,3:5]/2>=-1e-9),name
            assert np.all(a[:,1:3]+a[:,3:5]/2<=1+1e-9),name
            if floor is not None: assert np.all(a[:,5]>=floor-1e-12),name
    return env


def selftest():
    a=np.array([[0,.2,.3,.1,.2,.8]])
    assert np.allclose(undo_flip(undo_flip(a,1),1),a)
    assert np.allclose(undo_flip(undo_flip(a,2),2),a)
    rows=np.array([[0,.5,.5,.2,.2,.8,0],[0,.52,.5,.2,.2,.4,1]])
    f=wbf(rows,n_models=2)
    assert len(f)==1 and np.isclose(f[0,1],(.5*.8+.52*.4)/1.2) and np.isclose(f[0,5],.6)
    assert np.isclose(wbf(rows[:1],n_models=2)[0,5],.4)
    assert len(wbf(np.zeros((0,7))))==0
    gt={}; preds={}
    for c in range(5):
        gt[str(c)]=np.array([[c,.5,.5,.2,.2]])
        preds[str(c)]=np.array([[c,.5,.5,.2,.2,.2]])
    cal=calibrate(preds,gt)
    assert max(cal['gammas'])<=1 and cal['after']['total']['f1_macro']==1
    print('SELFTEST OK',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--selftest',action='store_true');args=parser.parse_args()
    if args.selftest: selftest();return
    work=ROOT/'runs/hora_extra_pc2';work.mkdir(parents=True,exist_ok=True)
    model=ROOT/'best_pc2_1024.pt'; source=ROOT/'best.pt.zip'
    if not model.exists(): shutil.copy2(source,model)
    digest=hashlib.sha256(model.read_bytes()).hexdigest()
    metadata_path=work/'provenance.json'
    if metadata_path.exists(): assert json.loads(metadata_path.read_text())['model_sha256']==digest
    save_json(metadata_path,{'model_sha256':digest,'source':str(source),'imgsz':1024,'tta':['original','hflip','vflip'],
                            'iou_wbf':.60,'raw_conf_min':.01,'device':'cuda:0','test_means':'spacelab_dataset/images/oculto',
                            'pc1_manifest_sha256':hashlib.sha256((ROOT/'manifiesto_A.json').read_bytes()).hexdigest(),
                            'pc2_training_split_independently_verified':False})
    val=images(ROOT/'derived/development/images/val_dev',889)
    test=images(ROOT/'spacelab_dataset/images/oculto',1743)
    gt=ev.leer_gt(str(ROOT/'derived/development/labels/val_dev'))
    assert set(gt)=={p.stem for p in val}
    infer_tta(model,val,work/'views_val',work/'status.json')
    candidates=[]
    for mode in ['support','mean','max']:
        preds=fuse_cache(work/'views_val',mode)
        cal=calibrate(preds,gt);cal['wbf_conf_mode']=mode
        save_json(work/f'calibration_{mode}.json',cal)
        print('CANDIDATE',mode,cal['after']['total']['score'],'mAP',cal['after']['total']['map50_95'], 'thresholds',cal['class_thresholds'],flush=True)
        candidates.append(cal)
    chosen=max(candidates,key=lambda c:c['after']['total']['score'])
    save_json(work/'selected_calibration.json',chosen)
    val_preds=fuse_cache(work/'views_val',chosen['wbf_conf_mode'])
    write_preds(ROOT/'preds_pc2_val',val_preds)
    write_preds(work/'calibrated_val',val_preds,chosen['gammas'])
    reloaded=strict_check(work/'calibrated_val',set(gt))
    verified=ev.evaluar(gt,reloaded,umbral=chosen['global_threshold'])
    save_json(work/'metricas_val_exportadas.json',verified)
    assert abs(verified['total']['score']-chosen['after']['total']['score'])<1e-5
    infer_tta(model,test,work/'views_oculto',work/'status.json')
    test_preds=fuse_cache(work/'views_oculto',chosen['wbf_conf_mode'])
    write_preds(ROOT/'preds_pc2_test',test_preds)
    write_preds(work/'calibrated_oculto',test_preds,chosen['gammas'])
    strict_check(ROOT/'preds_pc2_val',set(gt),.01)
    strict_check(ROOT/'preds_pc2_test',{p.stem for p in test},.01)
    submission=ROOT/'envio_solo_pc2_1024_wbf.zip'
    fmt.crear_zip(str(work/'calibrated_oculto'),str(submission),chosen['global_threshold'])
    strict_check(submission,{p.stem for p in test})
    assert fmt.verificar(str(submission),str(ROOT/'spacelab_dataset/images/oculto'))
    transfer=ROOT/'transferencia_pc2.zip'
    with zipfile.ZipFile(transfer,'w',zipfile.ZIP_DEFLATED) as z:
        for directory in [ROOT/'preds_pc2_val',ROOT/'preds_pc2_test']:
            for p in sorted(directory.glob('*.txt')): z.write(p,f'{directory.name}/{p.name}')
        z.write(model,'best_pc2_1024.pt')
        z.write(metadata_path,'provenance.json')
        z.write(work/'selected_calibration.json','calibration_val.json')
    with zipfile.ZipFile(transfer) as z:
        assert z.testzip() is None
        assert sum(n.startswith('preds_pc2_val/') for n in z.namelist())==889
        assert sum(n.startswith('preds_pc2_test/') for n in z.namelist())==1743
    save_json(work/'status.json',{'phase':'completed','submission':str(submission),'transfer':str(transfer),'validation':verified['total']})
    print('COMPLETED',submission,transfer,flush=True)


if __name__=='__main__': main()
