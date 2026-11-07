"""Seleccion y entrega final: modelos fijados, validacion real y respaldo verificado."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time
import zipfile
import hashlib
import numpy as np
from optimizar_hora_extra import ROOT, ev, fmt, calibrate, save_json, write_preds, strict_check, images, infer_tta, fuse_cache
from comparar_fusion_hora_extra import combine, load

WORK=ROOT/'runs/final26'


def suppress(preds,threshold):
    if threshold is None:return preds
    out={}
    for name,a in preds.items():
        if len(a)==0:out[name]=a;continue
        a=a[np.argsort(-a[:,5],kind='stable')]
        b=np.column_stack([a[:,1:3]-a[:,3:5]/2,a[:,1:3]+a[:,3:5]/2])
        alive=np.ones(len(a),bool)
        for i in range(len(a)):
            if not alive[i]:continue
            ids=np.flatnonzero(alive & (a[:,0]!=a[i,0]) & (np.arange(len(a))>i))
            if not len(ids):continue
            wh=np.maximum(0,np.minimum(b[ids,2:],b[i,2:])-np.maximum(b[ids,:2],b[i,:2]))
            inter=wh[:,0]*wh[:,1]
            area=a[ids,3]*a[ids,4]+a[i,3]*a[i,4]-inter
            alive[ids[inter/np.maximum(area,1e-12)>threshold]]=False
        out[name]=a[alive]
    return out


def apply_gamma(preds,gammas):
    out={}
    for k,arr in preds.items():
        a=arr.copy()
        if len(a):a[:,5] **=np.asarray(gammas)[a[:,0].astype(int)]
        out[k]=a
    return out


def eval_preds(preds,gt):
    assert set(preds)==set(gt)
    return ev.evaluar(gt,ev.Envio(preds=preds),umbral=.33)['total']


def export(preds,record,name):
    directory=WORK/(name+'_oculto');write_preds(directory,preds)
    expected={p.stem for p in images(ROOT/'spacelab_dataset/images/oculto',1743)}
    candidate=ROOT/(name+'.zip')
    fmt.crear_zip(str(directory),str(candidate),.33)
    strict_check(candidate,expected)
    assert fmt.verificar(str(candidate),str(ROOT/'spacelab_dataset/images/oculto'))
    with zipfile.ZipFile(candidate) as z:assert z.testzip() is None
    # Every replacement is a completed verified ZIP; never expose a partial final file.
    temp=ROOT/'entrega_final.tmp';shutil.copy2(candidate,temp);temp.replace(ROOT/'entrega_final.zip')
    record['zip']=str(candidate);record['final_sha256']=hashlib.sha256(candidate.read_bytes()).hexdigest()
    save_json(WORK/'final_selected.json',record)
    print('FINAL_READY',json.dumps(record,ensure_ascii=False),flush=True)


def baseline():
    WORK.mkdir(parents=True,exist_ok=True)
    gtval=ev.leer_gt(str(ROOT/'derived/development/labels/val_dev'))
    gttest=ev.leer_gt(str(ROOT/'spacelab_dataset/labels/test'))
    val=ev.leer_envio(str(ROOT/'runs/hora_extra_fusion/selected_val_calibrated')).preds
    test=ev.leer_envio(str(ROOT/'runs/auditoria_test801/seleccion_por_clase_calibrated')).preds
    baseval=eval_preds(val,gtval);basetest=eval_preds(test,gttest)
    best={'kind':'baseline_postprocess','nms':None,'val':baseval,'test801':basetest}
    rows=[]
    for t in [None,.65,.80,.90]:
        v=eval_preds(suppress(val,t),gtval)
        rows.append({'nms':t,'val':v})
        print('BACKUP_NMS',t,v['score'],flush=True)
    choice=max(rows,key=lambda r:r['val']['score'])
    test_result=eval_preds(suppress(test,choice['nms']),gttest)
    choice['test801']=test_result
    if choice['val']['score']>baseval['score'] and test_result['score']>=basetest['score']:
        best.update(choice)
    save_json(WORK/'backup_comparison.json',{'candidates_val':rows,'chosen':best,'baseline_test801':basetest})
    hidden=ev.leer_envio(str(ROOT/'envio_mejor_validado_hora_extra.zip')).preds
    export(suppress(hidden,best['nms']),best,'envio_final_respaldo_validado')


def run():
    WORK.mkdir(parents=True,exist_ok=True)
    model=ROOT/'runs/E3_final_pc2_refine_1024/weights/best.pt'
    assert model.exists(),model
    gt=ev.leer_gt(str(ROOT/'derived/development/labels/val_dev'))
    infer_tta(model,images(ROOT/'derived/development/images/val_dev',889),WORK/'new_views_val',WORK/'status.json')
    modes=[]
    for mode in ['support','max']:
        pred=fuse_cache(WORK/'new_views_val',mode);cal=calibrate(pred,gt)
        modes.append({'mode':mode,'calibration':cal})
    chosen=max(modes,key=lambda c:c['calibration']['after']['total']['score'])
    save_json(WORK/'new_calibration.json',chosen)
    newval=fuse_cache(WORK/'new_views_val',chosen['mode'])
    write_preds(WORK/'new_val',newval)
    inputs={'new':newval,'pc1_tta':load(ROOT/'runs/hora_extra_fusion/pc1_tta_val'),
            'pc2_tta':load(ROOT/'preds_pc2_val'),'previous':load(ROOT/'runs/hora_extra_fusion/selected_val_raw')}
    configs=[{'name':s,'kind':'single','sources':[s]} for s in inputs]
    for other in ['pc1_tta','pc2_tta','previous']:
        for iou in [.6,.7]:
            for mode in ['support','max']:
                configs.append({'name':f'new_{other}_{iou}_{mode}','kind':'wbf','sources':['new',other],
                                'iou':iou,'mode':mode,'weights':[1,1]})
    for iou in [.6,.7]:
        configs.append({'name':f'new_three_{iou}','kind':'wbf','sources':['new','pc1_tta','pc2_tta'],
                        'iou':iou,'mode':'support','weights':[1,1,1]})
    reports=[];best=None;best_pred=None
    for cfg in configs:
        pred=combine(inputs,cfg);cal=calibrate(pred,gt);r={'config':cfg,'calibration':cal}
        reports.append(r);print('NEW_CANDIDATE',cfg['name'],cal['after']['total']['score'],flush=True)
        save_json(WORK/'comparison_val.json',reports)
        if best is None or cal['after']['total']['score']>best['calibration']['after']['total']['score']:
            best=r;best_pred=pred
    class_configs=[]
    for name in ev.CLASES:
        def value(r):
            c=r['calibration']['after']['total']['por_clase'][name];return .7*c['ap50_95']+.3*c['f1']
        class_configs.append(max(reports,key=value)['config'])
    cfg={'name':'new_class_selection','kind':'class_select','class_configs':class_configs,
         'sources':sorted({s for c in class_configs for s in c['sources']})}
    pred=combine(inputs,cfg);cal=calibrate(pred,gt);r={'config':cfg,'calibration':cal};reports.append(r)
    if cal['after']['total']['score']>best['calibration']['after']['total']['score']:best=r;best_pred=pred
    save_json(WORK/'comparison_val.json',reports)
    calibrated=apply_gamma(best_pred,best['calibration']['gammas'])
    nms_options=[{'nms':t,'val':eval_preds(suppress(calibrated,t),gt)} for t in [None,.65,.80,.90]]
    post=max(nms_options,key=lambda r:r['val']['score'])
    best.update(post);save_json(WORK/'new_selected_val.json',best)
    backup=json.loads((WORK/'final_selected.json').read_text(encoding='utf-8'))
    print('NEW_VAL_SELECTED',best['config']['name'],best['val']['score'],'backup',backup['val']['score'],flush=True)
    if best['val']['score']<=backup['val']['score']:
        save_json(WORK/'status.json',{'phase':'completed','decision':'keep_backup_val','backup':backup});return
    # Verify the frozen new candidate on local test before replacing the deliverable.
    infer_tta(model,images(ROOT/'spacelab_dataset/images/test',801),WORK/'new_views_test801',WORK/'status.json')
    newtest=fuse_cache(WORK/'new_views_test801',chosen['mode']);write_preds(WORK/'new_test801',newtest)
    test_inputs={'new':newtest,'pc1_tta':load(ROOT/'runs/auditoria_test801/pc1_tta'),
                 'pc2_tta':load(ROOT/'runs/auditoria_test801/pc2_tta')}
    # The previous calibrated audit is inverted using its frozen gamma parameters.
    old=json.loads((ROOT/'runs/hora_extra_fusion/selected.json').read_text(encoding='utf-8'))
    oldtest=ev.leer_envio(str(ROOT/'runs/auditoria_test801/seleccion_por_clase_calibrated')).preds
    test_inputs['previous']=apply_gamma(oldtest,1/np.asarray(old['calibration']['gammas']))
    gttest=ev.leer_gt(str(ROOT/'spacelab_dataset/labels/test'))
    # Preselect at most three alternatives by val_dev before the test comparison.
    # Test801 is a comparison gate here, not an untouched estimate of final accuracy.
    alternatives=[best]
    for r in sorted(reports,key=lambda r:r['calibration']['after']['total']['score'],reverse=True):
        if len(alternatives)>=3:break
        if r['config']['name']==best['config']['name']:continue
        if r['calibration']['after']['total']['score']<=backup['val']['score']:continue
        alternatives.append(dict(r,nms=None,val=r['calibration']['after']['total']))
    audited=[]
    for r in alternatives:
        tc=apply_gamma(combine(test_inputs,r['config']),r['calibration']['gammas'])
        tc=suppress(tc,r['nms']);tr=eval_preds(tc,gttest)
        r['test801']=tr;audited.append(r)
        print('TEST_COMPARISON',r['config']['name'],tr['score'],flush=True)
    best=max(audited,key=lambda r:r['test801']['score'])
    save_json(WORK/'test_comparison.json',audited)
    testcal=apply_gamma(combine(test_inputs,best['config']),best['calibration']['gammas'])
    testcal=suppress(testcal,best['nms']);test_result=best['test801']
    write_preds(WORK/'candidate_test801_calibrated',testcal)
    best['test801']=test_result;save_json(WORK/'new_selected_test.json',best)
    print('NEW_TEST',test_result['score'],'backup',backup['test801']['score'],flush=True)
    if test_result['score']<backup['test801']['score']:
        save_json(WORK/'status.json',{'phase':'completed','decision':'keep_backup_test','new':best,'backup':backup});return
    infer_tta(model,images(ROOT/'spacelab_dataset/images/oculto',1743),WORK/'new_views_oculto',WORK/'status.json')
    newhidden=fuse_cache(WORK/'new_views_oculto',chosen['mode']);write_preds(WORK/'new_oculto',newhidden)
    hidden_inputs={'new':newhidden,'pc1_tta':load(ROOT/'runs/hora_extra_fusion/pc1_tta_oculto'),
                   'pc2_tta':load(ROOT/'preds_pc2_test'),'previous':load(ROOT/'runs/hora_extra_fusion/selected_oculto_raw')}
    hidden=apply_gamma(combine(hidden_inputs,best['config']),best['calibration']['gammas'])
    hidden=suppress(hidden,best['nms'])
    best['kind']='refined_model';best['model']=str(model)
    export(hidden,best,'envio_final_refinado')
    save_json(WORK/'status.json',{'phase':'completed','decision':'refined_selected','selected':best})


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('action',choices=['baseline','run']);a=p.parse_args()
    baseline() if a.action=='baseline' else run()
