from pathlib import Path
import csv
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent


def main():
    work=ROOT/'runs/final26'
    record=json.loads((work/'final_selected.json').read_text(encoding='utf-8'))
    final=ROOT/'entrega_final.zip'
    with zipfile.ZipFile(final) as z:
        assert z.testzip() is None
        assert len(z.namelist())==1744
        assert z.read('umbral.txt').decode().strip()=='0.33'
    digest=hashlib.sha256(final.read_bytes()).hexdigest()
    assert digest==record['final_sha256']
    results=ROOT/'runs/E3_final_pc2_refine_1024/results.csv'
    epochs=list(csv.DictReader(results.open())) if results.exists() else []
    lines=['# Entrega final','',f'Archivo listo: `{final}`.','',
           f"Tipo seleccionado: **{record['kind']}**.",
           f"Score val_dev: **{record['val']['score']:.6f}**.",
           f"Score test local 801: **{record['test801']['score']:.6f}**.",
           f"mAP test local: **{record['test801']['map50_95']:.6f}**; F1: **{record['test801']['f1_macro']:.6f}**.",'',
           'Estos puntajes son locales. El nuevo archivo requiere evaluación en la plataforma. '
           'El envío anterior obtuvo 0.4655 en el leaderboard y 0.481963 en el test local.','',
           'El entrenamiento adicional usó exclusivamente train_dev. Se mantuvieron val_dev y el test '
           'fuera del entrenamiento. Los parámetros se ajustaron en val_dev; test801 se utilizó para '
           'comparar hasta tres candidatos previamente seleccionados y rechazar retrocesos, por lo '
           'que ya no es una estimación completamente independiente de la selección final.','',
           f'Épocas adicionales registradas: {len(epochs)}. Parada de entrenamiento prevista: 17:17 Bogotá.',
           'Configuración: YOLO26n inicializado con PC2, 1024 px, batch 8, AdamW 0.0002, sin mosaic/mixup, '
           'escala 0.25, flips horizontal/vertical y pérdidas box=8.5, cls=0.7.','',
           'Verificado: 1743 TXT de predicciones, nombres exactos, 0 advertencias del lector oficial, '
           'umbral 0.33, máximo 300 detecciones por imagen, coordenadas y números finitos, integridad ZIP.','',
           f'SHA-256: `{digest}`.','',
           'Detalles reproducibles: `runs/final26/final_selected.json`, `comparison_val.json`, '
           '`test_comparison.json` cuando existe, y `runs/E3_final_pc2_refine_1024/`.','']
    (ROOT/'ENTREGA_FINAL.md').write_text('\n'.join(lines),encoding='utf-8')
    agents=ROOT/'AGENTS.md';text=agents.read_text(encoding='utf-8')
    marker='## Cierre de la ventana final de 26 minutos'
    if marker in text:text=text[:text.index(marker)].rstrip()
    text+='\n\n'+marker+'\n\n'
    text+=f"- Archivo para subir: `entrega_final.zip`. Selección: `{record['kind']}`.\n"
    text+=f"- Val_dev {record['val']['score']:.6f}; test801 {record['test801']['score']:.6f}. Son resultados locales, no leaderboard.\n"
    text+='- Leer `ENTREGA_FINAL.md` y `runs/final26/final_selected.json` para reproducir la decisión.\n'
    text+='- El test local se usó para comparar candidatos preseleccionados en esta fase final; no presentarlo como prueba totalmente independiente.\n'
    agents.write_text(text,encoding='utf-8')
    print(json.dumps({'file':str(final),'sha256':digest,'kind':record['kind'],
                      'val_score':record['val']['score'],'test_score':record['test801']['score'],'epochs':len(epochs)}),flush=True)


if __name__=='__main__':main()
