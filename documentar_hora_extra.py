"""Genera el registro final a partir de las metricas y ZIP realmente producidos."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent


def main():
    p2=ROOT/'runs/hora_extra_pc2'
    fusion=ROOT/'runs/hora_extra_fusion'
    cal=json.loads((p2/'selected_calibration.json').read_text(encoding='utf-8'))
    selected=json.loads((fusion/'selected.json').read_text(encoding='utf-8'))
    comparison=json.loads((fusion/'comparison.json').read_text(encoding='utf-8'))
    s=cal['after']['total'];best=selected['calibration']['after']['total']
    lines=['# Resultados de la hora extra — 7 de noviembre de 2026','',
           'La inferencia se ejecutó en la GPU local con el checkpoint importado de PC 2. '
           'El checkpoint contiene resultados de las épocas 1 a 25 de E1b_B_1024_seed42.','',
           '## Archivos listos','',
           '| Archivo en la raíz de Entrega | Contenido | SHA-256 |',
           '|---|---|---|']
    for name,description in [
        ('envio_solo_pc2_1024_wbf.zip','1743 TXT de oculto calibrados + umbral.txt (0.33)'),
        ('transferencia_pc2.zip','889 TXT val_dev + 1743 TXT oculto sin gamma, best_pc2_1024.pt y metadatos'),
        ('envio_mejor_validado_hora_extra.zip','Mejor configuración medida en val_dev; 1743 TXT + umbral.txt')]:
        path=ROOT/name
        with zipfile.ZipFile(path) as z: assert z.testzip() is None
        lines.append(f'| `{name}` | {description} | `{hashlib.sha256(path.read_bytes()).hexdigest()}` |')
    lines += ['', '## Validación y selección','',
              'Todos los puntajes siguientes son locales, sobre las mismas 889 imágenes de val_dev con '
              'el evaluador oficial. No son puntajes del leaderboard. No se usaron etiquetas de oculto '
              'ni del test local para ajustar estas configuraciones. El manifiesto de entrenamiento '
              'de PC 2 no vino en los archivos recibidos; su coincidencia con el de PC 1 no pudo '
              'verificarse independientemente.','',
              '| Configuración | Score | mAP50:95 | F1 macro |','|---|---:|---:|---:|']
    for r in sorted(comparison,key=lambda r:r['calibration']['after']['total']['score'],reverse=True):
        m=r['calibration']['after']['total'];lines.append(f"| {r['config']['name']} | {m['score']:.6f} | {m['map50_95']:.6f} | {m['f1_macro']:.6f} |")
    lines += ['',f"Selección final: `{selected['config']['name']}`; score **{best['score']:.6f}**.",
              f"Envío individual PC 2: score **{s['score']:.6f}**, mAP **{s['map50_95']:.6f}**, F1 **{s['f1_macro']:.6f}**.",
              '', '## TTA y calibración PC 2','',
              '- Inferencia a 1024, vistas original, horizontal y vertical. Se revierten las transformaciones antes de WBF.',
              '- WBF con IoU 0.60. Se compararon confianzas por soporte, media y máximo; ganó máximo en val_dev.',
              '- Exportación sin gamma con confianza mínima 0.01, máximo 300 cajas por imagen.',
              '- Umbrales por clase buscados entre 0.01 y 0.91, paso 0.01, con umbral global 0.33.',
              '- Bajo recall se definió como recall < 0.50 a umbral 0.33. Estas clases tienen gamma <= 1.',
              '- Se mantienen las mismas cajas antes y después de gamma y se verifica nuevamente el score del TXT exportado.',
              '', '| Clase | Umbral original | Gamma | Recall bajo |','|---|---:|---:|---|']
    names=['small_boat','cargo_ship','fishing_vessel','warship','work_vessel']
    for c,name in enumerate(names):
        lines.append(f"| {name} | {cal['class_thresholds'][c]:.2f} | {cal['gammas'][c]:.6f} | {'Sí' if cal['low_recall_at_033'][c] else 'No'} |")
    if selected['config']['kind']=='class_select':
        lines += ['', '## Selección final por clase','',
                  'Se eligió por clase la configuración con mayor 0.7 × AP50:95 + 0.3 × F1 '
                  'en val_dev y se volvió a evaluar el conjunto completo después de limitar a 300 cajas por imagen.','',
                  '| Clase | Configuración |','|---|---|']
        for name,cfg in zip(names,selected['config']['class_configs']):
            lines.append(f"| {name} | `{cfg['name']}` |")
    lines += ['', '## Ubicación y reproducción','',
              '- `optimizar_hora_extra.py`: crea el envío PC 2 y la transferencia; reutiliza las vistas guardadas.',
              '- `comparar_fusion_hora_extra.py`: acciones `tta-pc1`, `sliced-val` y `search`.',
              '- `runs/hora_extra_pc2/`: caché TTA, comparación de WBF, calibración y métricas exportadas.',
              '- `runs/hora_extra_fusion/`: pruebas PC 1 y comparación de fusión con configuración final.',
              '- `preds_pc2_val/` y `preds_pc2_test/`: predicciones crudas para fusionar, sin aplicar gamma dos veces.',
              '- `preds_pc2_test` significa las 1743 imágenes de `spacelab_dataset/images/oculto`. El `test` local tiene 801 imágenes.',
              '- `best.pt.zip` recibido era un checkpoint PyTorch con extensión adicional; se copió como `best_pc2_1024.pt`.',
              '- Los ZIP de envío se verificaron con el lector oficial, nombres exactos, 0 advertencias, cajas válidas y máximo 300 detecciones.',
              '', 'Las estimaciones de 0.55–0.61 que figuraban antes en AGENTS.md eran proyecciones. '
              'WBF no garantiza mejorar el puntaje ni encontrar el centro exacto; la selección aquí se basa en mediciones.','']
    (ROOT/'RESULTADOS_HORA_EXTRA.md').write_text('\n'.join(lines),encoding='utf-8')
    marker='## Actualización verificada de la hora extra — 2026-11-07'
    agent=ROOT/'AGENTS.md';text=agent.read_text(encoding='utf-8')
    if marker in text: text=text[:text.index(marker)].rstrip()
    text+='\n\n'+marker+'\n\n'
    text+=f'- PC 2 importado: `best_pc2_1024.pt`, E1b_B_1024_seed42, 25 épocas registradas.\n'
    text+=f'- TTA 1024 + WBF IoU 0.60 + gamma: score val_dev {s["score"]:.6f}; ZIP `envio_solo_pc2_1024_wbf.zip`.\n'
    text+=f'- Mejor opción medida: `{selected["config"]["name"]}`, score val_dev {best["score"]:.6f}; ZIP `envio_mejor_validado_hora_extra.zip`.\n'
    text+='- `transferencia_pc2.zip` contiene 889 predicciones val y 1743 oculto sin gamma, modelo y metadatos.\n'
    text+='- Leer `RESULTADOS_HORA_EXTRA.md` y los JSON de `runs/hora_extra_pc2/` y `runs/hora_extra_fusion/` para métricas, comandos y procedencia.\n'
    text+='- Las estimaciones anteriores de score no son mediciones. Estos resultados son de validación local, no del leaderboard. WBF no garantiza mejoras.\n'
    text+='- No fusionar carpetas calibradas ni aplicar gamma dos veces. Usar predicciones crudas y recalibrar en val_dev.\n'
    agent.write_text(text,encoding='utf-8')
    print(ROOT/'RESULTADOS_HORA_EXTRA.md')


if __name__=='__main__':main()
