# Resultados de la hora extra — 7 de noviembre de 2026

La inferencia se ejecutó en la GPU local con el checkpoint importado de PC 2. El checkpoint contiene resultados de las épocas 1 a 25 de E1b_B_1024_seed42.

## Archivos listos

| Archivo en la raíz de Entrega | Contenido | SHA-256 |
|---|---|---|
| `envio_solo_pc2_1024_wbf.zip` | 1743 TXT de oculto calibrados + umbral.txt (0.33) | `c112361623e3bd4433c0b54c5cb91ad21510f3c9952dee2f27ca16bef64504ff` |
| `transferencia_pc2.zip` | 889 TXT val_dev + 1743 TXT oculto sin gamma, best_pc2_1024.pt y metadatos | `408aa5ffdbf7b47bdc9aba84b52c09ad132721fa92776de558f82ae2e37c7465` |
| `envio_mejor_validado_hora_extra.zip` | Mejor configuración medida en val_dev; 1743 TXT + umbral.txt | `63477ca0783c295d9741fd6aa9b48a322bc5fcc768e1e4cecd092b5abf57a4a6` |

## Validación y selección

Todos los puntajes siguientes son locales, sobre las mismas 889 imágenes de val_dev con el evaluador oficial. No son puntajes del leaderboard. No se usaron etiquetas de oculto ni del test local para ajustar estas configuraciones. El manifiesto de entrenamiento de PC 2 no vino en los archivos recibidos; su coincidencia con el de PC 1 no pudo verificarse independientemente.

| Configuración | Score | mAP50:95 | F1 macro |
|---|---:|---:|---:|
| seleccion_por_clase | 0.561929 | 0.500707 | 0.704781 |
| pc1_tta_pc2_tta_0.7_support | 0.559168 | 0.498748 | 0.700147 |
| pc1_tta_pc2_tta_0.6_support | 0.553082 | 0.490594 | 0.698887 |
| pc1_tta_pc2_tta_0.7_max | 0.552250 | 0.492699 | 0.691203 |
| pc1_full_pc2_tta_0.7_support | 0.551914 | 0.491444 | 0.693009 |
| pc1_tta_pc2_tta_0.6_max | 0.549298 | 0.487908 | 0.692541 |
| three_support | 0.548359 | 0.483281 | 0.700208 |
| pc1_full_pc2_tta_0.7_max | 0.547422 | 0.488646 | 0.684565 |
| pc1_full_pc2_tta_0.6_max | 0.544444 | 0.482390 | 0.689236 |
| pc1_tta | 0.542064 | 0.479505 | 0.688036 |
| pc1_full_pc2_tta_0.6_support | 0.540973 | 0.476323 | 0.691822 |
| pc2_tta | 0.539894 | 0.476909 | 0.686858 |
| pc1_tta_pc1_tiles_0.7_support | 0.537170 | 0.473292 | 0.686218 |
| pc1_tiles_pc2_tta_0.7_support | 0.533738 | 0.471137 | 0.679808 |
| pc1_tta_pc2_tta_0.5_max | 0.532671 | 0.465137 | 0.690250 |
| pc1_tta_pc1_tiles_0.6_support | 0.531797 | 0.464845 | 0.688017 |
| pc1_full_pc1_tiles_0.7_support | 0.529058 | 0.466209 | 0.675708 |
| pc1_full | 0.527803 | 0.465107 | 0.674093 |
| pc1_tiles_pc2_tta_0.6_support | 0.527580 | 0.460717 | 0.683595 |
| pc1_full_pc2_tta_0.5_max | 0.527019 | 0.459030 | 0.685659 |
| pc1_tta_pc2_tta_0.5_support | 0.525256 | 0.458460 | 0.681112 |
| pc1_full_pc2_tta_0.5_support | 0.519606 | 0.450730 | 0.680317 |
| pc1_full_pc1_tiles_0.6_support | 0.515514 | 0.447618 | 0.673938 |
| three_max | 0.513399 | 0.452742 | 0.654931 |
| pc1_tta_pc1_tiles_0.5_support | 0.511219 | 0.440437 | 0.676377 |
| pc1_tiles_pc2_tta_0.5_support | 0.511001 | 0.440460 | 0.675597 |
| pc1_tta_pc1_tiles_0.7_max | 0.506856 | 0.445009 | 0.651166 |
| pc1_tta_pc1_tiles_0.6_max | 0.502661 | 0.438297 | 0.652843 |
| pc1_full_pc1_tiles_0.5_support | 0.500239 | 0.430389 | 0.663223 |
| pc1_tiles_pc2_tta_0.7_max | 0.500195 | 0.439007 | 0.642968 |
| pc1_full_pc1_tiles_0.7_max | 0.497256 | 0.436375 | 0.639311 |
| pc1_tiles_pc2_tta_0.6_max | 0.494258 | 0.430299 | 0.643495 |
| pc1_tta_pc1_tiles_0.5_max | 0.491961 | 0.423248 | 0.652291 |
| pc1_full_pc1_tiles_0.6_max | 0.491438 | 0.426391 | 0.643215 |
| pc1_tiles_pc2_tta_0.5_max | 0.485256 | 0.416857 | 0.644852 |
| pc1_full_pc1_tiles_0.5_max | 0.481806 | 0.413254 | 0.641759 |
| pc1_tiles | 0.449161 | 0.376273 | 0.619234 |

Selección final: `seleccion_por_clase`; score **0.561929**.
Envío individual PC 2: score **0.539894**, mAP **0.476909**, F1 **0.686858**.

## TTA y calibración PC 2

- Inferencia a 1024, vistas original, horizontal y vertical. Se revierten las transformaciones antes de WBF.
- WBF con IoU 0.60. Se compararon confianzas por soporte, media y máximo; ganó máximo en val_dev.
- Exportación sin gamma con confianza mínima 0.01, máximo 300 cajas por imagen.
- Umbrales por clase buscados entre 0.01 y 0.91, paso 0.01, con umbral global 0.33.
- Bajo recall se definió como recall < 0.50 a umbral 0.33. Estas clases tienen gamma <= 1.
- Se mantienen las mismas cajas antes y después de gamma y se verifica nuevamente el score del TXT exportado.

| Clase | Umbral original | Gamma | Recall bajo |
|---|---:|---:|---|
| small_boat | 0.37 | 1.115072 | No |
| cargo_ship | 0.40 | 1.209946 | No |
| fishing_vessel | 0.19 | 0.667575 | Sí |
| warship | 0.49 | 1.554164 | No |
| work_vessel | 0.45 | 1.388418 | No |

## Selección final por clase

Se eligió por clase la configuración con mayor 0.7 × AP50:95 + 0.3 × F1 en val_dev y se volvió a evaluar el conjunto completo después de limitar a 300 cajas por imagen.

| Clase | Configuración |
|---|---|
| small_boat | `three_support` |
| cargo_ship | `pc1_tta_pc2_tta_0.7_support` |
| fishing_vessel | `pc1_tta_pc2_tta_0.7_support` |
| warship | `pc1_full_pc2_tta_0.7_support` |
| work_vessel | `pc1_tta_pc2_tta_0.7_support` |

## Ubicación y reproducción

- `optimizar_hora_extra.py`: crea el envío PC 2 y la transferencia; reutiliza las vistas guardadas.
- `comparar_fusion_hora_extra.py`: acciones `tta-pc1`, `sliced-val` y `search`.
- `runs/hora_extra_pc2/`: caché TTA, comparación de WBF, calibración y métricas exportadas.
- `runs/hora_extra_fusion/`: pruebas PC 1 y comparación de fusión con configuración final.
- `preds_pc2_val/` y `preds_pc2_test/`: predicciones crudas para fusionar, sin aplicar gamma dos veces.
- `preds_pc2_test` significa las 1743 imágenes de `spacelab_dataset/images/oculto`. El `test` local tiene 801 imágenes.
- `best.pt.zip` recibido era un checkpoint PyTorch con extensión adicional; se copió como `best_pc2_1024.pt`.
- Los ZIP de envío se verificaron con el lector oficial, nombres exactos, 0 advertencias, cajas válidas y máximo 300 detecciones.

Las estimaciones de 0.55–0.61 que figuraban antes en AGENTS.md eran proyecciones. WBF no garantiza mejorar el puntaje ni encontrar el centro exacto; la selección aquí se basa en mediciones.


## Resultado confirmado en plataforma — envío de las 16:49

El usuario confirmó que subió `envio_mejor_validado_hora_extra.zip`. La captura del 7 de noviembre muestra: **score 0.4655**, **mAP50:95 0.4131**, **F1 0.5877**, puesto **7**, 4 envíos. La diferencia respecto al score val_dev 0.561929 es 0.096429. El score anterior confirmado era 0.4138: mejora absoluta 0.0517.

La auditoría `auditar_test_hora_extra.py` finalizó sobre las 801 imágenes etiquetadas del test local, sin recalibrar ni seleccionar parámetros en test: combinación **0.481963** (mAP **0.425585**, F1 **0.613514**), PC 2 TTA **0.467600**, PC 1 TTA **0.448292**, PC 1 completa **0.435451**. La combinación conserva la mejora, aunque val_dev sobreestima su resultado externo. Pesqueros: AP **0.141210**, recall **0.391041**. Ver `AUDITORIA_TEST801.md` y `runs/auditoria_test801/results.json`. Los ZIP anteriores no se modificaron.
