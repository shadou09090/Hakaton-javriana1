# Auditoría independiente del test local (801 imágenes)

Configuraciones congeladas antes de evaluar. No se recalibró ni se eligieron parámetros sobre test.

| Configuración | Score val_dev | Score test801 | mAP test801 | F1 test801 |
|---|---:|---:|---:|---:|
| pc1_full | 0.527803 | 0.435451 | 0.379821 | 0.565254 |
| pc2_tta | 0.539894 | 0.467600 | 0.408773 | 0.604863 |
| pc1_tta | 0.542064 | 0.448292 | 0.395339 | 0.571849 |
| seleccion_por_clase | 0.561929 | 0.481963 | 0.425585 | 0.613514 |

El usuario confirmó para `envio_mejor_validado_hora_extra.zip` en el leaderboard: score 0.4655, mAP 0.4131, F1 0.5877, puesto 7.

| Clase (combinación) | AP val_dev | AP test801 | Recall test801 | F1 test801 |
|---|---:|---:|---:|---:|
| small_boat | 0.3297 | 0.3163 | 0.5863 | 0.6067 |
| cargo_ship | 0.6034 | 0.4687 | 0.5042 | 0.5722 |
| fishing_vessel | 0.2506 | 0.1412 | 0.3910 | 0.3963 |
| warship | 0.7335 | 0.7621 | 0.8494 | 0.8674 |
| work_vessel | 0.5863 | 0.4396 | 0.6425 | 0.6249 |
