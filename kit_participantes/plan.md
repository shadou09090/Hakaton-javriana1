# Plan de entrenamiento para SpaceLAB

## Objetivo

Entrenar un detector YOLO con PyTorch y Ultralytics que localice y clasifique embarcaciones en imágenes satelitales. La salida son cajas horizontales, cinco clases y una confianza por detección. La métrica que decide es `0.7 × mAP@0.5:0.95 + 0.3 × F1macro@0.5`. El objetivo práctico es producir pronto un ZIP reproducible y medir si mayor resolución, recortes y datos sintéticos mejoran ese punto de partida.

**Orden de trabajo:** verificar CUDA y el flujo de exportación → entrenar YOLO26n con datos reales → probar resolución e inferencia por recortes → probar balanceo y sintéticos → congelar la configuración → evaluar en test y generar el ZIP.

## 1. Entradas, reglas y evidencia

La raíz del proyecto en el primer computador es `C:\Users\gniko\OneDrive\Documentos\tarea javeriana\Entrega`. Contiene `kit_participantes` y `spacelab_dataset`. En el segundo equipo, la ruta absoluta puede ser distinta. Recibir las rutas como argumentos y guardar código nuevo, resultados y derivados fuera de las carpetas de datos y del kit oficial.

| Split | Imágenes JPEG | Cajas verificadas | Uso en este plan |
|---|---:|---:|---|
| `train` | 5.926 | 22.155 | Crear entrenamiento interno y validación de desarrollo |
| `test` | 801 | 3.964 | Evaluación final local de configuraciones congeladas |
| `oculto` | 1.743 | Sin etiquetas distribuidas | Inferencia para la entrega |

| ID | Clase | Cajas en train | Cajas en test |
|---:|---|---:|---:|
| 0 | `small_boat` | 5.600 | 800 |
| 1 | `cargo_ship` | 5.600 | 841 |
| 2 | `fishing_vessel` | 4.988 | 826 |
| 3 | `warship` | 4.255 | 697 |
| 4 | `work_vessel` | 1.712 | 800 |

- **Calidad.** En train y test, cada JPG tiene su TXT y las filas revisadas tienen cinco campos válidos. Se verificaron límites de las cajas con tolerancia de redondeo y se buscaron filas idénticas por archivo. Se calcularon hashes SHA-256 de los JPG de los tres splits y no aparecieron duplicados byte a byte. Esto no mide escenas visualmente parecidas.
- **Escenas.** Los nombres de train y test incluyen fuente y un identificador previo al sufijo del tile. Agrupando por los dos primeros campos separados por `__` aparecen 5.143 grupos en train, 741 en test y cero compartidos. Usar esa clave como aproximación de escena, incluyendo el prefijo de la fuente. Las fuentes son FAIR1M, ShipRSImageNet, VHRShips y xView.
- **Escala.** En train hay 4.837 cajas (21,8 %) con área menor al 0,05 % de la imagen. De ellas, 3.316 son `small_boat` y 1.312 son `fishing_vessel`. El lado corto proyectado de 2.934 cajas queda bajo 8 px al escalar proporcionalmente el lado largo a 640, frente a 684 a 1024. El área media calculada como `w × h` sobre las 22.155 cajas es 1,90 %, distinta del ~0,3 % descriptivo del PDF. Usar los conteos por tamaño para decidir qué probar. Una escena de train llega a 316 cajas.
- **Balance.** `work_vessel` tiene el 7,7 % de las cajas de train y el 20,2 % de las de test. AP y F1 se promedian por clase. Medir las cinco clases por separado y prestar especial atención a las confusiones de `work_vessel`.
- **Reglas.** El PDF permite pesos públicos preentrenados, aumentos y sintéticos derivados de train. Excluye datos etiquetados externos y recuperar o etiquetar a mano el oculto. El pseudoetiquetado automático propio del oculto está permitido, pero queda fuera del flujo principal para concentrar el cómputo en experimentos verificables. Limitar el generador a recortes y fondos del entrenamiento interno.
- **Competencia.** El PDF fija un máximo de 15 envíos con al menos 20 minutos entre ellos y cierre a las 16:30, hora del servidor, el 7 de noviembre de 2026. Para el ranking final, el organizador toma el mejor envío público de cada equipo y lo ordena por su puntaje privado. Preparar un ZIP de respaldo y comprobar cuánto tarda la inferencia completa antes de decidir el último entrenamiento.

## 2. Preparación y partición

1. Verificar `torch.cuda.is_available()`, el nombre del dispositivo, una operación en CUDA y un paso real de entrenamiento e inferencia. El equipo local reportó una RTX 4050 Laptop con 6.141 MiB de VRAM y Python 3.14.4. Elegir versiones compatibles de Python, PyTorch con CUDA, torchvision y Ultralytics según los paquetes disponibles. Registrar versiones, driver, semilla y el SHA-256 del peso inicial. Usar Colab/Kaggle con GPU si el entorno local no permite arrancar a tiempo.
2. Crear `train_dev` (~85 %) y `val_dev` (~15 %) a partir del train oficial con semilla 42. Asignar grupos completos de escena y aproximar la distribución por fuente y presencia multiclase, incluida la clase 4. Guardar un manifiesto con ruta, split, fuente, escena, hash y conteos por clase. Comprobar que las cinco clases estén presentes en ambas partes y que escenas y hashes sean disjuntos. No volver a dividir después de generar recortes o sintéticos.
3. Crear un YAML nuevo para las corridas con `train_dev` y `val_dev`, conservando los cinco IDs. Si se usan listas, comprobar que Ultralytics resuelve bien la ruta correspondiente a cada TXT. Los scripts y el YAML del kit apuntan por defecto a `test` para validación y el script YOLO también predice el oculto al finalizar. Para los experimentos separar entrenamiento, evaluación e inferencia y escribir cualquier caché en una vista de trabajo. Mantener los archivos originales como referencia.
4. Crear un smoke test de 1–2 épocas con un subconjunto fijo de 64–128 imágenes reales de entrenamiento y ejemplos de las cinco clases. Verificar etiquetas visibles, pérdidas finitas, checkpoint recargable, evaluación oficial con un subconjunto de `val_dev` y exportación de archivos para imágenes con y sin detecciones. Hacer también una prueba corta sobre las escenas más densas para estimar VRAM y velocidad antes del entrenamiento completo.

## 3. Entrenamiento y orden de experimentos

### Inicio de las pruebas

La implementación está en `preparar_datos.py`, `entrenar_prueba.py` y `app_streamlit.py` en la raíz. El smoke test con 128 imágenes de entrenamiento y 64 de validación completó dos épocas y guardó checkpoint y evaluación oficial. El split final tiene 5.037 imágenes para entrenamiento y 889 para validación. Tras comprobar memoria con escenas densas y aumentos, se seleccionó para E0 un lote de 8 imágenes, dos workers, 25 épocas y precisión mixta. Consultar el estado real en `runs/E0_A_768_seed42/status.json`. La búsqueda de algoritmos cuDNN está desactivada para evitar un arranque costoso.

Preparar tiles de 512 px, 20 % de solapamiento y 70 % de visibilidad mínima exclusivamente desde `train_dev`. Usar dimensiones reales para imágenes menores al tile, posiciones únicas y muestreo de fondo determinista. La mezcla de tiles e imágenes completas se entrena como corrida separada después de E0. El sobremuestreo de clase 4 se mide en una variante posterior.

Según la aclaración de los organizadores transmitida por el usuario, se permiten imágenes de bases externas excluyendo sus etiquetas originales. El pseudoetiquetado de esas imágenes requiere aclaración antes de incorporarlo al experimento.

Parámetros iniciales para YOLO26n con pesos públicos `yolo26n.pt`: `imgsz=768`, `batch=2`, `amp=True` en CUDA, `workers=0` en Windows durante el arranque, `seed=42`, 20–30 épocas con paciencia 8–10 y guardado de checkpoints periódicos si hay espacio. Empezar con `fliplr=0.5` y `flipud=0.5`, variaciones suaves de brillo y contraste, y mosaic moderado o desactivado al final. Los giros exactos de 90° son una variante opcional con transformación de cajas comprobada. Los valores son propuestas iniciales. Medir VRAM y tiempo por época. Si hay OOM, probar `batch=1`, lotes de inferencia más pequeños o volver a 640/768. Guardar `best.pt`, `last.pt` y configuración efectiva. Usar `last.pt` para reanudar una corrida interrumpida.

| Corrida | Cambio medido | Criterio |
|---|---|---|
| E0 | YOLO26n, 768, imágenes completas reales | Obtener el primer score fiable, ZIP de prueba con val_dev y estimación de tiempo |
| E1a | Mismo checkpoint, inferencia a 1024 frente a 768 | Medir el aporte de resolución en inferencia sin otro entrenamiento |
| E1b | Entrenar YOLO26n a 1024 desde el mismo peso inicial | Comparar a la misma resolución de inferencia con E1a y medir el costo |
| E2 | Mismo checkpoint, imagen completa + tiles | Medir recall y AP de objetos chicos, duplicados y latencia |
| E3 | Entrenar con sobremuestreo de imágenes reales con clase 4 | Control sencillo del efecto de aumentar la exposición a clase 4 |
| E4 | Misma base que E3, con sintéticos derivados de train | Medir el aporte de composición frente a E0/E1 y al control E3 |
| E5 | YOLO26s o entrenamiento con tiles reales | Ejecutar si el análisis de errores y el presupuesto de GPU lo justifican |

Para comparaciones entrenadas, fijar split, semilla, peso inicial, tamaño de imagen y presupuesto comparable de actualizaciones cuando sea posible. Registrar pasos efectivos, batch, tiempo, VRAM pico y exposición real por clase. Si dos corridas usan distinto cómputo, dejarlo explícito. E3 puede empezar con peso de muestreo 2× para imágenes con clase 4. Si se materializan listas o copias, confirmar que el cargador no deduplica la exposición prevista. Usar el control real como opción de bajo costo si los sintéticos no aportan.

**Regla de decisión propuesta.** Usar el score oficial de `val_dev`. Como criterio práctico inicial, pedir al menos +0,005 de score absoluto para adoptar una etapa que añade complejidad. Revisar AP y F1 por clase, recall de barcos chicos y presupuesto de inferencia. Para diferencias cercanas, preferir la variante más simple o confirmar con una segunda semilla si hay tiempo. No interpretar ese margen como significancia estadística. Conservar siempre el mejor candidato ya exportable.

### Reparto de pruebas entre dos computadores

El equipo dispone de dos computadores con especificaciones equivalentes, según el usuario. En el primero se detectó una RTX 4050 Laptop de aproximadamente 6 GB. Verificar CUDA y VRAM en cada equipo. Ejecutar una corrida independiente por GPU.

| Fase | Equipo A · principal | Equipo B · segundo |
|---|---|---|
| Preparación | Crear y congelar el split agrupado, el subconjunto de smoke test y la configuración común | Copiar el kit, el dataset, el código, los manifiestos y los pesos iniciales. Verificar hashes, rutas y CUDA |
| Primeras corridas en paralelo | E0 a 768 px, solo datos reales | E1b a 1024 px, solo datos reales. Probar primero el lote 2 con una escena densa |
| Comparación de escala y recortes | Con el checkpoint de E0, correr E1a a 768 y 1024, y E2 con recortes si hay margen | Exportar E1b a 1024 sobre el mismo `val_dev`. Comparar con E1a a 1024 para aislar el efecto de entrenar a mayor resolución |
| Balanceo y sintéticos | Mantener un candidato reproducible y medir costo de inferencia. Puede preparar el banco de recortes y fondos mientras la GPU entrena | Ejecutar E3 con sobremuestreo real y después E4 con 10 % sintético. Fijar resolución e inferencia para ambos y compartir resultados con A |
| Cierre | Reunir resultados, congelar el candidato, evaluar en test y generar el ZIP final en un equipo | Entregar checkpoints, manifiestos y métricas. Si hace falta, comprobar que puede cargar el checkpoint ganador y reproducir predicciones de una muestra |

**Configuración común.** Generar el split una sola vez. Compartir un manifiesto con rutas relativas al dataset, IDs de escena y hashes para que cada equipo resuelva su propia ruta. Fijar semilla, peso inicial y su SHA-256, versiones de PyTorch, torchvision y Ultralytics, revisión del código, evaluador y presupuesto de entrenamiento. Para E0 y E1b fijar el mismo batch efectivo y las mismas épocas o actualizaciones si la memoria lo permite. Si 1024 requiere un batch menor, registrar ese cambio y, si hace falta una comparación controlada, repetir 768 con el mismo batch. Si 1024 sigue sin caber con batch 1, conservar E0 y usar B para el balanceo o la síntesis.

**Intercambio de resultados.** Usar almacenamiento local para imágenes, cachés y checkpoints durante las corridas. Crear nombres únicos, por ejemplo `E0_A_768_seed42` y `E1b_B_1024_seed42`. Transferir los archivos cuando termine su escritura y comprobar el hash del checkpoint recibido. Por corrida compartir `best.pt`, `last.pt` si se necesita reanudar, configuración efectiva, hash del split, JSON del evaluador, umbral de F1, tiempo y VRAM pico. Para E4 incluir además el manifiesto de sintéticos. Mantener una tabla común con una fila por corrida y enlaces a sus resultados. El equipo A consolida la decisión y valida el ZIP.

## 4. Datos sintéticos a partir de train

Implementar primero **copy-paste con máscara**. El experimento busca aportar variedad de fondos y contextos a `work_vessel`. Añadir `small_boat` solo si persisten fallos en embarcaciones pequeñas y la revisión visual confirma que la composición conserva su silueta.

1. Construir un banco de donantes desde `train_dev`. Empezar con 50–100 recortes revisables de `work_vessel` de varias escenas y tamaños. Extraer una máscara con OpenCV GrabCut u otro método clásico y revisar una muestra. Un rectángulo copiado con todo su fondo introduce bordes artificiales. Descartar donantes con máscaras dudosas, barcos vecinos dentro de la máscara o recortes severos.
2. Usar fondos o imágenes receptoras de `train_dev`. Elegir zonas plausibles de agua o muelle y evitar tierra, estructuras y cajas existentes. Cuando el fondo venga de un recorte aparentemente vacío, comprobar visualmente que no queden barcos sin anotación. En receptores con barcos, conservar todas sus etiquetas y rechazar un pegado que los tape.
3. Aplicar giros en el plano, flips y escalas moderadas ajustadas a los tamaños observados por clase y a la perspectiva cenital. Ajustar color, ruido y nitidez con moderación para que el recorte encaje en el fondo. Transformar imagen y máscara juntas. Derivar la caja horizontal del contorno visible, recortarla a la imagen y convertirla a `clase xc yc w h`. Rechazar instancias demasiado truncadas, fuera de rango o de aspecto poco realista.
4. Generar un piloto de 100–200 imágenes con 1–3 barcos añadidos por imagen. Guardar semilla, donante, receptor, grupo de ambos, parámetros, caja y máscara en un manifiesto JSONL o CSV. Revisar al menos 40 composiciones con cajas dibujadas, incluyendo extremos de tamaño y contraste. Comprobar automáticamente procedencia exclusiva de `train_dev`, geometría, ausencia de solapamientos prohibidos y correspondencia imagen–TXT.
5. En E4 empezar con 10 % sintético por imágenes muestreadas y 90 % reales, manteniendo la validación real. Si mejora, probar 20 % y opcionalmente unas pocas épocas de ajuste final con imágenes reales. Registrar la proporción efectiva y el número de instancias añadidas por clase. Comparar con E3 controlando, en lo posible, actualizaciones y exposición a clase 4. Retirar la mezcla si aumenta falsos positivos o baja el score global sin aportar una mejora reproducible.

## 5. Inferencia por tiles y evaluación

- **Tiles.** Probar ventanas iniciales de 640 o 768 px nativos con 20–25 % de solapamiento en imágenes grandes. Anclar las últimas ventanas a los bordes y conservar dimensiones originales. Combinar predicciones de imagen completa y de tiles. Pasar cajas del tile a coordenadas de la imagen original, recortar, fusionar por clase con NMS como primera opción y aplicar el tope de 300 al final, por imagen. Validar el IoU de fusión en `val_dev` con una cuadrícula pequeña, por ejemplo 0,50/0,60/0,70. Revisar barcos juntos y detecciones partidas en bordes. WBF y TTA explícita con transformaciones inversas quedan para cuando haya tiempo y mejora medible. El kit advierte que su versión de YOLO26 ignora `augment=True` para TTA. Fijar y registrar la versión y el modo de inferencia efectivo.
- **Tiles de entrenamiento, si E5 lo justifica.** Derivarlos solo de `train_dev` y guardar el grupo de la imagen original. Mantener parte de las imágenes completas para barcos grandes. Recortar y remapear etiquetas según una regla explícita de área visible, por ejemplo conservar instancias con al menos 70 % del área de su caja. Si un tile deja un barco claramente visible pero excluido por esa regla, descartar ese tile para evitar falsos negativos de entrenamiento. Validar sobre imágenes originales.
- **Evaluador.** Predecir con un piso bajo de confianza, empezando por `0.001`, y exportar hasta las 300 detecciones de mayor confianza por imagen después de fusionar. El mAP usa las cajas exportadas con independencia del umbral de F1. Ajustar un umbral global en `val_dev` con `evaluar.py --barrer-umbral`. En el CLI, el barrido informa el umbral sugerido, pero la tabla principal y el campo `total` del JSON siguen usando el umbral anterior. Hacer una segunda evaluación con `--umbral VALOR` o incluir ese valor en `umbral.txt`. El score de desarrollo usa el mismo split que el ajuste y sirve para comparar, no para estimar de forma independiente la generalización.
- **Selección.** Reportar score, mAP50–95, mAP50, F1 macro y AP, precisión, recall y F1 por clase. Para diagnóstico, usar los prefijos para construir un CSV `id,fuente,particion` y pasarlo con `--meta`. Separar errores en barcos muy pequeños, escenas densas, confusión de clase, duplicados y cajas imprecisas. Congelar checkpoint, resolución, modo de inferencia, fusión y umbral antes de evaluar `test`. Evaluar ahí el candidato y, si se desea cuantificar el avance, el baseline preseleccionado con su umbral ya fijado. Evitar recalibrar con test y reservar el oculto para exportación.

## 6. Entrega y verificación

Un TXT por cada una de las **1.743** imágenes de `images/oculto`, mismo nombre base. Cada fila contiene `clase xc yc ancho alto confianza` en coordenadas normalizadas respecto a la imagen original. Crear el TXT vacío si no hay detecciones. Incluir `umbral.txt` con el valor congelado. Usar `escribir_txt` y `crear_zip` de `formato_envio.py`. Comprobar adicionalmente que coincidan exactamente los 1.743 nombres, sin duplicados ni extras, porque el verificador oficial puede devolver éxito con advertencias por archivos faltantes. Verificar campos finitos, caja válida, clase 0–4, confianza 0–1 y máximo 300 filas.

Ejemplos en PowerShell desde la raíz del proyecto. `$kit` y `$data` apuntan a las carpetas originales en cualquiera de los dos equipos. Las carpetas de predicciones y las etiquetas de desarrollo son productos de la implementación. `0.30` es un ejemplo y se reemplaza por el umbral seleccionado.

```powershell
$root = (Get-Location).Path
$kit = Join-Path $root 'kit_participantes'
$data = Join-Path $root 'spacelab_dataset'
python "$kit/evaluar.py" --gt ".\derived\labels\val_dev" --envio ".\runs\E0\pred_val" --barrer-umbral
python "$kit/evaluar.py" --gt ".\derived\labels\val_dev" --envio ".\runs\E0\pred_val" --umbral 0.30 --json ".\runs\E0\metricas_val.json"
python "$kit/evaluar.py" --gt "$data/labels/test" --envio ".\runs\final\pred_test" --umbral 0.30 --json ".\runs\final\metricas_test.json"
python "$kit/formato_envio.py" zip --carpeta ".\runs\final\pred_oculto" --salida ".\runs\final\envio_spacelab.zip" --umbral 0.30
python "$kit/formato_envio.py" verificar --envio ".\runs\final\envio_spacelab.zip" --imagenes "$data/images/oculto"
```

**Criterio de término.** ZIP validado, checkpoint recargable, métricas de test con umbral congelado, versiones y configuraciones guardadas, manifiestos de partición y sintéticos si aplica, y comandos suficientes para reconstruir las predicciones. Dejar el envío al leaderboard como acción manual del equipo. Medir la duración de inferencia y reservar ese tiempo más margen para validar el ZIP.

## Guía para asistentes de IA

Consulta `AGENTS.md` en la raíz del proyecto para ver el mapa de archivos, los contratos del dataset y las instrucciones para implementar o retomar este plan con Codex, ChatGPT o Astra.

## Referencias

- Reglamento oficial y formato de evaluación. `kit_participantes/reto_spacelab.pdf`.
- Código oficial de referencia. En `kit_participantes/`, consulta `entrenar_baseline.py`, `evaluar.py` y `formato_envio.py`. El YAML distribuido está en `spacelab_dataset/data.yaml`.
- Documentación oficial para implementación. [PyTorch y CUDA](https://pytorch.org/get-started/locally/), [parámetros de Ultralytics](https://docs.ultralytics.com/usage/cfg), [formatos de detección](https://docs.ultralytics.com/datasets/detect), [inferencia por recortes](https://docs.ultralytics.com/guides/sahi-tiled-inference) y [modos de inferencia de YOLO26](https://docs.ultralytics.com/guides/end2end-detection).
