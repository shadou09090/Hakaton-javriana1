# AGENTS.md · SpaceLAB GeoVision Colombia

## Propósito y punto de entrada

Este proyecto prepara un detector de embarcaciones en imágenes satelitales para el reto SpaceLAB. El enfoque del equipo es YOLO26 con PyTorch y Ultralytics, con experimentos de resolución, recortes y datos sintéticos. Lee [el plan de pruebas](kit_participantes/plan.md) antes de implementar. Sigue el alcance que pida el usuario y retoma el estado comprobado de las corridas existentes.

La raíz en el computador principal es `C:\Users\gniko\OneDrive\Documentos\tarea javeriana\Entrega`. En el otro computador puede ser distinta. Identifica la raíz por la presencia de `kit_participantes/` y `spacelab_dataset/`. Las rutas de esta guía son relativas a esa raíz. Si la sesión abrió `ChatGPT\Entrega`, comprueba la ruta antes de buscar los datos. Si tu asistente no carga automáticamente esta guía, adjunta `AGENTS.md` y `kit_participantes/plan.md` a su contexto.

## Mapa del proyecto

| Ruta | Uso |
|---|---|
| `kit_participantes/reto_spacelab.pdf` | Fuente oficial de reglas, métricas y formato. Consultar las secciones 3–6 ante dudas |
| `kit_participantes/LEEME.txt` | Inventario y flujo rápido del kit |
| `kit_participantes/plan.md` | Estrategia, experimentos E0–E5, reparto entre dos equipos, sintéticos y entrega |
| `kit_participantes/entrenar_baseline.py`, `baseline_colab.ipynb` | Referencia oficial YOLO26 para local o Colab/Kaggle |
| `kit_participantes/entrenar_baseline_frcnn.py`, `baseline_frcnn_colab.ipynb` | Referencia alternativa con Faster R-CNN y torchvision |
| `kit_participantes/evaluar.py` | Evaluación oficial local, desglose y barrido del umbral de F1 |
| `kit_participantes/formato_envio.py` | Conversión de cajas, empaquetado y verificación del envío |
| `spacelab_dataset/data.yaml` | Mapeo de clases y rutas oficiales. Apunta `val` a `test` |
| `spacelab_dataset/images/train` y `labels/train` | 5.926 imágenes y 22.155 cajas para crear el split de desarrollo |
| `spacelab_dataset/images/test` y `labels/test` | 801 imágenes y 3.964 cajas para evaluación local final |
| `spacelab_dataset/images/oculto` | 1.743 imágenes sin etiquetas distribuidas, para inferencia final |
| `spacelab_dataset.zip` | Respaldo del dataset. Usa la copia extraída si está disponible |

## Contratos de datos y reglas de trabajo

- IDs fijos en este orden. `0 small_boat`, `1 cargo_ship`, `2 fishing_vessel`, `3 warship`, `4 work_vessel`. Etiqueta de entrenamiento `clase xc yc w h`, con cajas horizontales y geometría normalizada. Predicción `clase xc yc w h confianza`.
- Crea una sola partición reproducible `train_dev` y `val_dev` desde `train`. Como aproximación de escena, usa los primeros dos componentes del nombre separados por `__`. Mantén los grupos juntos y verifica que ambas particiones tengan las cinco clases. El manifiesto debe usar rutas relativas al dataset.
- Deriva recortes, donantes y fondos sintéticos únicamente de `train_dev`. Conserva trazabilidad. Usa `val_dev` para comparar modelos y calibrar el umbral, `test` para comprobar el candidato congelado y `oculto` para inferencia final. Respeta las restricciones del reto sobre datos etiquetados externos y etiquetas del oculto. Los pesos preentrenados públicos están permitidos.
- El baseline del kit evalúa en `test` y también predice el oculto. Para pruebas de desarrollo, separa esas etapas con tus propios comandos y usa un YAML de trabajo. Mantén el dataset original y los scripts oficiales como referencia y escribe los derivados fuera de sus carpetas.
- Si hace falta implementar, usa `scripts/` para comandos, `configs/` para configuraciones, `derived/` para manifiestos y datos derivados, y `runs/` para resultados por corrida, o reutiliza una estructura existente equivalente. Estas son rutas propuestas. Parametriza `--data-root`, `--kit-root` y `--output-root`. Excluye datasets, pesos, cachés, credenciales y resultados pesados de Git. Comparte artefactos solo dentro del equipo.

## Cómo retomar y repartir el trabajo

Implementación disponible en la raíz del proyecto. `preparar_datos.py` crea el split agrupado y los tiles corregidos. `entrenar_prueba.py` ejecuta entrenamiento CUDA y evaluación oficial en desarrollo. `app_streamlit.py` muestra datos, recortes y métricas en vivo. `INICIAR.md` contiene los comandos. Consultar `runs/*/status.json` para conocer el estado real de cada corrida.

El usuario consultó a los organizadores y comunicó que se admiten bases externas sin utilizar sus etiquetas originales. El uso de pseudoetiquetas en esas imágenes queda pendiente de aclaración. La implementación inicial utiliza el dataset oficial.

1. Comprueba qué scripts, manifiestos, checkpoints y métricas existen. Un documento de planificación no demuestra que una corrida haya terminado. En cada equipo verifica `nvidia-smi`, `torch.cuda.is_available()`, una operación CUDA y un paso real de entrenamiento e inferencia. En A se detectó una RTX 4050 Laptop de aproximadamente 6 GB. B tiene especificaciones equivalentes según el usuario y requiere verificación local.
2. Tras el smoke test, A entrena E0 a 768 px y B entrena E1b a 1024 px. Copia el mismo split, pesos iniciales, versiones y código a ambos equipos. Usa un identificador y una carpeta de salida exclusivos por corrida. Si B no tiene margen de VRAM a 1024, prueba batch 1 y registra el cambio. Si sigue sin caber, B puede avanzar con E3/E4 a la resolución viable.
3. A prueba E1a/E2 y consolida el candidato. B compara E3 (sobremuestreo real) con E4 (sintéticos). Guarda por corrida configuración efectiva, semilla, hash del split y de pesos, tiempo, VRAM, métrica global y por clase, y manifiesto sintético cuando aplique. Transfiere checkpoints completos y verifica su hash antes de cargarlos.
4. Ante una interrupción, inspecciona los registros y la configuración antes de reanudar con `last.pt`. Para iniciar una variante con cambios, usa una carpeta nueva y registra el checkpoint o peso inicial. Reporta qué se ejecutó y qué queda pendiente con un comando reproducible.

## Validación y entrega

- Usa el score oficial `0.7 × mAP@0.5:0.95 + 0.3 × F1macro@0.5`. `evaluar.py --barrer-umbral` sugiere el umbral, pero el resultado principal conserva el valor anterior. Vuelve a ejecutar con `--umbral` fijado. Usa un piso de exportación bajo y conserva las confianzas para mAP.
- Genera un TXT por imagen del oculto, incluso si está vacío. Fusiona y limita a 300 cajas por imagen en coordenadas originales, y empaqueta con `formato_envio.py` más `umbral.txt`. Exige 1.743 nombres exactos y revisa advertencias del verificador. Prueba el caso sin detecciones y el remapeo de cajas de tiles si se usa.
- Entrega el ZIP validado, checkpoint, métricas reales y pasos para reproducirlo. La carga al leaderboard corresponde al equipo y se realiza cuando el usuario lo indique. El token queda fuera de archivos, comandos de ejemplo y registros.
