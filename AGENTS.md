# AGENTS.md · SpaceLAB GeoVision Colombia — Equipo: "Los interpoolers"

## Propósito, Reglas Fundamentales y Métrica Oficial

Este proyecto prepara el detector de embarcaciones satelitales del equipo **"Los interpoolers"** para el reto SpaceLAB GeoVision Colombia.
- **Objetivo Competitivo:** Superar la barrera de 0.5450 (actual líder #1: Redjaguar con 0.5448) y alcanzar **0.6000+** para asegurar la victoria.
- **Puntaje Oficial:**
  $$\text{Score}_{\text{Final}} = 0.70 \times \text{mAP@0.5:0.95} + 0.30 \times F1_{\text{macro@0.5}}$$
- **Contrato de Clases (IDs fijos obligatorios):**
  - `0`: `small_boat`
  - `1`: `cargo_ship`
  - `2`: `fishing_vessel`
  - `3`: `warship`
  - `4`: `work_vessel`
- **Límites de Detección:** Máximo 300 cajas por imagen. Cajas en coordenadas normalizadas `clase xc yc w h confianza`. Geometría válida: $w \ge 0.0005$ y $h \ge 0.0005$. Archivos en `oculto`: exactamente 1.743 archivos `.txt`.

---

## Historial de Envíos en Plataforma Oficial

| # | Archivo Enviado | Umbral | Detecciones | mAP@0.5:0.95 | F1@0.5 | Score Final | Estado / Proyección |
|---|---|---|---|---|---|---|---|
| **1** | `envio_spacelab_baseline.zip` | 0.36 | 28.514 | 0.3531 | 0.5013 | **0.3975** | Evaluado (Puesto #7). 13 cajas con $w,h \le 0$ ignoradas por validador. |
| **2** | `envio_spacelab_1024.zip` | 0.33 | 48.912 | 0.3594 | 0.5407 | **0.4138** | Evaluado (Puesto #6). 1024px nativo, subida de +0.0163 (+0.0394 F1). |
| **3** | `envio_spacelab_1024_calibrado.zip` | 0.33 | 50.602 | *Pendiente* | *Pendiente* | *Est. 0.44-0.46* | Calibración gamma monótona + floor 0.005. |
| **4** | `envio_spacelab_sliced_E2.zip` | 0.33 | 55.381 | *Pendiente* | *Pendiente* | *Est. 0.48-0.52* | Slicing SAHI 512px + Padding reflexivo sobre modelo de tiles E2. |
| **5** | `envio_spacelab_final_pc1_pc2.zip` | 0.33 | 48.212 (8.005 $\ge 0.33$) | *Pendiente* | *Pendiente* | *Est. 0.55-0.62* | **ENVÍO DEFINITIVO**. WBF anti-duplicados (IoU 0.38) entre PC 1 Sliced y PC 2 1024px real. Densidad óptima de 4.59 barcos/img. |

---

## Diagnóstico Matemático y Auditoría Exhaustiva

### 1. ¿Por qué F1@0.5 subió +0.0394 pero mAP@0.5:0.95 solo +0.0063?
- **F1@0.5** evalúa únicamente si la caja solapa con $\text{IoU} \ge 0.50$. Al pasar de 768px a 1024px y bajar el umbral a 0.33, muchas cajas de barcos pequeños alcanzaron $\text{IoU} \approx 0.52 - 0.60$, activándose como verdaderos positivos a umbral 0.50.
- **mAP@0.5:0.95** es el promedio del mAP en los 10 cortes de IoU: $\{0.50, 0.55, 0.60, \dots, 0.90, 0.95\}$. A 1024px con un solo modelo y sin promediado espacial (WBF), las cajas presentan un jitter natural de 1–3 píxeles en bordes satelitales, colapsando su precisión en $\text{IoU} \ge 0.75$.
- **Solución implementada:** Weighted Boxes Fusion (WBF) en `ensemble.py`, que promedia de forma continua y ponderada los vértices de las cajas superpuestas, recentrando la predicción en el centroide exacto del barco.

### 2. Disparidad entre `val_dev` (0.5218) y `test`/Plataforma (0.4138)
- **Auditoría de Fuga (Data Leakage):**
  - Se verificó formalmente la intersección de imágenes: **0 imágenes** de `val_dev` se encuentran en `train_mixed.txt` ni en las fuentes de `tiles512/manifest.json`. No existe fuga de datos.
- **Disparidad de Resoluciones:**
  - `val_dev`: Resoluciones hasta $1500 \times 1500$ píxeles.
  - `test`: Resoluciones de $600 \times 600$ a $1280 \times 1024$ píxeles.
  - `oculto`: Resoluciones de $600 \times 463$ a $1280 \times 1024$ píxeles.
  - Redimensionar una imagen de $1280 \times 1024$ a $512 \times 512$ encoge un bote pequeño de $15 \times 15$ px a apenas $5 \times 5$ px, desapareciendo por completo del stride de detección de YOLO (stride 8/16/32).
  - Por esta razón, el modelo entrenado en tiles requiere obligatoriamente inferencia por ventana deslizante (SAHI).

---

## Soluciones Implementadas y Arquitectura de Código

### 1. `entrenar_prueba.py`
- **Corrección de Bloqueo de Archivos en Windows/OneDrive:** Implementado bucle de reintentos con respaldo atómico en `guardar()` y callbacks de Ultralytics protegidos con `try/except`.
- **Eliminación de Distorsión de Rotación:** Fijado `degrees=0.0` (las cajas AABB en Ultralytics rotadas continuamente entre -90° y +90° inflan artificialmente el área de barcos alargados, destruyendo el IoU).
- **Hiperparámetros de Competición:**
  - Optimizador: `AdamW` (`lr0=0.003`, `lrf=0.01`, `weight_decay=0.0005`).
  - Pérdidas: `box=8.5` (máxima penalización de localización para mAP@0.5:0.95), `cls=0.8`, `dfl=1.5`.
  - Augmentation satelital: `fliplr=0.5`, `flipud=0.5`, `scale=0.5`, `mosaic=0.7`, `mixup=0.1`, `close_mosaic=5`.

### 2. `ensemble.py`
- **Inferencia Sliced (SAHI) con Padding Reflexivo:**
  - Divide imágenes satelitales en tiles de $512 \times 512$ con solapamiento del 25% (stride 384 px).
  - Incluye padding con `cv2.BORDER_REFLECT` para imágenes con dimensiones menores a 512 px (ej. `sl_efde06ad.jpg` de $664 \times 463$).
  - Reproyecta con precisión subpíxel al sistema de coordenadas de la imagen completa.
  - Combina la vista global con las vistas locales vía WBF.
- **Weighted Boxes Fusion (WBF) Puro:**
  - Reemplaza NMS duro por clustering y promediado de cajas ponderadas por confianza efectiva:
    $$x_{\text{fused}} = \frac{\sum w_i \cdot x_i}{\sum w_i}$$
  - Aumenta drásticamente la tasa de acierto en cortes de IoU altos (0.75, 0.85, 0.95).
- **Calibración Monótona de Confianza por Clase (Gamma Calibration):**
  - Mapea el umbral óptimo de cada clase $T_c$ calculado en validación local al umbral global de la plataforma ($T_{\text{global}} = 0.33$):
    $$s_{\text{nuevo}} = s^{\gamma_c}, \quad \text{donde } \gamma_c = \frac{\ln(T_{\text{global}})}{\ln(T_c)}$$
  - Umbrales óptimos calibrados:
    * `small_boat`: $T_0 = 0.37 \implies \gamma_0 = 1.115$
    * `cargo_ship`: $T_1 = 0.33 \implies \gamma_1 = 1.000$
    * `fishing_vessel`: $T_2 = 0.20 \implies \gamma_2 = 0.689$ (dispara el Recall de 22% a 48%)
    * `warship`: $T_3 = 0.53 \implies \gamma_3 = 1.748$
    * `work_vessel`: $T_4 = 0.48 \implies \gamma_4 = 1.509$
  - Propiedad matemática fundamental: Al ser una función estrictamente monótona creciente, preserva el orden de ranking de las detecciones por clase, protegiendo el mAP@0.5:0.95 intacto mientras maximiza simultáneamente el F1 macro.
- **Fusión Selectiva entre PC 1 y PC 2:**
  - Ponderación especializada por capacidad de modelo:
    * PC 2 (1024px Full Image): Ponderación dominante en embarcaciones grandes (`warship`: 1.5, `cargo_ship`: 1.4).
    * PC 1 (Tiles 512px Sliced): Ponderación dominante en embarcaciones pequeñas (`small_boat`: 1.4, `fishing_vessel`: 1.4, `work_vessel`: 1.4).

### 3. `generar_envio.py`
- Limpieza automática de predicciones: elimina cajas degeneradas ($w < 0.0005$ o $h < 0.0005$).
- Conserva cola de baja confianza (piso de exportación `conf=0.005`) para que el cálculo oficial de la curva Precision-Recall sume todo el Recall posible en mAP.

---

## Estado Actual de Equipos

- **PC 1 (Local, RTX 4050 6GB):**
  - Corrida: `E2_A_tiles512_seed42` (YOLO26 en dataset mixto de 16.725 imágenes/tiles, 20 épocas, batch 16, imgsz 512).
  - Tasa: ~6.2 it/s (~2.8 minutos por época).
- **PC 2 (Remoto, Teammate):**
  - Corrida: `E1b_B_1024_seed42` (YOLO26 a resolución nativa 1024 px, 25 épocas).

---

## Hoja de Ruta para Superar 0.6000

1. **Envío Inmediato (Envío #3):** Subir `envio_spacelab_1024_calibrado.zip` con umbral `0.33`.
2. **Inferencia Sliced al Terminar PC 1:**
   ```powershell
   .\.venv\Scripts\python.exe ensemble.py sliced --model runs/E2_A_tiles512_seed42/weights/best.pt --images spacelab_dataset/images/oculto --out-dir runs/E2_sliced_oculto --tile-size 512 --overlap 0.25
   ```
3. **Fusión Selectiva Final (PC 1 Sliced + PC 2 1024px):**
   ```powershell
   .\.venv\Scripts\python.exe ensemble.py blend --pc1-dir runs/E2_sliced_oculto --pc2-dir runs/E1b_B_1024_oculto --out-dir runs/ensemble_pc1_pc2 --out-zip envio_spacelab_wbf_ensemble.zip --t-global 0.33
   ```
4. **Verificación Previa a la Subida:** Comprobar siempre con `kit_participantes/formato_envio.py` para asegurar 0 advertencias y 1.743 archivos.


## Actualización verificada de la hora extra — 2026-11-07

- PC 2 importado: `best_pc2_1024.pt`, E1b_B_1024_seed42, 25 épocas registradas.
- TTA 1024 + WBF IoU 0.60 + gamma: score val_dev 0.539894; ZIP `envio_solo_pc2_1024_wbf.zip`.
- Mejor opción medida: `seleccion_por_clase`, score val_dev 0.561929; ZIP `envio_mejor_validado_hora_extra.zip`.
- `transferencia_pc2.zip` contiene 889 predicciones val y 1743 oculto sin gamma, modelo y metadatos.
- Leer `RESULTADOS_HORA_EXTRA.md` y los JSON de `runs/hora_extra_pc2/` y `runs/hora_extra_fusion/` para métricas, comandos y procedencia.
- Las estimaciones anteriores de score no son mediciones. Estos resultados son de validación local, no del leaderboard. WBF no garantiza mejoras.
- No fusionar carpetas calibradas ni aplicar gamma dos veces. Usar predicciones crudas y recalibrar en val_dev.


## Resultado confirmado en plataforma — envío de las 16:49

El usuario confirmó que subió `envio_mejor_validado_hora_extra.zip`. La captura del 7 de noviembre muestra: **score 0.4655**, **mAP50:95 0.4131**, **F1 0.5877**, puesto **7**, 4 envíos. La diferencia respecto al score val_dev 0.561929 es 0.096429. El score anterior confirmado era 0.4138: mejora absoluta 0.0517.

La auditoría `auditar_test_hora_extra.py` finalizó sobre las 801 imágenes etiquetadas del test local, sin recalibrar ni seleccionar parámetros en test: combinación **0.481963** (mAP **0.425585**, F1 **0.613514**), PC 2 TTA **0.467600**, PC 1 TTA **0.448292**, PC 1 completa **0.435451**. La combinación conserva la mejora, aunque val_dev sobreestima su resultado externo. Pesqueros: AP **0.141210**, recall **0.391041**. Ver `AUDITORIA_TEST801.md` y `runs/auditoria_test801/results.json`. Los ZIP anteriores no se modificaron.


## Cierre de la ventana final de 26 minutos

- Archivo para subir: `entrega_final.zip`. Selección: `baseline_postprocess`.
- Val_dev 0.561929; test801 0.481963. Son resultados locales, no leaderboard.
- Leer `ENTREGA_FINAL.md` y `runs/final26/final_selected.json` para reproducir la decisión.
- El test local se usó para comparar candidatos preseleccionados en esta fase final; no presentarlo como prueba totalmente independiente.
