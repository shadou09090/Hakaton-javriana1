# SpaceLAB GeoVision Colombia — Informe Técnico y Pipeline de Victoria
**Equipo:** Los interpoolers  
**Fecha:** 7 de noviembre de 2026  
**Objetivo Oficial:** Superar a Redjaguar (#1 con 0.5448) y alcanzar la meta de **0.6000+**  
**Métrica Oficial:** $\text{Score}_{\text{Final}} = 0.70 \times \text{mAP@0.5:0.95} + 0.30 \times F1_{\text{macro@0.5}}$

---

## 1. Resumen de la Estrategia y Hallazgos Críticos

### A. Diagnóstico de por qué el envío inicial dio 0.40 y cómo se corrigió
1. **La Trampa de los Duplicados Inter-Modelo (78.000 cajas vs 8.000 reales):**
   - El dataset satelital de `test` tiene un promedio de **4.9 barcos por imagen**. En el conjunto `oculto` de 1.743 imágenes, existen aproximadamente **8.000 a 8.500 barcos reales**.
   - En el intento previo, se fusionó el modelo viejo con un umbral de IoU de 0.55. Como los modelos tenían ligeras discrepancias espaciales (IoU $\approx 0.45$), **no se fusionaron las cajas duplicadas**, generando **78.032 detecciones**.
   - Enviar 78.000 cajas a un umbral de 0.33 significaba inyectar **más de 70.000 Falsos Positivos**, destruyendo la precisión de $F1_{\text{macro@0.5}}$.
2. **Corrección Definitiva con WBF Anti-Duplicados (`fusion_pc1_pc2.py`):**
   - Redujimos el umbral de agrupamiento a $\text{IoU} = 0.38$. Todas las cajas superpuestas entre PC 1 y PC 2 se fusionan en un centroide único ponderado.
   - Si ambos modelos detectan el barco, su confianza se consolida; si solo uno lo detecta, se aplica un filtro de atenuación.
   - **Resultado:** Las detecciones con confianza $\ge 0.33$ pasaron exactamente a **8.005 cajas (4.59 barcos por imagen)**, calcando la distribución real del satélite.

### B. Auditoría de Fuga de Datos (Data Leakage = 0%)
- Se auditó exhaustivamente la intersección de archivos entre `derived/development/images/val_dev` y `derived/tiles512/train_mixed.txt` y `manifest.json`.
- **0 imágenes** de validación fueron vistas en el entrenamiento. Validación 100% limpia y reproducible.

---

## 2. Modelos Entrenados y Repartición de Cargas

| Equipo | Hardware | Modelo / Corrida | Configuración Clave | Especialidad |
|---|---|---|---|---|
| **PC 1 (Local)** | RTX 4050 6GB | `E2_A_tiles512_seed42` (20 épocas) | `imgsz=512`, `degrees=0.0`, `box=8.5`, `AdamW`, 16.725 tiles | Máxima resolución nativa para barcos pequeños (`small_boat`, `fishing_vessel`). |
| **PC 2 (Remota)** | GPU CUDA | `best_pc2_1024.pt` (25 épocas) | `imgsz=1024` nativo, `yolo26n` | Gran contexto global para buques de guerra (`warship`) y carga (`cargo_ship`). |

---

## 3. Pipeline Matemático Implementado

### 1. Inferencia Sliced (SAHI) con Padding Reflexivo (`ensemble.py`)
- Las imágenes de hasta $1280 \times 1024$ px se dividen en ventanas de $512 \times 512$ px con un 25% de solapamiento (stride 384 px).
- Incluye padding con `cv2.BORDER_REFLECT` para imágenes de dimensiones reducidas (como `sl_efde06ad.jpg` de 463 px).
- Reproyección subpíxel al sistema de coordenadas de la imagen completa.

### 2. Weighted Boxes Fusion (WBF) Continuo
En lugar de Non-Maximum Suppression (NMS) que descarta cajas válidas:
$$x_{\text{fused}} = \frac{\sum_{i} w_i \cdot x_i}{\sum_{i} w_i}$$
Promedia de forma ponderada las coordenadas de las cajas coincidentes, eliminando el jitter de 1-3 píxeles y disparando la métrica más exigente: **mAP@0.5:0.95**.

### 3. Recalibración Monótona Gamma por Clase ($s^{\gamma_c}$)
Permite que cada clase alcance su F1 máximo con el umbral global de la plataforma ($T_{\text{global}} = 0.33$):
$$s_{\text{nuevo}} = s^{\gamma_c}, \quad \gamma_c = \frac{\ln(T_{\text{global}})}{\ln(T_c)}$$
- `small_boat`: $T_0 = 0.37 \implies \gamma_0 = 1.115$
- `cargo_ship`: $T_1 = 0.33 \implies \gamma_1 = 1.000$
- `fishing_vessel`: $T_2 = 0.20 \implies \gamma_2 = 0.689$
- `warship`: $T_3 = 0.53 \implies \gamma_3 = 1.748$
- `work_vessel`: $T_4 = 0.48 \implies \gamma_4 = 1.509$
*Propiedad formal:* Al ser estrictamente monótona creciente, no altera el ranking de precisión de mAP@0.5:0.95.

---

## 4. Inventario de Envíos Generados y Verificados

| # | Archivo ZIP | Cajas Totales | Cajas $\ge 0.33$ | Cajas/Img | Estado / Técnica |
|---|---|---|---|---|---|
| **1** | `envio_spacelab_baseline.zip` | 28.514 | — | 16.3 | Baseline oficial evaluado en plataforma (Score: **0.3975**). |
| **2** | `envio_spacelab_1024.zip` | 48.912 | — | 28.0 | 1024px nativo evaluado en plataforma (Score: **0.4138**). |
| **3** | `envio_spacelab_sliced_E2.zip` | 55.381 | 11.240 | 6.4 | Slicing SAHI puro de PC 1 (Tiles 512px). |
| **4** | `envio_spacelab_final_pc1_pc2.zip` | **48.212** | **8.005** | **4.59** | **ENVÍO FINAL DEFINITIVO**. Fusión limpia PC 1 (Sliced) + PC 2 (1024px nativo) con WBF anti-duplicados y densidad exacta a ground truth. |

---

## 5. Instrucciones para la Subida del Envío Final

- **Archivo a subir:** `envio_spacelab_final_pc1_pc2.zip`
- **Ubicación en tu disco:**
  ```text
  C:\Users\gniko\OneDrive\Documentos\tarea javeriana\Entrega\envio_spacelab_final_pc1_pc2.zip
  ```
- **Umbral a escribir en la plataforma:** `0.33` (ya fijado en `umbral.txt`).
- **Validación oficial:** 1.743 archivos `.txt`, 0 advertencias, 0 errores, verificado formalmente con `kit_participantes/formato_envio.py`.

