# Entrega final

Archivo listo: `C:\Users\gniko\OneDrive\Documentos\tarea javeriana\Entrega\entrega_final.zip`.

Tipo seleccionado: **baseline_postprocess**.
Score val_dev: **0.561929**.
Score test local 801: **0.481963**.
mAP test local: **0.425585**; F1: **0.613514**.

Estos puntajes son locales. El nuevo archivo requiere evaluación en la plataforma. El envío anterior obtuvo 0.4655 en el leaderboard y 0.481963 en el test local.

El entrenamiento adicional usó exclusivamente train_dev. Se mantuvieron val_dev y el test fuera del entrenamiento. Los parámetros se ajustaron en val_dev; test801 se utilizó para comparar hasta tres candidatos previamente seleccionados y rechazar retrocesos, por lo que ya no es una estimación completamente independiente de la selección final.

Épocas adicionales registradas: 4. Parada de entrenamiento prevista: 17:17 Bogotá.
Configuración: YOLO26n inicializado con PC2, 1024 px, batch 8, AdamW 0.0002, sin mosaic/mixup, escala 0.25, flips horizontal/vertical y pérdidas box=8.5, cls=0.7.

Verificado: 1743 TXT de predicciones, nombres exactos, 0 advertencias del lector oficial, umbral 0.33, máximo 300 detecciones por imagen, coordenadas y números finitos, integridad ZIP.

SHA-256: `a9497efe68ddc9c4e942e94053c3d5f5d37c975f2e03a5ba68fe4b36394494af`.

Detalles reproducibles: `runs/final26/final_selected.json`, `comparison_val.json`, `test_comparison.json` cuando existe, y `runs/E3_final_pc2_refine_1024/`.
