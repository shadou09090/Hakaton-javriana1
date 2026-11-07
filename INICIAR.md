# Ejecutar las pruebas SpaceLAB

Abre PowerShell en la raíz del proyecto. El entorno está en `.venv`.

```powershell
.\.venv\Scripts\python.exe preparar_datos.py split --output-root derived/development --workers 4
.\.venv\Scripts\python.exe preparar_datos.py tiles --output-root derived/tiles512 --workers 4
.\.venv\Scripts\python.exe entrenar_prueba.py --data derived/development/smoke.yaml --name smoke_nuevo --epochs 2 --workers 2 --batch 8
.\.venv\Scripts\python.exe entrenar_prueba.py --name E0_A_768_seed42 --epochs 25 --workers 2 --batch 8
.\.venv\Scripts\python.exe -m streamlit run app_streamlit.py --server.address 127.0.0.1
```

Cada preparación requiere una carpeta de salida nueva. Conserva el manifiesto de escenas para las siguientes pruebas. Las copias en `derived/development` permiten al cargador escribir cachés en la vista de trabajo. Los tiles de 512 px tienen 20 % de solapamiento, conservan al menos 70 % del área de cada caja intersectada y rechazan ventanas con fragmentos excluidos. Mantienen el 10 % determinista de los recortes sin cajas. El YAML `derived/tiles512/mixed.yaml` mezcla imágenes completas y tiles, con validación completa.

La primera prueba de dos épocas comprueba el flujo técnico. El baseline E0 usa imágenes completas a 768. Para probar posteriormente la mezcla, usa `--data derived/tiles512/mixed.yaml` y un nombre nuevo. Las salidas de cada entrenamiento están en `runs/<nombre>`. `status.json` informa progreso y errores, `results.csv` registra épocas y `metricas_val.json` contiene la evaluación oficial al terminar.

En la verificación local pasaron cinco pruebas geométricas. Se prepararon 5.037 imágenes para entrenamiento, 889 para validación y 11.688 tiles con 21.046 cajas. El smoke test `smoke_A_768_w0` completó dos épocas, guardó checkpoints y pasó la evaluación oficial sobre 64 imágenes. Su score de 0,00433 sirve como comprobación del flujo. La prueba de memoria posterior con batch 8 y dos workers avanzó con aproximadamente 2,23 GB reportados por el entrenador. Se desactivó la búsqueda de algoritmos cuDNN (`benchmark=False`) para reducir el costo de arranque. El tiempo y el pico real de cada corrida se registran en `status.json`.

Para reproducir el entorno en B, crea su `.venv` con Python 3.14 de 64 bits e instala el archivo `requirements-lock.txt` de una corrida terminada usando `python -m pip install -r ruta/al/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128`. Comprueba CUDA antes de entrenar.

En el segundo equipo, copia el código, kit, dataset y manifiesto, junto con las mismas versiones y pesos iniciales. Las listas/YAML generados contienen rutas absolutas. Reconstrúyelos con `preparar_datos.py split --manifest ruta/al/manifest.json --output-root derived/development` en una salida nueva. Esto verifica los hashes y conserva la partición. Cambia a 1024 con `--imgsz 1024 --name E1b_B_1024_seed42`.

La interfaz es de consulta local y actualiza el monitor cada cinco segundos. El archivo `iniciar_interfaz.ps1` permite iniciarla manualmente. Los checkpoints de una corrida pueden reanudarse con `--resume runs/<nombre>/weights/last.pt --name <nombre>`, manteniendo la configuración original.
