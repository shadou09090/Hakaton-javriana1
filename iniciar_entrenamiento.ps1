$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" entrenar_prueba.py --name E0_A_768_seed42 --epochs 25 --workers 2 --batch 8
