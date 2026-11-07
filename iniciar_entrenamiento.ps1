$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$last = Join-Path $PSScriptRoot 'runs\E0_A_768_seed42\weights\last.pt'
if (Test-Path $last) {
    & "$PSScriptRoot\.venv\Scripts\python.exe" entrenar_prueba.py --name E0_A_768_seed42 --epochs 25 --workers 2 --batch 8 --resume $last
} else {
    & "$PSScriptRoot\.venv\Scripts\python.exe" entrenar_prueba.py --name E0_A_768_seed42 --epochs 25 --workers 2 --batch 8
}
