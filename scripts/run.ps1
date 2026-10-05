# Run the app using a venv outside OneDrive (avoids cloud-file locks on native libs).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = "C:\Users\sadhu\mir-venv"
if (-not (Test-Path "$Venv\Scripts\python.exe")) {
    Write-Host "Creating venv at $Venv ..."
    python -m venv $Venv
    & "$Venv\Scripts\python.exe" -m pip install -r "$Root\backend\requirements.txt"
}
$env:PYTHONPATH = $Root
Set-Location $Root
& "$Venv\Scripts\python.exe" -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
