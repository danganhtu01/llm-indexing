# One-shot Windows setup for claude-indexing.
# Installs Tesseract (OCR) + GitHub CLI, creates a venv, installs deps, fetches data.
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

Write-Host '== claude-indexing setup ==' -ForegroundColor Cyan

# 1. System tools via winget (skip if already present)
if (-not (Get-Command tesseract -ErrorAction SilentlyContinue) -and
    -not (Test-Path 'C:\Program Files\Tesseract-OCR\tesseract.exe')) {
  Write-Host 'Installing Tesseract OCR...'
  winget install --id UB-Mannheim.TesseractOCR -e --accept-source-agreements --accept-package-agreements
}
if (-not (Get-Command gh -ErrorAction SilentlyContinue) -and
    -not (Test-Path 'C:\Program Files\GitHub CLI\gh.exe')) {
  Write-Host 'Installing GitHub CLI...'
  winget install --id GitHub.cli -e --accept-source-agreements --accept-package-agreements
}

# 2. Python venv + deps
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
  Write-Host 'Creating venv...'
  python -m venv (Join-Path $root '.venv')
}
& $py -m pip install --upgrade pip
& $py -m pip install -e $root

# 3. Dictionaries + Tesseract language data (eng + vie)
& $py (Join-Path $root 'scripts\fetch_data.py')

Write-Host ''
Write-Host 'Done. Try:' -ForegroundColor Green
Write-Host '  .\.venv\Scripts\claude-index index E:\ --out index_out'
Write-Host '  .\.venv\Scripts\claude-index search "ngan hang" --index index_out'
