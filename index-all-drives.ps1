# index-all-drives.ps1  -- PowerShell script (machine-independent)
# Indexes EVERY plugged-in drive into %SystemDrive%\index_out (usually the C: drive) with
# bilingual (VI+EN) OCR. AUTO-RESUMES on interruption AND on crashes -- it keeps retrying
# with --resume as long as each attempt commits MORE files; it only gives up if two
# attempts in a row make ZERO progress (a single file is genuinely crashing the parser).
# Re-run anytime to continue. Delete the output folder to rebuild from scratch.
#
# Paths resolve relative to THIS script (the repo root), so it runs unchanged on any machine.
# RUN IN VS CODE:
#   powershell -ExecutionPolicy Bypass -File .\index-all-drives.ps1
#   (or press F5 with the PowerShell extension). Optional: add  -Out 'D:\my_index'.
#   Stop with Ctrl+C; re-run to continue.
param([string]$Out)

$ErrorActionPreference = 'Continue'
$root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }   # repo root = this script's folder
Set-Location $root

$exe    = Join-Path $root '.venv\Scripts\claude-index.exe'
$py     = Join-Path $root '.venv\Scripts\python.exe'
$out    = if ($Out) { $Out } else { Join-Path $env:SystemDrive 'index_out' }   # default: %SystemDrive%\index_out (usually C:)
$db     = Join-Path $out 'index.sqlite'
$marker = Join-Path $out '.DONE'
$cores  = [Environment]::ProcessorCount         # all logical cores
$env:OMP_THREAD_LIMIT = '1'                      # 1 thread per Tesseract -> no oversubscription

function Get-IndexedCount {
    if (-not (Test-Path $db)) { return 0 }
    try { return [int](& $py -c "import sqlite3,sys; print(sqlite3.connect(sys.argv[1]).execute('SELECT COUNT(*) FROM files').fetchone()[0])" $db) }
    catch { return 0 }
}

$drives = Get-Volume |
  Where-Object { $_.DriveLetter -and $_.DriveType -in 'Fixed','Removable' -and $_.FileSystemType } |
  Sort-Object DriveLetter | ForEach-Object { "$($_.DriveLetter):\" }
Write-Host "Repo   : $root"
Write-Host "Drives : $($drives -join '   ')"
Write-Host "Workers: $cores      Output: $out"

if (-not (Test-Path $exe)) {
    Write-Host "claude-index not found at $exe -- run scripts\install_windows.ps1 first."; return
}
if (Test-Path $marker) { Write-Host "Already complete ($marker). Delete $out to rebuild."; return }

$resume = Test-Path $db
Write-Host ("Mode   : " + $(if ($resume) { 'RESUME existing index' } else { 'FRESH build' }))

$logdir = Join-Path $root 'logs'; New-Item -ItemType Directory -Force $logdir | Out-Null
$stamp  = ([TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow,'SE Asia Standard Time')).ToString('yyyyMMddHHmm') + 'VN'

$attempt = 0; $noProgress = 0
$prevCount = Get-IndexedCount
do {
    $attempt++
    $flags = @('index') + $drives + @('--out', $out, '--ocr', 'auto', '--workers', $cores)
    if ($resume) { $flags += '--resume' }
    $log = Join-Path $logdir "${stamp}_attempt$attempt.log"
    Write-Host "`n=== attempt #$attempt  ($(if ($resume) { 'resume' } else { 'fresh' }))  indexed so far: $prevCount  ->  $log ==="

    & $exe @flags 2>&1 | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
    $resume = $true                              # every rerun after the first CONTINUES; never wipes
    if ($code -eq 0) { break }

    $nowCount = Get-IndexedCount
    if ($nowCount -gt $prevCount) {
        Write-Host ("Crashed (exit {0}) but progressed {1} -> {2} files. Auto-resuming in 15s..." -f $code, $prevCount, $nowCount)
        $noProgress = 0
    } else {
        $noProgress++
        Write-Host ("Crashed (exit {0}) with NO new files (still {1}). Strike {2}/2." -f $code, $nowCount, $noProgress)
        if ($noProgress -ge 2) {
            Write-Host "Stopping: two attempts in a row made zero progress -> one specific file is crashing the parser."
            Write-Host "  Open the END of $log, find the last file (or 'MuPDF error'), move/rename it, then re-run."
            break
        }
    }
    $prevCount = $nowCount
    Start-Sleep -Seconds 15
} while ($attempt -lt 200)

if ($code -eq 0) {
    New-Item -ItemType File -Force -Path $marker | Out-Null
    Write-Host "`nDONE in $attempt attempt(s).  Index: $out   (sidecars: $out\sidecar)"
} else {
    Write-Host "`nStopped after $attempt attempt(s) (last exit $code). Re-run this file to continue."
}
