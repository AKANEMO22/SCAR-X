param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$scarEntry = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'run_app.py'))
$scarListener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $scarListener) { Write-Output 'SCAR-X is not listening on this port.'; exit 0 }
$scarProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($scarListener.OwningProcess)"
if ($scarProcess.CommandLine -notmatch [regex]::Escape($scarEntry)) {
    throw 'Cannot verify this process belongs to this SCAR-X checkout. Use Ctrl+C in its own terminal.'
}
Stop-Process -Id $scarProcess.ProcessId
Write-Output 'Stopped SCAR-X. The last SQLite-committed checkpoint remains saved.'
