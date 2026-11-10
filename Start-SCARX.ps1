$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$port = 8765
for ($i = 0; $i -lt $args.Count; $i++) {
    if ($args[$i] -eq '--port' -and ($i + 1) -lt $args.Count) { $port = [int]$args[$i + 1] }
}
$listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    Write-Error "Port $port is already in use. Stop that SCAR-X process or pass another --port. A second launch is not started."
}
python (Join-Path $PSScriptRoot 'run_app.py') @args
