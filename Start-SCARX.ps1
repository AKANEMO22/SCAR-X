$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python (Join-Path $PSScriptRoot 'run_app.py') @args
