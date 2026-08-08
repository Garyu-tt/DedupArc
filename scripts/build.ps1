$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "build-windows.ps1") @args
