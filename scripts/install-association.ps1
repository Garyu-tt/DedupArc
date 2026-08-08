$ErrorActionPreference = 'Stop'
$Root = Join-Path $env:LOCALAPPDATA 'Programs\DedupArc'
$Exe = Join-Path $Root 'DedupArc.exe'
$Icon = Join-Path $Root 'DedupArc.ico'
if (-not (Test-Path $Exe)) { throw 'DedupArc не установлен.' }
reg add 'HKCU\Software\Classes\.darc' /ve /d 'DedupArc.Archive' /f | Out-Null
reg add 'HKCU\Software\Classes\DedupArc.Archive' /ve /d 'Архив DedupArc' /f | Out-Null
reg add 'HKCU\Software\Classes\DedupArc.Archive\DefaultIcon' /ve /d $Icon /f | Out-Null
reg add 'HKCU\Software\Classes\DedupArc.Archive\shell\open\command' /ve /d ('"' + $Exe + '" "%1"') /f | Out-Null
Write-Host 'Ассоциация .darc обновлена.'
