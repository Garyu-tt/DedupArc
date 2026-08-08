param(
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
$Build = Join-Path $Root 'build'
$Dist = Join-Path $Root 'dist'
$Payload = Join-Path $Root 'cmd\deduparc-setup\payload'
$VersionSource = Get-Content (Join-Path $Root 'internal\version\version.go') -Raw
if ($VersionSource -notmatch 'Current\s*=\s*"([^"]+)"') { throw 'Не удалось определить версию DedupArc.' }
$Version = $Matches[1]

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Не найдена команда '$Name'."
    }
}

function New-PayloadManifest([string]$Dir) {
    $Files = @{}
    Get-ChildItem -File -Recurse $Dir | Where-Object { $_.Name -ne 'payload-manifest.json' } | ForEach-Object {
        $Rel = $_.FullName.Substring($Dir.Length + 1).Replace('\\','/')
        $Files[$Rel] = (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLowerInvariant()
    }
    $Json = @{ files = $Files } | ConvertTo-Json -Depth 4
    $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText((Join-Path $Dir 'payload-manifest.json'), $Json, $Utf8NoBom)
}

Require-Command 'go'
Remove-Item -Recurse -Force $Build,$Dist -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $Build,$Dist | Out-Null

Push-Location $Root
try {
    if (-not $SkipTests) {
        go test ./...
        go vet ./...
    }


    $env:CGO_ENABLED='0'
    $env:GOOS='windows'
    $env:GOARCH='amd64'

    go build -trimpath -ldflags '-s -w' -o (Join-Path $Build 'deduparc-core.exe') .\cmd\deduparc
    go build -trimpath -ldflags '-s -w -H=windowsgui' -o (Join-Path $Build 'DedupArc.exe') .\cmd\deduparc-launcher
    go build -trimpath -ldflags '-s -w -H=windowsgui' -o (Join-Path $Build 'DedupArcUpdater.exe') .\cmd\deduparc-updater

    Remove-Item -Recurse -Force $Payload -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $Payload | Out-Null
    Copy-Item (Join-Path $Build 'DedupArc.exe') $Payload
    Copy-Item (Join-Path $Build 'DedupArcUpdater.exe') $Payload
    Copy-Item (Join-Path $Build 'deduparc-core.exe') $Payload
    Copy-Item (Join-Path $Root 'assets\DedupArc.ico') $Payload
    Copy-Item (Join-Path $Root 'README.md') $Payload
    Copy-Item (Join-Path $Root 'LICENSE') $Payload
    Copy-Item (Join-Path $Root 'THIRD_PARTY_NOTICES.md') $Payload
    New-PayloadManifest $Payload

    go build -trimpath -ldflags '-s -w -H=windowsgui' -o (Join-Path $Build 'DedupArcSetup.exe') .\cmd\deduparc-setup

    $Portable = Join-Path $Dist "DedupArc-$Version-windows-amd64"
    New-Item -ItemType Directory -Force -Path $Portable | Out-Null
    Copy-Item (Join-Path $Payload '*') $Portable
    Copy-Item (Join-Path $Build 'DedupArcSetup.exe') (Join-Path $Portable 'Uninstall.exe')

    $UpdateZip = Join-Path $Dist "DedupArc-update-$Version.zip"
    Compress-Archive -Path (Join-Path $Portable '*') -DestinationPath $UpdateZip -CompressionLevel Optimal
    $SetupOut = Join-Path $Dist "DedupArc-$Version-Setup.exe"
    Copy-Item (Join-Path $Build 'DedupArcSetup.exe') $SetupOut

    if ($env:DEDUPARC_RELEASE_KEY) {
        $Url = "https://github.com/Garyu-tt/DedupArc/releases/download/v$Version/DedupArc-update-$Version.zip"
        go run .\cmd\deduparc-release manifest $Version $Url $UpdateZip "Обновление DedupArc $Version"
    } else {
        Write-Warning 'DEDUPARC_RELEASE_KEY не задан: update.json и update.sig не создаются.'
    }

    $Checksums = Join-Path $Dist 'SHA256SUMS.txt'
    $ChecksumFiles = @($SetupOut, $UpdateZip)
    if (Test-Path (Join-Path $Dist 'update.json')) { $ChecksumFiles += (Join-Path $Dist 'update.json') }
    if (Test-Path (Join-Path $Dist 'update.sig')) { $ChecksumFiles += (Join-Path $Dist 'update.sig') }
    $Lines = foreach ($File in $ChecksumFiles) {
        $Hash = (Get-FileHash -Algorithm SHA256 $File).Hash.ToLowerInvariant()
        "$Hash  $([System.IO.Path]::GetFileName($File))"
    }
    $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines($Checksums, $Lines, $Utf8NoBom)

    Write-Host ''
    Write-Host 'Готово:'
    Write-Host "  $SetupOut"
    Write-Host "  $UpdateZip"
}
finally {
    Pop-Location
}
