$ErrorActionPreference = "Stop"
$Repo = "vedpatil-dev/butler-agent"
$Url = "https://github.com/$Repo/releases/latest/download/butler-windows-x86_64.exe"
$InstallDir = Join-Path $env:LOCALAPPDATA "Programs\Butler"
$BinPath = Join-Path $InstallDir "butler.exe"

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Write-Host "Downloading Butler executable for Windows..."
Invoke-WebRequest -Uri $Url -OutFile $BinPath

$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$InstallDir", "User")
    $env:Path = "$env:Path;$InstallDir"
}

Write-Host ""
Write-Host "Installed to $BinPath"
Write-Host "Run:"
Write-Host "  butler setup --install   # check/install GPU/NPU drivers & runtimes"
Write-Host "  butler                   # start chatting"
