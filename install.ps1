$ErrorActionPreference = "Stop"
$Repo = "git+https://github.com/vedpatil-dev/butler-agent.git@V1"

Write-Host "Installing Butler (local-agent)..."

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (manages Python 3.12+ automatically)..."
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

uv tool install --force --python 3.12 $Repo

Write-Host ""
Write-Host "Installed! Run:"
Write-Host "  butler setup --install   # check/install GPU/NPU drivers & runtimes"
Write-Host "  butler                   # start chatting"
