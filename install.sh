#!/usr/bin/env sh
set -eu

REPO="git+https://github.com/vedpatil-dev/butler-agent.git@V1"

echo "Installing Butler (local-agent)..."

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv (manages Python 3.12+ automatically)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

uv tool install --force --python 3.12 "$REPO"

echo ""
echo "Installed! Run:"
echo "  butler setup --install   # check/install GPU/NPU drivers & runtimes"
echo "  butler                   # start chatting"
