#!/usr/bin/env sh
set -eu

REPO="vedpatil-dev/butler-agent"
BASE_URL="https://github.com/$REPO/releases/latest/download"
BIN_DIR="${HOME}/.local/bin"
mkdir -p "$BIN_DIR"

OS="$(uname -s)"
ARCH_RAW="$(uname -m)"
case "$ARCH_RAW" in
    x86_64|amd64) ARCH="x86_64" ;;
    arm64|aarch64) ARCH="arm64" ;;
    *)
        echo "Unsupported architecture: $ARCH_RAW"
        exit 1
        ;;
esac

case "$OS" in
    Linux)
        SUFFIX=""
        if ldd --version 2>&1 | grep -qi musl || ls /lib/ld-musl-*.so.1 >/dev/null 2>&1; then
            SUFFIX="-musl"
        fi
        URL="$BASE_URL/butler-linux-${ARCH}${SUFFIX}"
        echo "Downloading Butler binary for Linux ($ARCH${SUFFIX})..."
        curl -fsSL "$URL" -o "$BIN_DIR/butler"
        chmod +x "$BIN_DIR/butler"
        ln -sf "$BIN_DIR/butler" "$BIN_DIR/agent"
        ;;
    Darwin)
        URL="$BASE_URL/Butler-macOS.dmg"
        TMP_DMG="$(mktemp /tmp/butler.XXXXXX.dmg)"
        MNT_DIR="$(mktemp -d /tmp/butler-mnt.XXXXXX)"
        echo "Downloading Butler DMG for macOS ($ARCH)..."
        curl -fsSL "$URL" -o "$TMP_DMG"
        hdiutil attach "$TMP_DMG" -mountpoint "$MNT_DIR" -nobrowse -quiet
        cp "$MNT_DIR/butler" "$BIN_DIR/butler"
        chmod +x "$BIN_DIR/butler"
        ln -sf "$BIN_DIR/butler" "$BIN_DIR/agent"
        hdiutil detach "$MNT_DIR" -quiet || true
        rm -rf "$TMP_DMG" "$MNT_DIR"
        ;;
    *)
        echo "Unsupported OS: $OS (download binaries from https://github.com/$REPO/releases)"
        exit 1
        ;;
esac

echo ""
echo "Installed to $BIN_DIR/butler"
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) echo "Add to PATH: export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac
echo "Run:"
echo "  butler setup --install   # check/install GPU/NPU drivers & runtimes"
echo "  butler                   # start chatting"
