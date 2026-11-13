#!/usr/bin/env sh
set -eu

REPO="vedpatil-dev/butler-agent"
BASE_URL="https://github.com/$REPO/releases/latest/download"
BIN_DIR="${HOME}/.local/bin"
mkdir -p "$BIN_DIR"

OS="$(uname -s)"
case "$OS" in
    Linux)
        URL="$BASE_URL/butler-linux-x86_64"
        echo "Downloading Butler binary for Linux..."
        curl -fsSL "$URL" -o "$BIN_DIR/butler"
        chmod +x "$BIN_DIR/butler"
        ln -sf "$BIN_DIR/butler" "$BIN_DIR/agent"
        ;;
    Darwin)
        URL="$BASE_URL/Butler-macOS.dmg"
        TMP_DMG="$(mktemp /tmp/butler.XXXXXX.dmg)"
        MNT_DIR="$(mktemp -d /tmp/butler-mnt.XXXXXX)"
        echo "Downloading Butler DMG for macOS..."
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
