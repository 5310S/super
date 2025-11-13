#!/usr/bin/env bash
set -euo pipefail

echo "[+] Creating virtual environment (.venv)"
python3 -m venv .venv
echo "[+] Activating virtual environment"
source .venv/bin/activate
echo "[+] Installing supervisor package in editable mode"
pip install --upgrade pip
pip install -e '.[dev]'
echo ""
echo "Setup complete."
echo "- CLI: run 'codex-supervisor --help'"
echo "- GUI: run 'codex-supervisor-gui' or 'python3 gui.py'"
echo ""
echo "To leave the environment, run 'deactivate'."
