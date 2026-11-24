#!/bin/bash
# Starts VanishCam on macOS: Python environment, components and webcam.
set -euo pipefail
cd "$(dirname "$0")"

fail() {
  echo
  echo "Vanish Lab could not start. See the README for common fixes."
  read -r -p "Press Enter to close..." _ || true
  exit 1
}
trap fail ERR

VENV_DIR=".venv-local"
VENV_PY="$VENV_DIR/bin/python"

if [ -x "$VENV_PY" ] && ! "$VENV_PY" -c "import sys" >/dev/null 2>&1; then
  echo "[Vanish Lab] The Python environment stopped working, most likely because Python was upgraded. Rebuilding it..."
  rm -rf "$VENV_DIR"
fi

if [ ! -x "$VENV_PY" ]; then
  echo "[Vanish Lab] Creating the local Python environment..."
  for PYTHON in python3.13 python3.12 /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 \
                /usr/local/bin/python3.13 /usr/local/bin/python3.12; do
    if command -v "$PYTHON" >/dev/null 2>&1; then
      "$PYTHON" -m venv "$VENV_DIR"
      break
    fi
  done
  if [ ! -x "$VENV_PY" ]; then
    echo "[Vanish Lab] Python 3.12 or 3.13 is required: pygame has no packages for Python 3.14 and later."
    false
  fi
fi

if ! "$VENV_PY" -c "import cv2, mediapipe, numpy, pyvirtualcam" >/dev/null 2>&1; then
  echo "[Vanish Lab] Installing the required components. Internet is needed only this once..."
  "$VENV_PY" -m pip install --upgrade pip >/dev/null
  "$VENV_PY" -m pip install -r requirements.txt
fi

LOG="$HOME/Library/Logs/VanishCam.log"
mkdir -p "$(dirname "$LOG")"
if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt 2000000 ]; then
  mv "$LOG" "$LOG.1"
fi

echo "[Vanish Lab] Starting the webcam... (log: $LOG)"
PYTHONUNBUFFERED=1 "$VENV_PY" vanish_cam.py "$@" 2>&1 | tee -a "$LOG"
