#!/usr/bin/env bash
# macOS / Linux
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Lần đầu chạy: đang cài thư viện..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python download_models.py
.venv/bin/python -W ignore app.py
