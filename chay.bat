@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Lan dau chay: dang cai dat thu vien, vui long doi vai phut...
  python -m venv .venv || (echo Can cai Python 3.10 tro len: https://www.python.org/downloads/ & pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -r requirements.txt || (pause & exit /b 1)
)
.venv\Scripts\python download_models.py || (pause & exit /b 1)
.venv\Scripts\python -W ignore app.py
pause
