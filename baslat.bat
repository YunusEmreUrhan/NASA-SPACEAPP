@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================================
echo   TERRA-TWIN: Otomatik Kurulum ve Baslatma Araci
echo ========================================================

if not exist "venv\Scripts\python.exe" (
    echo [1/3] Sanal ortam ^(venv^) olusturuluyor...
    python -m venv venv
)

echo [2/3] Gerekli kutuphaneler kontrol ediliyor...
.\venv\Scripts\python.exe -m pip install --upgrade pip -q
.\venv\Scripts\python.exe -m pip install -r requirements.txt -q

echo [3/3] Sistem dogrulaniyor ve baslatiliyor...
.\venv\Scripts\python.exe run.py
pause
