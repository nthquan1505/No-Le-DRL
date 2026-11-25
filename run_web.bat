@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ============================================================
echo   DRL Credits Calculation & Evaluation Checker Web App
echo   URL: http://127.0.0.1:8080
echo   Nhan Ctrl+C de dung server.
echo ============================================================
start "" http://127.0.0.1:8080
"C:\msys64\ucrt64\bin\python.exe" "%~dp0server.py" --port 8080
pause
