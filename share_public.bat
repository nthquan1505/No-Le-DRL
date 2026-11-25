@echo off
chcp 65001 > nul
echo ============================================================
echo   Tao link Public Internet cho VA DRL (port 8080)
echo   Copy link https://...lhr.life ben duoi de gui cho moi nguoi!
echo   (Nho mo run_web.bat truoc khi chay file nay)
echo ============================================================
ssh -T -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -R 80:127.0.0.1:8080 nokey@localhost.run
pause
