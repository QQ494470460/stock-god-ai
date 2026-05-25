@echo off
chcp 65001 >nul
title 选股神 - AI选股看板
cd /d "%~dp0"
echo ==============================
echo   选股神 Stock God AI
echo   正在启动... 稍等片刻
echo ==============================
echo.
echo 看板启动后，浏览器会自动打开
echo 如果没自动打开，请访问: http://localhost:8501
echo.
echo 按 Ctrl+C 可以关闭看板
echo ==============================
echo.
streamlit run src/ui/app.py --server.headless true
pause