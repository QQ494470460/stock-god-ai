@echo off
chcp 65001 >nul
title 选股神 - 训练模型
cd /d "%~dp0"
echo ==============================
echo   选股神 Stock God AI
echo   开始训练AI模型...
echo ==============================
echo.
echo ⚠ 训练可能需要较长时间
echo 请保持电脑运行，不要关闭此窗口
echo ==============================
echo.
python scripts/train.py
echo.
echo ==============================
echo   训练完成！
echo ==============================
pause