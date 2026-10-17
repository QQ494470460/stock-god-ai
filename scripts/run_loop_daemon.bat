@echo off
REM ===========================================================================
REM 选股神AI · 闭环交易常驻启动器
REM
REM 由「启动文件夹」快捷方式在用户登录时调用（无需管理员权限）。
REM
REM 特性：
REM   · 脚本自带单实例锁（data\.run_loop.lock）：重复拉起不会产生第二份实例，
REM     也绝不会对同一份计划重复下单。
REM   · 异常退出（非「已有实例」）自动重启，最多 5 次，避免无限空转。
REM   · 退出码 3 = 已有实例在跑，直接放弃重启（此时另一个实例才是正主）。
REM   · 用 pythonw.exe（无控制台窗口）+ stdout 重定向到日志文件；
REM     run_loop.py 的 tick 行带 flush=True，重定向不会丢日志。
REM   · 日志超过 10MB 自动轮转为 run_loop.log.1（只保留一份历史）。
REM ===========================================================================
setlocal enabledelayedexpansion

set "ROOT=C:\Users\49447\.qclaw\workspace\stock-god-ai"
set "PYW=C:\Users\49447\AppData\Local\Programs\Python\Python312\pythonw.exe"
set "LOG=%ROOT%\logs\run_loop.log"

cd /d "%ROOT%" || exit 1
if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"

REM 日志轮转：超过 10MB 就挪走，避免无限增长
if exist "%LOG%" (
    for %%A in ("%LOG%") do if %%~zA GTR 10485760 (
        move /y "%LOG%" "%LOG%.1" >nul 2>&1
    )
)

echo. >> "%LOG%"
echo ===== [!DATE! !TIME!] 启动器拉起闭环 ===== >> "%LOG%"

set /a tries=0
:loop
set /a tries+=1
"%PYW%" scripts\run_loop.py --no-notify >> "%LOG%" 2>&1
set "rc=!ERRORLEVEL!"

echo ===== [!DATE! !TIME!] 闭环退出 code=!rc!（第 !tries! 次）===== >> "%LOG%"

if "!rc!"=="3" (
    echo ===== 已有闭环实例在运行，启动器退出（避免双实例）===== >> "%LOG%"
    exit /b 0
)
if !tries! GEQ 5 (
    echo ===== 已连续重启 5 次，放弃。请检查日志。===== >> "%LOG%"
    exit /b !rc!
)

timeout /t 30 /nobreak >nul
goto loop
