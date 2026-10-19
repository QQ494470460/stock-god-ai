@echo off
REM ===========================================================================
REM StockGodAI - resident closed-loop trading launcher
REM
REM Invoked by the Startup-folder shortcut at user logon (no admin needed).
REM
REM KEEP THIS FILE STRICTLY ASCII + CRLF.
REM cmd.exe decodes batch files with the OEM code page (936/GBK on zh-CN) and
REM its parser is line-ending sensitive: UTF-8 Chinese comments and LF-only
REM line endings silently corrupt if-blocks and goto labels, so the launcher
REM dies with "'M' is not recognized as an internal or external command".
REM
REM Two log files on purpose:
REM   logs\launcher.log  - this launcher's own bookkeeping (rotated at 1MB)
REM   logs\run_loop.log  - the resident loop's stdout/stderr
REM They must NOT be the same file: the running loop holds run_loop.log open
REM for its whole lifetime, so a second launch's ">> samefile" redirection
REM fails BEFORE pythonw even runs, and a failed redirection does NOT set
REM ERRORLEVEL (verified) - hence the python probe below.
REM
REM Duplicate-instance handling:
REM   scripts\loop_lock_probe.py exits 1 when data\.run_loop.lock is held.
REM   That mirrors run_loop.py's own check exactly (a writable open is refused
REM   while another process holds the byte-range lock), and a *process exit
REM   code* is the only reliable signal available to cmd.
REM
REM Auto-restart: abnormal exit restarts at most 5 times. run_loop.py itself
REM exits 3 when it finds an existing instance, and we treat that as final.
REM ===========================================================================
setlocal enabledelayedexpansion

REM Derive the project root from this script's own location (works in the dev
REM tree and in a packaged copy alike). %~dp0 ends with a backslash.
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PYW=C:\Users\49447\AppData\Local\Programs\Python\Python312\pythonw.exe"
REM Force UTF-8 stdio. Without this, pythonw + ">> log" leaves stdout on the
REM OEM code page (936/GBK on zh-CN) and run_loop.py's emoji output dies with
REM UnicodeEncodeError -> exit code 1 -> 5 retries -> the trading loop silently
REM never runs. run_loop.py reconfigures stdio itself too; this is belt+braces.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set "LOG=%ROOT%\logs\run_loop.log"
set "LLOG=%ROOT%\logs\launcher.log"

cd /d "%ROOT%" || exit 1
if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"

REM rotate the launcher log above 1MB
if exist "%LLOG%" (
    for %%A in ("%LLOG%") do if %%~zA GTR 1048576 (
        move /y "%LLOG%" "%LLOG%.1" >nul 2>&1
    )
)

REM rotate the loop log above 10MB (only reachable when no instance is alive)
if exist "%LOG%" (
    for %%A in ("%LOG%") do if %%~zA GTR 10485760 (
        move /y "%LOG%" "%LOG%.1" >nul 2>&1
    )
)

echo ===== launcher start %DATE% %TIME% ===== >> "%LLOG%"

REM Fail-closed duplicate probe (exit code, not ERRORLEVEL).
"%PYW%" scripts\loop_lock_probe.py
if errorlevel 1 (
    echo ===== lock is held: a loop instance is alive, launcher exits ===== >> "%LLOG%"
    exit /b 0
)

set /a tries=0
:loop
set /a tries+=1
"%PYW%" scripts\run_loop.py --no-notify >> "%LOG%" 2>&1
set "rc=!ERRORLEVEL!"

echo ===== loop exited code=!rc! (try !tries!) ===== >> "%LLOG%"

if "!rc!"=="3" (
    echo ===== another instance is running, launcher exits ===== >> "%LLOG%"
    exit /b 0
)
if !tries! GEQ 5 (
    echo ===== 5 attempts failed, giving up. check the logs. ===== >> "%LLOG%"
    exit /b !rc!
)

REM ping instead of timeout: timeout aborts when stdin is redirected (headless).
ping -n 31 127.0.0.1 >nul
goto loop
