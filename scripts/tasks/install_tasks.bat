@echo off
REM ===========================================================================
REM StockGodAI - register the two resident processes as Windows scheduled tasks.
REM
REM Double-click this file. It needs no administrator rights (the tasks run
REM under your own account, "only when the user is logged on", so no password
REM is stored).
REM
REM Why scheduled tasks instead of only the Startup folder:
REM   A Startup entry fires once at logon. If a resident process dies at 2pm,
REM   nothing brings it back until the next reboot. These tasks add a 10-minute
REM   watchdog: if the process is gone, Task Scheduler starts it again. The
REM   batch launcher itself guards with a cross-process single-instance lock,
REM   so a watchdog tick while the process is alive exits immediately.
REM
REM Tasks created:
REM   StockGodAI-RunLoop     trading loop        (scripts\run_loop_daemon.bat)
REM   StockGodAI-Scheduler   after-hours jobs    (scripts\scheduler_daemon.bat)
REM ===========================================================================
setlocal

set "DIR=%~dp0"

schtasks /create /xml "%DIR%StockGodAI-RunLoop.xml" /tn "StockGodAI-RunLoop" /f
if errorlevel 1 (
    echo [FAIL] could not register StockGodAI-RunLoop
) else (
    echo [ OK ] StockGodAI-RunLoop registered
)

schtasks /create /xml "%DIR%StockGodAI-Scheduler.xml" /tn "StockGodAI-Scheduler" /f
if errorlevel 1 (
    echo [FAIL] could not register StockGodAI-Scheduler
) else (
    echo [ OK ] StockGodAI-Scheduler registered
)

echo.
echo Verify with:  schtasks /query /tn StockGodAI-RunLoop
echo Remove with:  scripts\tasks\uninstall_tasks.bat
echo.
pause
