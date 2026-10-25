@echo off
REM ===========================================================================
REM StockGodAI - remove the two scheduled tasks.
REM The processes themselves are NOT killed by this: the Startup-folder .vbs
REM entries keep working on their own, just without the 10-minute watchdog.
REM ===========================================================================
setlocal

schtasks /delete /tn "StockGodAI-RunLoop" /f
if errorlevel 1 (
    echo [FAIL] could not delete StockGodAI-RunLoop (maybe it was never installed)
) else (
    echo [ OK ] StockGodAI-RunLoop removed
)

schtasks /delete /tn "StockGodAI-Scheduler" /f
if errorlevel 1 (
    echo [FAIL] could not delete StockGodAI-Scheduler (maybe it was never installed)
) else (
    echo [ OK ] StockGodAI-Scheduler removed
)

echo.
echo The Startup-folder entries are untouched:
echo   %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\StockGodAI-RunLoop.vbs
echo   %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\StockGodAI-Scheduler.vbs
echo.
pause
