@echo off
REM ============================================================
REM  XuanGuShenAI training-task registrar
REM  Run AS ADMIN (right-click -> Run as administrator)
REM  Registers a daily 15:10 training task (SYSTEM, survives reboot)
REM  The training script has sqlite-resume + panel-cache + done-marker.
REM ============================================================
setlocal
set TASKNAME=XuanGuShenAI-Train
set BAT=C:/Users/49447/.qclaw/workspace/stock-god-ai/scripts/train_resilient.bat

echo Registering daily 15:10 training task (SYSTEM, reboot-safe)...
schtasks /Create /TN "%TASKNAME%" /TR "%BAT%" /SC DAILY /ST 15:10 /RU SYSTEM /F
if errorlevel 1 (
  echo.
  echo [FAILED] Registration rejected. Run this file AS ADMIN with an admin account.
  goto :end
)
echo [OK] Task registered: daily 15:10 (sqlite-resume + panel-cache; reboot-safe).
echo Triggering one run now to start immediately...
schtasks /Run /TN "%TASKNAME%"
echo Triggered. Watch progress in Task Scheduler or logs\optuna_train_*.log.
:end
pause

