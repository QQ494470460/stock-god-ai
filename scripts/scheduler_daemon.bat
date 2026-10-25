@echo off
REM ===========================================================================
REM StockGodAI - resident after-hours job scheduler launcher
REM
REM Hosts the after-hours jobs (data update 18:30 / daily pick 18:45 /
REM evolution 19:00 / RankIC 19:15 / industry map hourly) in a process that
REM does NOT depend on the Streamlit dashboard being open.
REM
REM KEEP THIS FILE STRICTLY ASCII + CRLF.
REM cmd.exe decodes batch files with the OEM code page (936/GBK on zh-CN) and
REM its parser is line-ending sensitive: UTF-8 Chinese comments and LF-only
REM line endings silently corrupt if-blocks and goto labels, so the launcher
REM dies with "'M' is not recognized as an internal or external command".
REM
REM Duplicate-instance handling: scripts\loop_lock_probe.py --name
REM .scheduler_daemon.lock exits 1 while that lock is held. It shares the
REM exact mechanism with the daemon (src.core.singleton), so the two can
REM never disagree about whether an instance is alive. A process exit code is
REM the only reliable signal available to cmd: ERRORLEVEL does NOT reflect a
REM failed redirection.
REM
REM Auto-restart: NEVER permanently give up. Exit 0 = intentional stop,
REM exit 3 = another instance already running. Everything else backoffs and
REM retries forever.
REM ===========================================================================
setlocal enabledelayedexpansion

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PYW=C:\Users\49447\AppData\Local\Programs\Python\Python312\pythonw.exe"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set "LOG=%ROOT%\logs\scheduler_daemon.log"
set "LLOG=%ROOT%\logs\scheduler_launcher.log"
set "MARK=%ROOT%\data\.scheduler_daemon.stopped"

cd /d "%ROOT%" || exit 1
if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"

REM Clear the previous round's intentional-stop marker, otherwise one Ctrl+C
REM makes the next launcher start think "it wanted to stop" and give up.
if exist "%MARK%" del /q "%MARK%" >nul 2>&1

if exist "%LLOG%" (
    for %%A in ("%LLOG%") do if %%~zA GTR 1048576 (
        move /y "%LLOG%" "%LLOG%.1" >nul 2>&1
    )
)

if exist "%LOG%" (
    for %%A in ("%LOG%") do if %%~zA GTR 10485760 (
        move /y "%LOG%" "%LOG%.1" >nul 2>&1
    )
)

echo ===== scheduler launcher start %DATE% %TIME% ===== >> "%LLOG%"

"%PYW%" scripts\loop_lock_probe.py --name .scheduler_daemon.lock
if errorlevel 1 (
    echo ===== scheduler lock is held: an instance is alive, launcher exits ===== >> "%LLOG%"
    exit /b 0
)

set "pingcnt=31"
:loop
"%PYW%" scripts\scheduler_daemon.py >> "%LOG%" 2>&1
set "rc=!ERRORLEVEL!"

echo ===== scheduler exited code=!rc! ===== >> "%LLOG%"

if "!rc!"=="0" (
    REM Exit code 0 says nothing about intent: an externally terminated
    REM process ends with 0 too. Only scheduler_daemon.py itself writes the
    REM stop marker, and only after receiving Ctrl+C / SIGTERM.
    if exist "%MARK%" (
        echo ===== intentional stop marker found: daemon asked to stop, launcher exits ===== >> "%LLOG%"
        del /q "%MARK%" >nul 2>&1
        exit /b 0
    )
    echo ===== exit 0 WITHOUT stop marker: treated as abnormal, self-heal ===== >> "%LLOG%"
)
if "!rc!"=="3" (
    echo ===== another instance is running, launcher exits ===== >> "%LLOG%"
    exit /b 0
)

if "!pingcnt!"=="31" ( set "pingcnt=61" ) else if "!pingcnt!"=="61" ( set "pingcnt=121" ) else if "!pingcnt!"=="121" ( set "pingcnt=241" ) else ( set "pingcnt=301" )
echo ===== backoff ~!pingcnt!s then retry, self-heal ===== >> "%LLOG%"
ping -n !pingcnt! 127.0.0.1 >nul
goto loop
