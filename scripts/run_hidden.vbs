' ===========================================================================
' StockGodAI - launch a batch file with NO visible console window.
'
' Usage:
'   wscript //B //NoLogo run_hidden.vbs "<full path to .bat>" ["<working dir>"]
'
' Why this exists:
'   The resident processes must not be tied to a visible console. A console
'   created by a plain "cmd /c" can flash on screen, and - worse - a console
'   that lives inside an interactive logon session gets CTRL_CLOSE_EVENT when
'   that session is torn down, which kills every child including pythonw.
'   Task Scheduler runs the task in its own session, so this wrapper only
'   adds the "no window" guarantee on top.
'
' Keep this file ASCII. WScript reads .vbs with the system ANSI code page.
' ===========================================================================
Option Explicit

If WScript.Arguments.Count < 1 Then
    WScript.Quit(2)
End If

Dim ws, bat
Set ws = CreateObject("WScript.Shell")
bat = WScript.Arguments(0)

If WScript.Arguments.Count >= 2 Then
    ws.CurrentDirectory = WScript.Arguments(1)
End If

' window style 0 = hidden, bWaitOnReturn = False (do not block Task Scheduler)
ws.Run """" & bat & """", 0, False
