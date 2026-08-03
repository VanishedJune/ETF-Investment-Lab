@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\build-desktop.ps1"
exit /b %ERRORLEVEL%
