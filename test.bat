@echo off
echo Running all backend tests via the project .venv...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\test.ps1"
exit /b %ERRORLEVEL%
