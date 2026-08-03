@echo off
echo Usage: restore.bat YYYY-MM-DD_HH-mm-ss
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\restore.ps1" -BackupName "%~1"
exit /b %ERRORLEVEL%
