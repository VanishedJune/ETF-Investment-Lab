@echo off
cd /d "%~dp0.."
start "" /b "%~dp0..\.venv\Scripts\python.exe" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8765
exit /b %ERRORLEVEL%
