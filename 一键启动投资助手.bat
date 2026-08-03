@echo off
setlocal
set "desktop_exe=%~dp0InvestmentLab.exe"
set "release_marker=%~dp0V3.4-13W.release"
if exist "%desktop_exe%" (
  if exist "%release_marker%" (
    start "" "%desktop_exe%"
    exit /b 0
  )
)
call "%~dp0start.bat"
exit /b %ERRORLEVEL%
