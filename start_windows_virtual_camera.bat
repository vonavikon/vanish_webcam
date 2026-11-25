@echo off
call "%~dp0start_windows.bat" --virtual-camera %*
exit /b %errorlevel%
