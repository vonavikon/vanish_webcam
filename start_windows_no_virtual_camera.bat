@echo off
call "%~dp0start_windows.bat" --no-virtual-camera %*
exit /b %errorlevel%
