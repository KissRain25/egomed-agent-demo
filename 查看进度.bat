@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PY=%EGOMED_PY%"
if not defined PY if exist "D:\ScienceApp\anaconda\envs\egomed\python.exe" set "PY=D:\ScienceApp\anaconda\envs\egomed\python.exe"
if not defined PY set "PY=python"

"%PY%" scripts\show_progress.py
echo.
pause
