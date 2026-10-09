@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem EgoMed 每日自动回归（N44）
rem   跑 client\run_regression.py -> 归档 runs\daily\history.csv + latest.md -> 与上次对比
rem   日常高频用：  每日回归.bat --quick       （只跑 mock 套件，秒级）
rem   完整（含真引擎）：每日回归.bat            （约 5~8 分钟，需端口 8000 空闲）
rem   注册成 Windows 计划任务：每日回归.bat --task-hint   （会打印 schtasks 命令）
rem
rem 注意：本文件【不暂停】（这样计划任务里不会卡住）；结果看 runs\daily\latest.md
rem 解释器解析顺序：EGOMED_PY 环境变量 -> 常见 conda 环境 -> PATH 中的 python

set "PY=%EGOMED_PY%"
if not defined PY if exist "D:\ScienceApp\anaconda\envs\egomed\python.exe" set "PY=D:\ScienceApp\anaconda\envs\egomed\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\envs\egomed\python.exe" set "PY=%USERPROFILE%\anaconda3\envs\egomed\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\envs\egomed\python.exe" set "PY=%USERPROFILE%\miniconda3\envs\egomed\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\anaconda3\envs\egomed\python.exe" set "PY=%LOCALAPPDATA%\anaconda3\envs\egomed\python.exe"
if not defined PY if exist "C:\ProgramData\anaconda3\envs\egomed\python.exe" set "PY=C:\ProgramData\anaconda3\envs\egomed\python.exe"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)

if not defined PY (
    echo [错误] 没找到可用的 Python 解释器（设 EGOMED_PY 或先跑「安装环境.bat」）
    exit /b 1
)

echo [info] 使用解释器：%PY%
"%PY%" "scripts\daily_regression.py" %*
exit /b %errorlevel%
