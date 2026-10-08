@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem EgoMed-Agent 现场控制台（N26）—— 双击打开图形界面
rem   选源 -> 设目标 -> 点「开始」，全程看实时指标与画面；「录制」可留演示素材
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
    echo [错误] 没找到可用的 Python 解释器。
    echo   办法 1：先双击「安装环境.bat」创建 egomed 环境；
    echo   办法 2：设置环境变量 EGOMED_PY 指向你的 python.exe
    echo.
    pause
    exit /b 1
)

echo [info] 使用解释器：%PY%
"%PY%" "client\live_console.py" %*
set "RC=%errorlevel%"

if not "%RC%"=="0" (
    echo.
    echo [警告] 控制台退出码 %RC%（常见：缺少 tkinter / 依赖未装全 / 端口 8000 被占用）
    echo   排查：先单独跑  python client\live_console.py --selftest
    if "%~1"=="" pause >nul
)
exit /b %RC%
