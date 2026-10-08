@echo off
chcp 65001 >nul
title EgoMed 环境自检
cd /d "%~dp0"

rem EgoMed-Agent 环境自检（N29）—— 换机器/交付时先跑它
rem   检查解释器/依赖/GPU/权重/样例/端口/磁盘等十余项，给出 PASS-WARN-FAIL 与修复建议
rem   解释器解析顺序：EGOMED_PY 环境变量 -> 常见 conda 环境 -> PATH 中的 python

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
"%PY%" "scripts\check_env.py" %*
set "RC=%errorlevel%"

if "%RC%"=="0" (
    echo.
    echo [完成] 自检通过（有 WARN 一般可忽略）
) else (
    echo.
    echo [警告] 存在 FAIL 项，按上面的 ^→ 提示处理后重跑本文件
)

if "%~1"=="" pause >nul
exit /b %RC%
