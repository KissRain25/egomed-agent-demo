@echo off
chcp 65001 >nul
title EgoMed 机位/佩戴校准
cd /d "%~dp0"

rem EgoMed-Agent 引导式机位校准（N36）—— 开工前先确认「对得准、看得清」
rem   抽查 8 帧发到服务端，统计「有结果 / 无结果」，并本地量亮度/对比/清晰度，
rem   最后给 PASS 或「需调整 + 具体原因」。报告落盘 runs\calibrate\<时间戳>\report.md
rem
rem 常用：
rem   机位校准.bat                                  （默认摄像头 --source cam）
rem   机位校准.bat --source url --input http://192.168.1.23:8080/video
rem   机位校准.bat --source video --input data/samples/acdc     （自动化自测）
rem
rem 前置：服务端要在跑（python server/api.py --real，或双击「启动模拟眼镜.bat」）
rem 解释器解析顺序：EGOMED_PY -> 常见 conda 环境 -> PATH 中的 python

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
    pause
    exit /b 1
)

"%PY%" "client\calibrate.py" %*
set "RC=%errorlevel%"
echo.
if "%RC%"=="0" (
    echo [完成] 机位校准通过，可以开工。
) else if "%RC%"=="2" (
    echo [注意] 环境/参数有问题（多半是服务端没起），按上面提示处理后重试。
) else (
    echo [警告] 机位需调整：按上面的 ^→ 建议改机位/补光后重跑本文件。
)
if "%~1"=="" pause >nul
exit /b %RC%
