@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem EgoMed-Agent 一键回归（N22）
rem   自动起服务端 -> 跑 mock + real 两套验收 -> 出 PASS/FAIL 汇总表 -> 收尾关服务端
rem 直接双击 = 完整回归（real 套件要等预热 60~90 秒）
rem 也可带参数：
rem   回归.bat --suite mock                     只跑快的（协议/坏帧，秒级；改代码后常用）
rem   回归.bat --suite real --concurrency 5
rem   回归.bat --keep-server                    跑完不关服务端
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
    echo   办法 2：设置环境变量 EGOMED_PY 指向你的 python.exe，例如
    echo           set EGOMED_PY=C:\path\to\envs\egomed\python.exe
    echo.
    pause
    exit /b 1
)

echo [info] 使用解释器：%PY%
"%PY%" "client\run_regression.py" %*
set "RC=%errorlevel%"
echo.

if "%RC%"=="0" goto :reg_ok

echo [警告] 回归未全通过（退出码 %RC%）
echo   排查建议：
echo     1) 看上方哪一条用例 FAIL，报告在 runs\acceptance\ ^<时间戳^> \report.md
echo     2) 服务端日志在 runs\regression\ ^<时间戳^> \server_err.txt
echo     3) 端口 8000 被占用会导致起服务端失败：netstat -ano ^| findstr :8000
if "%~1"=="" pause >nul
exit /b %RC%

:reg_ok
echo [完成] 回归全部通过。
if "%~1"=="" pause >nul
exit /b 0
