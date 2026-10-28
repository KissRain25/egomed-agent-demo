@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem EgoMed-Agent 唯一入口（骨架 v0.1）
rem   用法：egomed.bat            -> 打开现场控制台（GUI）
rem         egomed.bat doctor     -> 环境自检
rem         egomed.bat diag       -> 生成诊断包
rem         egomed.bat --help     -> 查看全部子命令
rem   解释器解析已收敛到 launcher.py，本文件只做兜底，不再重复 6 行候选路径。

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
    echo [E3001] 没找到可用的 Python 解释器。
    echo   办法 1：先双击「安装环境.bat」创建 egomed 环境；
    echo   办法 2：设置环境变量 EGOMED_PY 指向你的 python.exe
    echo.
    pause
    exit /b 2
)

"%PY%" "%~dp0launcher.py" %*
set "RC=%errorlevel%"

rem 无参数（双击打开 GUI）时失败才暂停，便于看到错误；带子命令时保持脚本可链式调用
if "%~1"=="" if not "%RC%"=="0" (
    echo.
    echo [警告] 退出码 %RC%（0 成功 / 1 取消 / 2 环境问题 / 3 连接问题 / 4 验收失败 / 5 未实现）
    echo   排查：egomed.bat doctor
    pause >nul
)
exit /b %RC%
