@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 【转发壳】N53 增量 1：等价命令 egomed.bat console
rem   EgoMed-Agent 现场控制台（N26）—— 选源 -> 设目标 -> 点「开始」，看实时指标与画面
rem   解释器解析已收敛到 egomed.bat + launcher.py，本文件不再自己找 python。

call "%~dp0egomed.bat" console %*
set "RC=%errorlevel%"

if not "%RC%"=="0" (
    echo.
    echo [警告] 控制台退出码 %RC%（常见：缺少 tkinter / 依赖未装全 / 端口 8000 被占用）
    echo   排查：先单独跑  python client\live_console.py --selftest
    if "%~1"=="" pause >nul
)
exit /b %RC%
