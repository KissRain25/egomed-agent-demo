@echo off
chcp 65001 >nul
title EgoMed 环境自检
cd /d "%~dp0"

rem 【转发壳】N53 增量 1：等价命令 egomed.bat check（scripts/check_env.py 十余项详细检查，N29）
rem   只想看精简版：egomed.bat doctor
rem   解释器解析已收敛到 egomed.bat + launcher.py，本文件不再自己找 python。

call "%~dp0egomed.bat" check %*
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
