@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 【转发壳】N53 增量 1：旧入口已并入统一入口，行为保持不变。
rem   等价命令：egomed.bat sim   （起服务端 -> 等预热就绪 -> 起命令行客户端）
rem   带参数示例：启动模拟眼镜.bat --source url --input http://192.168.1.23:8080/video
rem   解释器解析已收敛到 egomed.bat + launcher.py，本文件不再自己找 python。

call "%~dp0egomed.bat" sim %*
set "RC=%errorlevel%"
echo.

if "%RC%"=="0" goto :egomed_ok

echo [警告] 客户端退出码 %RC%
echo   常见原因：端口 8000 被占用 / 模型权重缺失 / 依赖未装全 / 采集源未就绪
echo   排查建议：① 双击「安装环境.bat」补齐环境；② 双击「环境自检.bat」；③ 看 runs\launch\ 下的日志
pause >nul
exit /b %RC%

:egomed_ok
echo [完成] 按任意键关闭窗口 ...
pause >nul
exit /b 0
