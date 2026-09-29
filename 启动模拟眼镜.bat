@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 一键启动「模拟眼镜」演示：起服务端(真引擎) -> 等预热 -> 起客户端
rem 交互式直接双击；也可带参数：启动模拟眼镜.bat --source url --input http://192.168.1.23:8080/video
rem 解释器解析顺序：EGOMED_PY 环境变量 -> 常见 conda 环境 -> PATH 中的 python
rem （不再写死某一台机器的绝对路径；换机器无需改脚本）

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
"%PY%" "client\launch_sim_glasses.py" %*
set "RC=%errorlevel%"
echo.

if "%RC%"=="0" goto :egomed_ok

echo [警告] 客户端退出码 %RC%
echo   常见原因：端口 8000 被占用 / 模型权重缺失 / 依赖未装全 / 采集源未就绪
echo   排查建议：① 双击「安装环境.bat」补齐环境；② 查看上方日志与 runs\ 下的日志文件
pause >nul
exit /b %RC%

:egomed_ok
echo [完成] 按任意键关闭窗口 ...
pause >nul
exit /b 0
