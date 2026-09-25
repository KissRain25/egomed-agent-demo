@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 一键启动「模拟眼镜」演示：起服务端(真引擎) -> 等预热 -> 起客户端
rem 交互式直接双击；也可带参数：启动模拟眼镜.bat --source url --input http://192.168.1.23:8080/video

set "PY=D:\ScienceApp\anaconda\envs\egomed\python.exe"
if not exist "%PY%" (
  echo [warn] 未找到 egomed 解释器，尝试使用 PATH 中的 python
  set "PY=python"
)

"%PY%" "client\launch_sim_glasses.py" %*

echo.
echo [完成] 按任意键关闭窗口 ...
pause >nul
