@echo off
chcp 65001 >nul
title EgoMed Demo 启动器
cd /d %~dp0

rem 检查端口 7860 是否已被占用
netstat -ano | findstr ":7860" | findstr "LISTENING" >nul 2>&1
if %errorlevel%==0 (
    echo [提示] 服务已经在运行了，直接帮你打开浏览器...
    start "" http://127.0.0.1:7860
    call :show_link
    goto :end
)

echo [1/2] 正在激活环境并后台启动 demo 服务，请稍候...
call conda activate egomed >nul 2>&1
start "" /min cmd /c "conda activate egomed && python start_demo.py"

echo [2/2] 等待模型加载（约 1~2 分钟），请耐心等待...
set /a cnt=0

:loop
timeout /t 5 >nul
netstat -ano | findstr ":7860" | findstr "LISTENING" >nul 2>&1
if %errorlevel%==0 (
    echo.
    echo 服务已就绪，自动打开浏览器...
    start "" http://127.0.0.1:7860
    call :show_link
    goto :end
)
set /a cnt+=1
if %cnt% GEQ 24 (
    echo.
    echo 等待超时。请手动打开浏览器访问: http://127.0.0.1:7860
    goto :end
)
echo 仍在加载中...（已等待 %cnt%0 秒）
goto :loop

goto :done

:show_link
for /f "tokens=6" %%i in ('findstr /c:"Running on public URL" "%cd%\runs\gradio_log.txt" 2^>nul') do set PUBURL=%%i
if defined PUBURL (
    echo.
    echo ============================================
    echo 公网链接（发给队友，异地也能打开）:
    echo %PUBURL%
    echo %PUBURL% | clip
    echo 已复制到剪贴板，直接 Ctrl+V 发给队友即可。
    echo ============================================
) else (
    echo.
    echo [注意] 还没找到公网链接，等 1 分钟后重开本文件，
    echo 或查看 runs\gradio_log.txt。
)
exit /b

:done
echo.
echo 提示: 想彻底关掉服务的话，打开任务管理器结束 python 进程即可。
echo.
pause
