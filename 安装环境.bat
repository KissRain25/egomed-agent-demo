@echo off
chcp 65001 >nul
title EgoMed 环境一键安装
cd /d %~dp0

echo ============================================
echo   EgoMed Demo 环境一键安装
echo   预计耗时 10~30 分钟，期间电脑别睡眠
echo ============================================
echo.

where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 没找到 conda！
    echo 请先安装 Anaconda: https://www.anaconda.com/download
    echo 装完后重新双击本文件。
    pause
    exit /b 1
)

echo [1/3] 创建 conda 环境 egomed (Python 3.12)...
conda create -n egomed python=3.12 -y
echo.

echo [2/3] 安装依赖包（最耗时，请耐心等待）...
call conda activate egomed
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.

echo [3/4] 下载模型权重（国内镜像，约 250MB）...
python scripts\download_models.py
echo.

echo [4/4] 环境自检（N29）...
python scripts\check_env.py
echo.
if errorlevel 1 (
    echo [警告] 自检存在 FAIL 项，按上面 ^→ 的建议处理后，可再双击「环境自检.bat」复检
)

echo ============================================
echo   安装完成！
echo   下一步（详见 docs\上手指南.md）：
echo     1) 双击「环境自检.bat」确认无 FAIL
echo     2) 双击「现场控制台.bat」跑一次（选源 -^> 点目标 -^> 开始）
echo     3) 双击「回归.bat」做验收（mock 8/8 + real 10~11/11）
echo ============================================
echo.
pause
