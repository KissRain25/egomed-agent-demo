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

echo [3/3] 下载模型权重（国内镜像，约 250MB）...
python scripts\download_models.py
echo.

echo ============================================
echo   安装完成！
echo   以后使用：双击「启动EgoMedDemo.bat」
echo ============================================
echo.
pause
