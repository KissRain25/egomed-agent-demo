@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem ============================================================
rem  EgoMed-Agent 复现补跑 · 一键续跑（重启电脑后运行这个）
rem    1) 下载看门狗：保证 HF 分片补齐（标注 + 测试病例完整帧）
rem    2) 评测排队器：数据齐了自动拉起单集评测（含失败重试）
rem  两者都有断点/单实例保护，重复运行是安全的。
rem ============================================================

set "PY=%EGOMED_PY%"
if not defined PY if exist "D:\ScienceApp\anaconda\envs\egomed\python.exe" set "PY=D:\ScienceApp\anaconda\envs\egomed\python.exe"
if not defined PY set "PY=python"

echo ============================================================
echo   EgoMed-Agent 复现补跑 · 一键续跑
echo   解释器: %PY%
echo ============================================================
echo.

if not exist "runs" mkdir "runs"

echo 提示：若下载日志出现 ProxyError，请先启动代理节点（127.0.0.1:1080）。
echo.

echo [1/2] 启动下载看门狗 ...
powershell -NoProfile -Command "Start-Process -FilePath '%PY%' -ArgumentList 'scripts\watch_downloads.py','--interval','300' -RedirectStandardOutput 'runs\_watchdog_stdout.txt' -RedirectStandardError 'runs\_watchdog_stderr.txt' -WindowStyle Hidden"

echo [2/2] 启动评测排队器 ...
powershell -NoProfile -Command "Start-Process -FilePath '%PY%' -ArgumentList 'scripts\run_eval_queue.py' -RedirectStandardOutput 'runs\_eval_queue_stdout.txt' -RedirectStandardError 'runs\_eval_queue_stderr.txt' -WindowStyle Hidden"

timeout /t 3 >nul
echo.
echo 已启动。查看进度：
echo    "%PY%" runs\tmp_night_check.py
echo.
echo 关键日志：
echo    runs\_download_watchdog.log     下载看门狗（待下载分片数）
echo    runs\_labels_download*.log      当前下载任务（速率/块重试）
echo    runs\_eval_queue.log            评测排队器（数据是否齐/是否在跑）
echo    runs\_eval_queue_DONE           出现此文件表示四集全部跑完
echo.
echo 四集结果（每集跑完写出）：
echo    runs\eval_yolo26_original_sam2_video_schedule_reset_retrack_correction\
echo        egomed5_yolo26m_original_sam2_online_schedule_reset_retrack_iou06\^<数据集^>\class_summary.csv
echo.
pause
