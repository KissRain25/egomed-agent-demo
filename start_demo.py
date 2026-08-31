import subprocess
import sys

# 用 egomed 环境启动 Gradio demo，后台运行，日志写入文件
cmd = [
    r"D:\ScienceApp\anaconda\envs\egomed\python.exe",
    r"D:\EgoMed-Agent\scripts\demo\egomed_demo.py",
]
log = open(r"D:\EgoMed-Agent\runs\gradio_log.txt", "w", encoding="utf-8")
proc = subprocess.Popen(
    cmd, cwd=r"D:\EgoMed-Agent", stdout=log, stderr=subprocess.STDOUT, shell=False
)
print("Gradio 启动中... PID:", proc.pid)
