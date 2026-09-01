import subprocess
import sys
from pathlib import Path

# 项目根目录 = 本文件所在目录（自动定位，不依赖具体电脑）
ROOT = Path(__file__).resolve().parent

# 确保 runs 目录存在（日志要写进去）
(ROOT / "runs").mkdir(parents=True, exist_ok=True)

# 用当前 Python 解释器（请先激活 egomed 环境再运行本脚本）
cmd = [
    sys.executable,
    str(ROOT / "scripts" / "demo" / "egomed_demo.py"),
]
log = open(ROOT / "runs" / "gradio_log.txt", "w", encoding="utf-8")
proc = subprocess.Popen(
    cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, shell=False
)
print("Gradio 启动中... PID:", proc.pid)
print("日志文件:", ROOT / "runs" / "gradio_log.txt")
