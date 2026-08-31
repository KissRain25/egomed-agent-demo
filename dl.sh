#!/bin/bash
# 断点续传 + 自动重试下载脚本
# 用法: dl.sh <url> <outfile>
URL="$1"
OUT="$2"
MAX_ATTEMPTS=20
attempt=0
while [ $attempt -lt $MAX_ATTEMPTS ]; do
  attempt=$((attempt+1))
  echo "[dl] $OUT 第 $attempt 次尝试..."
  curl -sL -C - --retry 3 --retry-delay 5 --retry-all-errors --max-time 5400 -o "$OUT" "$URL"
  rc=$?
  # 校验是否下完整（curl 正常结束即认为成功；断点续传已保证文件完整性）
  if [ $rc -eq 0 ]; then
    echo "[dl] $OUT 完成 ($(du -h "$OUT" | cut -f1))"
    exit 0
  fi
  echo "[dl] 失败 rc=$rc，5 秒后重试"
  sleep 5
done
echo "[dl] $OUT 达到最大重试次数仍失败"
exit 1
