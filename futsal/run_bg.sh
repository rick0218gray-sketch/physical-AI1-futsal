#!/bin/bash
# usage: run_bg.sh <logfile> <cmd...>
# Runs <cmd> detached (setsid+nohup) in f_futsal with the Genesis venv active.
LOG="$1"; shift
JOB=$(mktemp /tmp/f_futsal_job_XXXX.sh)
{
  echo 'cd "$(dirname "$0")"'
  echo 'source ../.venv/bin/activate'
  echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH'
  echo 'start=$(date +%s)'
  printf '%q ' "$@"; echo
  echo 'echo "EXIT $? ELAPSED $(( $(date +%s) - start ))s"'
} > "$JOB"
setsid nohup bash "$JOB" > "$LOG" 2>&1 < /dev/null &
echo "started pid $! log $LOG"
sleep 2
