#!/usr/bin/env bash
# 在任意目录执行均可：调用同目录下的 gpu_monitor.py
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DIR/gpu_monitor.py" "$@"
