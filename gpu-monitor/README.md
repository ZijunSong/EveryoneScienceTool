# gpu-monitor

在 Linux + NVIDIA 环境下，用 **`nvidia-smi`** 与 **`/proc`** 采集 GPU 与进程信息，生成**人类可读的 `.txt` 报告**；完整状态另存为同目录 **`last_snapshot.json`**，供下次运行对照「上次占用」与空闲卡上的日志尾部（启发式关键词）。**仅使用 Python 标准库**，无额外 pip 依赖。

## 运行方式

在 `gpu-monitor` 目录下任选其一：

```bash
./run_gpu_monitor.sh
# 或
python3 gpu_monitor.py
```

默认将报告写入 **`reports/`**（与脚本同级的 `reports/`），文件名形如 `gpu_monitor_YYYY-MM-DD_HH-MM-SS.txt`，并更新 **`latest.txt`** 软链指向最近一次报告。

## 常用参数

完整说明：`python3 gpu_monitor.py --help`

| 参数 | 说明 |
|------|------|
| `-o DIR` / `--output-dir DIR` | 报告输出目录（默认：`<gpu-monitor>/reports`） |
| `--idle-mem-mib N` | 无 compute 进程时，显存低于 N MiB 视为「空卡」（默认 128） |
| `--tail-lines N` | 分析日志时读取文件末尾行数（默认 400） |
| `--expected-gpus N` | 可选；实际卡数与 N 不一致时在报告顶部警告 |
| `--keep-reports N` | 目录中最多保留 N 个 `gpu_monitor_*.txt`，更早的会删除（默认 5） |

## 输出说明

- **`gpu_monitor_*.txt`**：当前时刻各卡利用率、显存、按父进程聚合的占用摘要；对判定为空闲的卡，可与上次快照对照并尝试解析命令行中的日志路径，读取尾部做简单正负向关键词提示。
- **`last_snapshot.json`**：结构化快照，便于程序或下次运行对照；不必手工打开。
- 若环境无 GPU 或 `nvidia-smi` 失败，报告中会给出相应提示。

## 定时任务（示例）

需要周期性快照时，可用 `cron` 调用 `run_gpu_monitor.sh` 或 `gpu_monitor.py`，并结合 `--keep-reports` 控制磁盘占用。
