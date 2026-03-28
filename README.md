# EveryoneScienceTool

个人科研与实验用的小型工具集，本仓库以**单仓库多子项目**形式维护。

## 包含内容

| 目录 | 说明 |
|------|------|
| [`agent-idea-debate/`](agent-idea-debate/) | 基于种子论文（或网页/本地文档）的 **双模型论文 idea 辩论流水线**：多轮生成—审稿—回应，定稿后生成 **论文大纲**（含可选的大纲审稿与修订）。支持 checkpoint、断点续跑与 arXiv 相关工作检索。 |
| [`gpu-monitor/`](gpu-monitor/) | **GPU 实验监控**：调用 `nvidia-smi` 汇总各卡占用与进程命令行，输出易读的 `.txt` 报告；对空闲卡可对照上次快照做日志尾部启发式分析。仅依赖 Python 标准库与系统 `nvidia-smi`。 |

## 快速开始

- **Idea 辩论流水线**：进入 `agent-idea-debate/`，安装依赖并配置 API，详见该目录 [`README.md`](agent-idea-debate/README.md)。
- **GPU 监控**：进入 `gpu-monitor/`，直接运行脚本即可，详见该目录 [`README.md`](gpu-monitor/README.md)。

## 仓库说明

- `agent-idea-debate` 中本地产物目录 `outputs/` 与本地入口脚本 `run.sh` 默认不纳入版本控制（见根目录 `.gitignore`），请在本机自行保留或从模板创建。
