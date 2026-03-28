# agent-idea-debate

基于 **OpenAI 兼容 API** 的论文 idea **辩论流水线**：用「生成器 / 审稿人 / 大纲」等角色对种子材料做多轮讨论，最终产出定稿 idea 与 **Markdown 论文大纲**；可选大纲审稿与修订。支持按论文标题建会话目录、checkpoint、断点续跑与 `--fresh` 重新跑。

## 环境

- Python 3.10+（建议）
- 依赖安装：

```bash
cd agent-idea-debate
python3 -m pip install -r requirements.txt
```

## 配置

- 默认读取同目录下的 `config.default.yaml`；若存在 **`config.yaml`**，会以它为配置（可先复制 `config.default.yaml` 再改）。
- 也可用 `--config /path/to/config.yaml` 指定。

## 认证与网关

通过环境变量或命令行传入（**勿将真实密钥提交到 git**）：

- `OPENAI_API_KEY`：必填
- `OPENAI_BASE_URL`：默认 `https://api.openai.com/v1`（兼容网关需带 `/v1`）
- `OPENAI_MODEL`：未单独指定三角色模型时的默认模型

命令行覆盖示例：`--api-key`、`--base-url`、`--model`，以及 `--generator-model` / `--reviewer-model` / `--outline-model`。

## 基本用法

对 **arXiv / 网页 URL** 或 **本地 `.pdf` / `.html` / `.txt` 路径** 运行主程序：

```bash
export OPENAI_API_KEY="你的密钥"
python3 run_pipeline.py "https://arxiv.org/abs/xxxx.xxxxx"
```

常用参数（完整列表见 `python3 run_pipeline.py --help`）：

| 参数 | 含义 |
|------|------|
| `--out-dir ./outputs` | 输出根目录（下按标题建子目录） |
| `--dry-run` | 只拉取与落盘，不调 API |
| `--fresh` | 同一路径重跑：旧结果归档到 `_archive/<时间>/` |
| `--no-resume` | 不续跑；若目录已有 checkpoint 需配合 `--fresh` |
| `--skip-outline` | 只跑辩论轮次，不生成大纲 |
| `--no-related-search` | 关闭 arXiv 相关工作检索 |
| `--max-rounds N` | 覆盖配置中的辩论轮数 |

输出目录中常见文件包括：`checkpoint.json`、`progress.md`、定稿后的 `final.md` / `final.json`，以及 **`outline_final.md`**（仅定稿大纲，便于归档或排序）。

## 辅助脚本

- **多篇大纲排序**（手动指定若干会话目录）：

  ```bash
  python3 rank_outlines.py outputs/dir_a/ outputs/dir_b/ -o outline_ranking.md
  ```

- **截稿/两篇工作战略问答**（输出到指定 md）：

  ```bash
  python3 outline_strategic_qa.py -o outputs/two_papers_plan.md
  ```

## 本地入口脚本（可选）

可在本机编写 `run.sh`（例如批量 URL、默认模型），**勿将含 API 密钥的脚本提交到 git**。在 [EveryoneScienceTool](https://github.com/ZijunSong/EveryoneScienceTool) 汇总仓库中，`run.sh` 默认被根目录 `.gitignore` 忽略。**不依赖** `run.sh` 也可完全通过 `python3 run_pipeline.py` 使用上述功能。
