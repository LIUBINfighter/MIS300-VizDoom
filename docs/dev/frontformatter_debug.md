---
title: "frontformatter debug"
date: 2025-12-19
tags: [debug, docker, vizdoom]
slug: frontformatter_debug
---

# frontformatter_debug 🐞

**概述**

简要记录本次在容器中调试 `main.py` 时的排查过程与结论。

## 环境与复现步骤 🔁

- 在项目根目录运行：
  - `docker-compose up`
  - 若想进入容器进行交互：`docker-compose exec vizdoom-lab bash`
  - 在容器内手动运行脚本：`python main.py`

## 排查要点与发现 ✅

- `main.py` 会快速执行并退出（会打印环境验收信息并退出）。
- `Dockerfile` 的 `ENTRYPOINT` 为 `xvfb-run ...`，`Xvfb` 作为前台进程保持运行，导致容器状态显示为 `Up` 即使主脚本已经退出。
- 因为脚本很快运行完并退出，如果用 `docker-compose up` 并在容器附着后才启动，会看不到已发生的输出。

## 解决建议 🔧

- 临时调试：使用 `docker-compose exec vizdoom-lab bash -lc "python main.py"` 或进入容器手动运行以查看输出。
- 保持容器可交互：在 `docker-compose.yml` 临时加 `command: tail -f /dev/null`，然后再进入容器手动运行脚本。
- 若希望 `docker-compose up` 时能看到脚本执行输出：把容器默认命令改为在前台运行脚本或添加一个调试开关（例如 `--watch`）让脚本在本地调试时持续运行。

## 结论与后续 💡

本次问题是典型的“主进程短命 + ENTRYPOINT 保持前台”的现象，已记录在此文档中。
