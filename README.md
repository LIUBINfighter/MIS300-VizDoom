# MIS300-VizDoom

## 😅看起来出了点问题

```bash
docker run --rm --shm-size=2g --entrypoint /bin/bash mis300-vizdoom:latest -c "./scripts/smoke_custom_train.sh custom_doom_basic /tmp/smoke_train 100 1 2 cpu"

```

## Plan

- Env & `docker-compose build`
  - dockerfile, docker-compose.yml
  - pyproject.toml
  - main.py
  - 时间瓶颈1 `RUN apt-get update && apt-get install -y`  大约需要 `120~150s`
  - 时间瓶颈2 安装依赖（pip or poetry），取决于构建参数，poetry 大约 `700~800s`，pip 方式通常更快
  - 总用时 `~1000s`（视安装方式而定）

## Docker 构建与验证说明 🔧

- 默认优先使用 `uv`（pip + venv），在构建镜像时可通过 `INSTALL_METHOD` 选择安装方式：

  - 使用 pip (默认/推荐)：

    docker build --build-arg INSTALL_METHOD=uv -t mis300-vizdoom:latest .

  - 使用 Poetry：

    docker build --build-arg INSTALL_METHOD=poetry -t mis300-vizdoom:poetry .

- 本地验证镜像（运行内置 demo、quick test 或短时训练）：

  - 运行默认主进程（容器内会以 `xvfb-run` 启动）：

    docker run --rm -it mis300-vizdoom:latest

  - 或执行快速测试脚本：

    docker run --rm -it mis300-vizdoom:latest python quick_test.py

  - 运行 custom training 的短时 smoke test（在容器内部运行脚本）：

    docker run --rm --shm-size=2g --entrypoint /bin/bash mis300-vizdoom:latest -c "./scripts/smoke_custom_train.sh custom_doom_basic /tmp/smoke_train 1000 1 1 cpu"

  - 注：若在 CI 或 runner 上出现共享内存相关崩溃（Bus error），请增加 `--shm-size` 或使用 `--ipc=host`。
  - Windows PowerShell（本地）运行短时训练：

    .\scripts\smoke_custom_train.ps1 -TrainDir C:\\tmp\\smoke_train -Steps 1000 -NumWorkers 1 -NumEnvs 1 -Device cpu

- 本地开发（Windows）：使用 `scripts\install.ps1`：

  - pip + venv（默认/首选）：

    .\scripts\install.ps1 -method pip -venv venv

  - Poetry：

    .\scripts\install.ps1 -method poetry

- *说明*：仓库同时保留 `pyproject.toml`（Poetry）和 `requirements.txt`。使用 `poetry export -f requirements.txt --without-hashes -o requirements.txt` 可以基于 lock 文件重新生成 `requirements.txt`。

## 

```docker
python -m sf_examples.vizdoom.train_vizdoom   --algo=APPO   --env=doom_defend_the_center   --experiment=defend_center_v1   --train_dir=./train_dir   --device=cpu   --num_workers=1   --num_envs_per_worker=2   --train_for_env_steps=500000   --save_every_sec=300   --with_wandb=False
```

```docker
python src/run_enjoy_safe.py     --env=doom_defend_the_center     --experiment=defend_center_v1     --save_video     --video_frames=1500     --max_num_episodes=5

```

## Documentation 技术上下文

## License

[Mozilla Public License Version 2.0](./LICENSE)
