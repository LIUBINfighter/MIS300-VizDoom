# 修复：VizDoom WAD 路径处理与自动复制

## 问题回顾
在容器/镜像中运行短时训练时，Rollout 进程报错：

  FileDoesNotExistException('File "/app/src/_vizdoom/usr/local/lib/python3.10/site-packages/vizdoom/scenarios/basic.wad" does not exist.')

原因是我们在 CFG 中写入了一个系统绝对路径（例如 `/usr/local/.../basic.wad`），但是 ViZDoom 的解析器在处理 `doom_scenario_path` 时对路径解析存在歧义，**会把这种路径当成相对路径并与 cfg 的目录拼接**，从而导致实际文件查找路径变成 `/app/src/_vizdoom/usr/local/.../basic.wad`（不存在），引发 `EnvCriticalError`。

## 我们的修复
在 `src/envs/vizdoom_env.py::_ensure_game_variables_in_cfg` 中：

- 当系统路径（`vzd.scenarios_path/.../*.wad`）存在但本地 `src/_vizdoom` 没有对应 WAD 时，脚本会尝试把系统 WAD 拷贝到 `src/_vizdoom/`，并在生成的 cfg 中使用 **仅文件名**（例如 `defend_the_center.wad`）而非绝对路径。
- 如果拷贝失败（权限/IO 错误），代码会回退为使用系统路径，但会把路径用 **引号** 包裹写入 cfg，从而降低解析歧义并发出告警日志。

此修复保证：
- Rollout 进程能在容器中正确找到并加载 WAD；
- 礼貌地处理容器与系统包之间的路径差异（尤其在 Docker build / runtime 的工作目录差异下）。

## 如何验证（手动）
1. 重建镜像（确保代码包含修改）：

   DOCKER_BUILDKIT=1 docker build --no-cache --build-arg INSTALL_METHOD=uv -t mis300-vizdoom:latest .

2. 运行 smoke custom training（带较小步数与合适 shm）：

   docker run --rm --shm-size=2g --entrypoint /bin/bash mis300-vizdoom:latest -c "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONFAULTHANDLER=1; PYTHONPATH=. python -u src/train_custom.py --env custom_doom_basic --num_workers 1 --num_envs_per_worker 2 --train_for_env_steps 10 --device cpu --train_dir /tmp/smoke_train_shm --save_every_sec 300 --with_wandb False --batch_size 32 --num_batches_per_epoch 1 --num_epochs 1"

3. 期望输出：不再出现 `FileDoesNotExistException`，训练能够初始化 Rollout 并开始采样，或若仍有问题可在日志中看到将 WAD 拷贝到 `src/_vizdoom` 的信息：

   [Info] Copied WAD from /usr/local/lib/.../basic.wad to /app/src/_vizdoom/basic.wad

## 补充建议
- 我们在 CI 的 smoke test 中已经默认使用 `--shm-size=2g` 来避免并行共享内存问题导致的 crash。
- 若将来遇到其它 scenario 的缺失，可手工从宿主复制：

  docker run --rm --entrypoint /bin/bash mis300-vizdoom:latest -c "cp /usr/local/lib/python3.10/site-packages/vizdoom/scenarios/<name>.wad /app/src/_vizdoom/"


---

此文档与代码修复已提交到仓库（见 commit），如需我把该修复拆分成更小的单元 test（如添加 unit test 或集成 smoke-test），我可以继续完善并提交 PR。