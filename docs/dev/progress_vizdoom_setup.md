# MIS300 ViZDoom 训练进展与可复用命令

本页整理当前可直接复制使用的命令，以及环境/训练的关键结论，方便后续持续训练与评估。

## 概览
- 环境测试：`vizdoom` 已在容器中无头渲染通过，场景能加载与运行。
- 训练：使用 `sf_examples.vizdoom.train_vizdoom` 在 `doom_defend_the_center` 场景下启动训练，CPU 模式，已生成首个 checkpoint。
- 评估：使用 `enjoy_vizdoom` 可运行评估并保存视频（注意 `--save_video` 为布尔开关）。
- 常见提示：PipeWire 音频相关警告可忽略；需使用正确的环境名与显示配置。

## 快速命令

### 1) 容器内显示配置
- 复用已有显示：
```bash
export DISPLAY=:99
```
- 或自动分配显示（推荐评估/录制视频时）：
```bash
xvfb-run -a <your command>
```

### 2) 场景环境测试
```bash
# 复用显示
export DISPLAY=:99
python src/test_vizdoom_env.py

# 或自动分配显示
xvfb-run -a python src/test_vizdoom_env.py
```

### 3) 短程训练（10k 步，双缓冲）
```bash
export DISPLAY=:99
python -m sf_examples.vizdoom.train_vizdoom \
  --algo=APPO \
  --env=doom_defend_the_center \
  --experiment=defend_center_v1 \
  --train_dir=./train_dir \
  --device=cpu \
  --num_workers=1 \
  --num_envs_per_worker=2 \
  --train_for_env_steps=10000 \
  --save_every_sec=120 \
  --with_wandb=False
```

- 若仅用 1 个环境，需关闭双缓冲拆分：
```bash
export DISPLAY=:99
python -m sf_examples.vizdoom.train_vizdoom \
  --algo=APPO \
  --env=doom_defend_the_center \
  --experiment=defend_center_v1 \
  --train_dir=./train_dir \
  --device=cpu \
  --num_workers=1 \
  --num_envs_per_worker=1 \
  --worker_num_splits=1 \
  --train_for_env_steps=10000 \
  --save_every_sec=120 \
  --with_wandb=False
```

### 4) 更长训练（50万步示例）
```bash
export DISPLAY=:99
python -m sf_examples.vizdoom.train_vizdoom \
  --algo=APPO \
  --env=doom_defend_the_center \
  --experiment=defend_center_v1 \
  --train_dir=./train_dir \
  --device=cpu \
  --num_workers=1 \
  --num_envs_per_worker=2 \
  --train_for_env_steps=500000 \
  --save_every_sec=300 \
  --with_wandb=False
```

### 5) 评估并保存视频
```bash
# 注意：--save_video 是布尔开关，不能写成 --save_video=True
xvfb-run -a python -m sf_examples.vizdoom.enjoy_vizdoom \
  --algo=APPO \
  --env=doom_defend_the_center \
  --experiment=defend_center_v1 \
  --train_dir=./train_dir \
  --max_num_episodes=5 \
  --save_video \
  --video_name=defend_center_v1_eval \
  --device=cpu
```

## 关键结论与踩坑记录
- 环境名需使用已注册的名称：示例包括 `doom_defend_the_center`、`doom_basic`、`doom_deadly_corridor`、`doom_my_way_home` 等。
- `--save_video` 为布尔开关，使用时不要写值；视频通常保存在 `train_dir/<experiment>/` 的子目录，查找 `*.mp4`。
- PipeWire 配置警告（client.conf）可忽略，不影响渲染与训练。
- 若使用自定义训练脚本，请确保环境注册函数为模块级（顶层）而非闭包/匿名函数，以避免 Python 多进程的 pickling 错误（例如：`Can't pickle local object '<locals>.make_env'`）。
- 训练短程运行日志示例：
  - 平均奖励打印：`Avg episode reward: 0.679`
  - 检查点生成：`train_dir/defend_center_v1/checkpoint_p0/checkpoint_000000004_16384.pth`

## 后续建议
- 扩展训练步数（如 500k–5M），并按需调整并行度以适配资源。
- 训练后运行评估并保存视频，观察击杀与命中趋势；如需自动统计指标，可添加轻量脚本基于 `enjoy_vizdoom` 输出汇总。
- 可使用 TensorBoard 监控：
```bash
tensorboard --logdir=./train_dir --host=0.0.0.0 --port=6006
```