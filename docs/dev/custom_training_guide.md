# 自定义训练与奖励建模指南

本文档记录了 MIS300-VizDoom 项目中自定义环境、奖励函数以及训练配置的演进过程。

## 1. 模块化结构 (src/envs)

为了支持自定义奖励和图像预处理，我们建立了以下结构：
- [src/envs/vizdoom_env.py](src/envs/vizdoom_env.py): 环境创建工厂，负责将 `DoomSpec` 与自定义 Wrapper 结合。
- [src/envs/wrappers.py](src/envs/wrappers.py): 存放自定义 Gymnasium Wrapper（如奖励塑形、图像清理）。
- [src/envs/__init__.py](src/envs/__init__.py): 自动注册 `custom_doom_*` 系列环境到 Sample Factory。

## 2. 奖励塑形策略 (Reward Shaping)

### 2.1 保卫中心 (Defend The Center)
- **目标**: 击杀尽可能多的怪物，同时保持生命值。
- **自定义奖励**:
  - `KILLCOUNT`: 每增加 1 个击杀，奖励 **+15.0**。
  - `HEALTH`: 每损失 1 点生命值，惩罚 **-0.2**。
  - `AMMO2`: 每次开火（弹药减少），惩罚 **-0.1**（鼓励精准度）。

### 2.2 死亡走廊 (Deadly Corridor)
- **目标**: 走到走廊尽头，同时清理沿途敌人。
- **自定义奖励**:
  - `POSITION_X`: 沿 X 轴前进，奖励 **diff_x * 0.01**。
  - `KILLCOUNT`: 每击杀一个敌人，奖励 **+20.0**。
  - `HEALTH`: 掉血惩罚 **-0.5**。
  - `AMMO2`: 开火惩罚 **-0.2**。

## 3. 核心训练命令

### 3.1 自定义训练入口
必须设置 `PYTHONPATH=.` 以确保 `src` 模块能被正确导入。

```bash
# 示例：训练自定义保卫中心
PYTHONPATH=. python src/train_custom.py \
    --algo=APPO \
    --env=custom_doom_defend_the_center \
    --experiment=defend_center_custom_v1 \
    --device=cpu \
    --num_workers=8 \
    --num_envs_per_worker=4 \
    --train_for_env_steps=5000000 \
    --reward_scale=0.1
```

## 4. 评估与视频生成 (黑魔法)

由于 PyTorch 2.6+ 的安全性限制和 Sample Factory 内部 Key 命名的变动，必须使用 [src/run_enjoy_safe.py](src/run_enjoy_safe.py) 进行评估。

### 4.1 运行评估
```bash
PYTHONPATH=. python src/run_enjoy_safe.py \
    --env=custom_doom_defend_the_center \
    --experiment=defend_center_custom_v1 \
    --save_video \
    --video_frames=2000 \
    --max_num_episodes=5
```

### 4.2 关键修复说明
- **安全白名单**: 注入 `np.dtype` 等类型到 `torch.serialization.add_safe_globals`。
- **Key 映射修复**: 自动将 `encoder.encoders.obs.enc` 映射为 `encoder.basic_encoder.enc`，解决模型加载时的 `RuntimeError`。

## 5. 实验日志

| 实验 ID | 场景 | 核心配置 | 状态 | 结论 |
| :--- | :--- | :--- | :--- | :--- |
| `defend_center_v1` | 默认 | 官方脚本, CPU | 已完成 | 基础表现正常，但杀敌不够积极。 |
| `defend_center_custom_v1` | 自定义 | 击杀+15, 弹药-0.1 | 训练中 | 杀敌积极性显著提升，命中率提高。 |
| `deadly_corridor_custom_v1` | 自定义 | 前进奖励 + 击杀奖励 | 待启动 | 预期解决稀疏奖励导致的“原地转圈”问题。 |

## 6. 注意事项
- **GPU vs CPU**: 在 Docker 中若未配置 NVIDIA Toolkit，请务必使用 `--device=cpu`。
- **TensorBoard**: 映射端口 6007:6006 以避免 Windows 端口占用冲突。
