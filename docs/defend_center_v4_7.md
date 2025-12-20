# Defend Center v4_7 — 训练与验证指南 ✅

该文档记录了 v4_7（实验名 `defend_center_v4_8`）的训练命令、重要超参数、验证步骤、以及调参建议，帮助复现实验并快速诊断问题。

---

## 1) 训练命令（你提供的）
```bash
PYTHONPATH=. python src/train_custom.py \
  --algo=APPO \
  --env=custom_doom_defend_the_center \
  --experiment=defend_center_v4_8 \
  --num_workers=8 \
  --num_envs_per_worker=4 \
  --train_for_env_steps=20000000 \
  --use_rnn=True \
  --rnn_size=512 \
  --rnn_type=gru \
  --device=cpu \
  --obs_subtract_mean=0.0 \
  --obs_scale=255.0
```

> 说明：实验名中包含 `v4_8`（保持命名一致性）。如果你想称之为 v4_7，可把 `--experiment` 改为 `defend_center_v4_7`。

---

## 2) 关键超参数解释 🔍
- **--algo=APPO**：异步近端策略优化（样本高效且常用于多环境并行训练）。
- **--num_workers / --num_envs_per_worker**：并行度设置 —— 总环境数 = workers * envs_per_worker（用于充分利用 CPU）。
- **--train_for_env_steps=20000000**：总环境交互步数（长期训练）。
- **--use_rnn=True, --rnn_size=512, --rnn_type=gru**：启用 RNN（GRU）；适合部分时间相关策略（瞄准与追踪目标）。
- **--device=cpu**：使用 CPU；若可用 GPU，改为 `--device=cuda` 以加速训练。
- **观测预处理**：`--obs_subtract_mean` 与 `--obs_scale` 控制观测归一化（与 `ImageCleaningWrapper` 的输出一致）。

---

## 3) 训练前检查（必做）⚠️
- 确认 `src/_vizdoom/defend_the_center.wad` 存在且 `src/_vizdoom/defend_the_center.cfg` 已被 patched（含 `available_game_variables = { KILLCOUNT HITCOUNT AMMO2 HEALTH FRAGCOUNT }`）。
- 确保 `src/envs/wrappers.py` 中：
  - `ImageCleaningWrapper` 使用 128x96 或你期望的分辨率；
  - `CompositeActionWrapper` 已启用；
  - `RewardShapingWrapper` 包含空枪惩罚（可根据实验灵活调整）。
- 检查音频抑制是否生效（减少 PipeWire 报错）：环境变量 `ALSOFT_DRIVERS=null` / `SDL_AUDIODRIVER=dummy`，或在 env 初始化中禁音。

---

## 4) 快速验证（短跑，先验证 pipeline）⚡
在开始 20M 步之前，先做短期训练生成 checkpoint：
```bash
# 例：短跑 10k steps，1 worker 2 envs（可快速生成 checkpoint）
PYTHONPATH=. python src/train_custom.py \
  --algo=APPO --env=custom_doom_defend_the_center --experiment=defend_center_v4_8_test \
  --num_workers=1 --num_envs_per_worker=2 --train_for_env_steps=10000 --device=cpu --use_rnn=True --rnn_size=256
```
完成后检查：
```bash
ls -la train_dir/defend_center_v4_8_test/checkpoint_p0
```
然后运行评估并保存视频（见 docs/defend_center_v4_6.md 中的脚本）。

---

## 5) 评估命令（复用）🎥
- 使用 `scripts/run_eval_latest.sh`（已存在或可创建）自动找到最近 checkpoint 并生成时间戳目录。示例：
```bash
./scripts/run_eval_latest.sh
# 或使用一行命令见 docs/defend_center_v4_6.md
```

---

## 6) 监控与关键指标 📊
- **平均奖励（avg reward）**：长期趋势最重要。
- **命中率 = sum(HIT_INC) / sum(AMMO_USED)**：衡量瞄准质量。
- **击杀率（KILL_INC）** 与平均 episode length（更长生命周期通常更好）。
- **AMMO_USED**：如果非常低，说明策略过度保守；如果非常高且命中率低，说明滥射。

在 TensorBoard 或日志中同时查看 `HIT_INC`, `AMMO_USED`, `KILL_INC`。

---

## 7) 调参建议（经验）🛠️
- **策略过于保守（几乎不射）**：把空枪惩罚从 `-1.0` 降到 `-0.2 ~ -0.5`，或略微增大 `HIT` 奖励。 
- **滥射但命中率低**：提升空枪惩罚或增加命中奖励（使命中更有价值）。
- **RNN 配置**：如果观察到记忆没有帮助（训练不稳定），尝试减小 `rnn_size`（256），或禁用 RNN 做对照实验。
- **分辨率权衡**：若经常看不到远处敌人，考虑上调到 `res_w=160,res_h=120` 或增加网络能力（层数/通道）。

---

## 8) 常见问题诊断清单 🔍
- **找不到 WAD/cfg**：确保 `src/_vizdoom` 下有 `*.wad` 和 `*.cfg`，或按前文复制场景文件。 
- **HITCOUNT 不出现在 `env.info`**：检查 cfg 是否包含 `available_game_variables`，或查看 wrapper 的后备查询日志。 
- **PipeWire/ALSA 报错频繁**：确认音频禁用设置已生效。

---

## 9) 下一步建议（运行策略）
1. 先用短跑训练（10k）验证 pipeline，生成 checkpoint 并执行评估生成视频。
2. 若视觉/命中率不佳，微调 `RewardShapingWrapper` 的空枪惩罚与命中奖励后再做中期训练（100k）。
3. 若资源允许，把训练改为 `--device=cuda` 并增加 `num_workers/num_envs_per_worker` 以加速训练。

---

如需我：
- 帮你在容器里跑一次短跑并把日志 / 视频目录贴回，或
- 把常用脚本（`scripts/run_eval_latest.sh`, `scripts/check_scenarios.py`）加入仓库并提交 PR，任选其一我来执行。