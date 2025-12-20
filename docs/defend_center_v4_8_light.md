# Defend Center v4_8_light — 训练记录

记录：轻量版 v4_8 训练命令与要点（用于实验追踪）。

---

## 训练命令（exact）
```bash
PYTHONPATH=. python src/train_custom.py \
  --algo=APPO \
  --env=custom_doom_defend_the_center \
  --experiment=defend_center_v4_8_light \
  --num_workers=4 \
  --num_envs_per_worker=4 \
  --train_for_env_steps=1000000 \
  --use_rnn=True \
  --rnn_size=256 \
  --rnn_type=gru \
  --device=cpu \
  --obs_subtract_mean=0.0 \
  --obs_scale=255.0 \
  --env_frameskip=2
```

> 备注：此为“轻量”配置（1M 步、较小 RNN、4 workers），便于快速迭代和超参调试。

---

## 关键超参数说明
- `--num_workers=4` / `--num_envs_per_worker=4`: 总环境数 = 16，可在 CPU 环境下平衡效率与稳定性。 
- `--train_for_env_steps=1000000`: 1M 步为短/中期试验，适合快速验参。 
- `--use_rnn=True, --rnn_size=256, --rnn_type=gru`: 适度的记忆能力，用于跟踪目标和短期动作序列。
- `--encoder_custom=convnet_simple`: 轻量编码器，减小计算与加速迭代。
- `--env_frameskip=2`: 每动作重复 2 帧（更平滑的控制）；如果需要更精细控制可改为 1（见 v4_7）。

---

## 验证与建议
1. **先短跑**：可先把 `--train_for_env_steps` 设为 10000 进行 smoke test，确认 pipeline（checkpoint、logs、评估）工作正常。 
2. **评估**：使用 `scripts/run_eval_latest.sh` 或之前记录的单行评估命令生成带时间戳视频并查看 `HIT_INC`、`AMMO_USED`、`KILL_INC`。
3. **调参建议**：若策略过于保守，减小空枪惩罚或提高 `HIT` 奖励；若滥射，增大空枪惩罚或增加负激励。 

---

## 日志记录建议
- 将训练命名为 `defend_center_v4_8_light` （已在 `--experiment` 中指定）。
- 在开始训练前确认 `src/_vizdoom/defend_the_center.cfg` 包含 `available_game_variables = { KILLCOUNT HITCOUNT AMMO2 HEALTH FRAGCOUNT }`。
- 建议将关键参数快照（命令行、随机种子、git commit id）写入 `train_dir/defend_center_v4_8_light/config.txt` 以便复现。

---

文件路径：`docs/defend_center_v4_8_light.md`
