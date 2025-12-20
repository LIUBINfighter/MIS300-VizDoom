# Defend Center v4_6 — 运行与评估指南 ✅

简要说明、验证命令与主要修改点，便于快速上手和复现。

---

## 概览
- 目标：让 `Defend the Center` 场景的 Agent 能精确瞄准并稳定训练。 
- 本分支/实验：**v4_6**（改进视觉预处理、支持组合动作、奖励重塑、cfg/WAD 自动修补、静音以提升 FPS）。

---

## 当前状态（已完成）
- 视觉处理：`src/envs/wrappers.py` 中 `ImageCleaningWrapper` 改为 128x96（4:3），使用 `cv2.INTER_AREA` 缩放。
- 动作：新增 `CompositeActionWrapper`，支持组合动作（如 `左+开火`）；并对底层 `VizdoomEnv` 做了 monkey-patch，以接受 list/tuple 动作。
- 奖励：`RewardShapingWrapper` 增加空枪惩罚（未命中惩罚）并增强对 `HITCOUNT/AMMO2` 的读取能力（后备查询）。
- cfg/WAD：`src/envs/vizdoom_env.py` 增加自动修补逻辑，确保 `available_game_variables` 包含 `HITCOUNT`，并把 WAD 安放到 `src/_vizdoom/`（若缺失则提示）。
- 音频：设置环境变量并在底层禁音，尽量抑制 PipeWire/ALSA 报错以提升训练帧率。

---

## 验证（debug & 可视化）
- 生成 agent 观察截图：
```bash
python src/debug_vision.py
# 截图会保存在 dist/debug_vision/，检查敌人是否与墙面区分明显
```

- 快速评估（自动找最新 checkpoint 并保存带时间戳视频目录）：
```bash
# 推荐先保存为脚本 scripts/run_eval_latest.sh 并运行
bash -lc "CKPT_DIR='train_dir/defend_center_v4_6/checkpoint_p0'; CKPT=\"\$(ls -1t \"\$CKPT_DIR\"/checkpoint_*.pth 2>/dev/null | head -n1 || true)\"; if [ -z \"\$CKPT\" ]; then CKPT=\"\$(find train_dir -type f -name 'checkpoint_*.pth' -print | sort -V | tail -n1 || true)\"; fi; if [ -z \"\$CKPT\" ]; then echo 'No checkpoints found'; exit 1; fi; TS=\"\$(date +%Y%m%d-%H%M%S)\"; VID_DIR=dist/enjoy_defend_center_v4_6_\${TS}; mkdir -p \"\$VID_DIR\"; PYTHONPATH=. python src/evaluate.py --checkpoint \"\$CKPT\" --env custom_doom_defend_the_center --episodes 3 --video-dir \"\$VID_DIR\" --device cpu"
```

---

## 如何快速生成 checkpoint（短跑训练示例）
```bash
# 用较短步数生成第一个 checkpoint 用于验证
python -m sf_examples.vizdoom.train_vizdoom \
  --algo=APPO \
  --env=doom_defend_the_center \
  --experiment=defend_center_v4_6 \
  --train_dir=./train_dir \
  --device=cpu \
  --num_workers=1 \
  --num_envs_per_worker=2 \
  --train_for_env_steps=10000 \
  --save_every_sec=60 \
  --with_wandb=False
```

---

## 日志/排查要点（常见问题）
- FileDoesNotExistException: 若引擎找不到 WAD，请将对应 `.wad` 文件复制到 `src/_vizdoom/`（容器内示例）：
```bash
cp /usr/local/lib/python3.10/site-packages/vizdoom/scenarios/defend_the_center.wad src/_vizdoom/
```
- HITCOUNT 不在 `env.info`: 确认 patched cfg 中 `available_game_variables` 包含 `HITCOUNT`，或检查 wrapper 日志（已添加后备查询/提醒）。
- PipeWire/ALSA 报错仍有少量输出：通常无害，但可通过 ENV 设置 `ALSOFT_DRIVERS=null` 与 `SDL_AUDIODRIVER=dummy` 并禁音以减少噪声。

---

## 推荐的下一步（优先级）
1. 启动短跑训练（10k 步）生成 checkpoint → 运行评估脚本并检查视频/命中率。 
2. 若命中率仍低：适当提高命中奖励或调整空枪惩罚阈值，进行 1-2 次小范围超参试验。 
3. 对比输入分辨率（128x96 vs 160x120）对远景目标的影响，决定最终输入尺寸。
4. 将 `scripts/run_eval_latest.sh` 与 `scripts/check_scenarios.py` 加入 CI（自动检测 WAD/cfg/checkpoint）。

---

## 文件/位置速查
- 视觉/奖励/动作：`src/envs/wrappers.py`
- cfg/WAD 修补与 env 创建：`src/envs/vizdoom_env.py`
- debug 截图：`src/debug_vision.py` → `dist/debug_vision/`
- 评估：`src/evaluate.py`

---

如需我把这份文档再转为更详细的 `README` 页面并提交 PR，或现在在容器里跑一次短训练 + 评估并把视频路径贴回，告诉我你的优先项即可。🎯
