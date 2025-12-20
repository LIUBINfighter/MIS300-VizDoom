# Wrappers 更新说明（v4_6 · 强化版）

简要描述本次对 `src/envs/wrappers.py` 的更新、设计初衷、验证步骤和后续建议。

---

## 1) 概览 🔍
本次更新旨在解决 Agent 在 `Defend the Center` 场景中的 **瞄准失败** 问题。核心思路：让观察更保留几何信息、让动作可组合（同时转头与开火）、并通过奖励引导精确射击。

受影响的类：
- `RewardShapingWrapper`（奖励重塑，加入“狙击”逻辑）
- `ImageCleaningWrapper`（视觉修复，保持 4:3 比例，目标分辨率 128x96）
- `CompositeActionWrapper`（组合动作支持，离散→按钮/列表映射）

---

## 2) 关键改动要点 ✨

### RewardShapingWrapper（狙击手逻辑） 🔫
- 目的：**抑制滥射**，鼓励在有把握时才开火。
- 改动：
  - 基础生存奖励：+0.02（鼓励存活）
  - 命中（HIT）奖励：+2.0/次
  - 击杀（KILL）奖励：+10.0/次
  - 空枪惩罚：**-1.0**（开枪但未命中时；比之前更严格）
  - 捡到弹药奖励：+0.5（鼓励弹药管理）
- 诊断信息：`info` 字段注入 `HIT_INC`, `KILL_INC`, `AMMO_USED` 便于观察和调参。

### ImageCleaningWrapper（视觉修复） 🖼️
- 目标分辨率：**128x96 (4:3)**，保持长宽比尽量不扭曲场景。
- 流程：HWC normalize → 垂直裁剪（去 HUD）→ cv2.INTER_AREA 缩放 → CHW → 归一化到 [0,1]
- 理由：避免把 4:3 画面挤压成正方形，保留远处敌人的像素信息，利于目标检测与角度学习。

### CompositeActionWrapper（组合动作） 🎯
- 离散动作空间扩展为 5 个：
  - 0: 左
  - 1: 右
  - 2: 开火
  - 3: 左 + 开火
  - 4: 右 + 开火
- 返回给底层的实际动作是基于按钮列表，例如 `[1,0,1]`。
- 与底层交互：如果底层环境接受 list/tuple action，则直接传入；否则需要映射到底层离散索引（wrapper 中做了回退策略）。

---

## 3) 为什么这些改变有效（简短解释） 💡
- 保持宽高比可防止准心与真实角度信息不一致，CNN 能更稳定地学习到像素→角度的映射。
- 允许“转头+开火”的组合减少 Agent 犯“先转后打”的延迟，提升实战效能。
- 更严格的空枪惩罚会使策略权衡是否开火，从而提升命中率（需要在训练中观察是否过严）。

---

## 4) 验证步骤（快速清单） ✅
1. 检查 cfg/WAD 是否就位：`ls src/_vizdoom/defend_the_center.*`；cfg 中应包含 `available_game_variables = { KILLCOUNT HITCOUNT AMMO2 HEALTH FRAGCOUNT }`。
2. 运行可视化调试：
   ```bash
   python src/debug_vision.py
   # 查看 dist/debug_vision/ 中截图，确认画面长宽比与 HUD 被切除
   ```
3. 评估（自动找最新 checkpoint 并生成时间戳视频）：
   - 推荐脚本：`scripts/run_eval_latest.sh`（若已存在，直接运行）
   - 或用一行命令（详见仓库 README 或 docs/defend_center_v4_6.md）
4. 观察指标：`average reward`, `HIT_INC / AMMO_USED`（计算命中率），episode length。

---

## 5) 调参建议与注意事项 ⚖️
- 若 Agent 变得**过于保守**（几乎不射）：降低空枪惩罚到 -0.2 或 -0.5，或提高命中奖励。
- 若 Agent 仍**滥射但命中率低**：进一步提高空枪惩罚 + 增强命中正奖励。
- 监控训练中 `AMMO_USED` 与 `HIT_INC` 的比率（命中率），并在多次种子试验上比较。

---

## 6) 未来改进方向（后续迭代）
- 支持更细粒度的转头角度（continuous turn）或把动作空间改为多维动作（turn amount + fire）。
- 在 RewardShaping 中加入基于射击精度的奖励（命中远处目标权重更高）。
- 加入自动化 CI 检测脚本：`scripts/check_scenarios.py`，在训练前检查 WAD/cfg 与 `available_game_variables`。

---

## 7) 相关文件速查 🔎
- wrappers: `src/envs/wrappers.py`
- env 创建 / cfg 修补：`src/envs/vizdoom_env.py`
- debug 图像生成：`src/debug_vision.py` → `dist/debug_vision/`
- 评估脚本：`src/evaluate.py`

---

如需我把这篇文档合并进 `docs/defend_center_v4_6.md` 或创建 PR，请回复我合并意向，我会把它合并并提交改动。