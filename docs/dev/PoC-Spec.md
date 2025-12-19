好的，这是一份基于**绞杀榕（Strangler Fig）模式**的重构 Spec 文档。

我们的核心策略是：**保持现有的 Sample Factory (SF) 训练管线（宿主）运行，逐步将内部的“黑盒”组件（环境处理、网络模型、评估逻辑）替换为符合作业要求的“白盒”自定义实现（新枝），直到旧逻辑完全被“绞杀”**。

---

# MIS300-VizDoom 重构技术规格说明书 (Spec Doc)

**版本**: 0.1 (PoC to MVP)
**策略模式**: 绞杀榕 (Strangler Fig Application)
**目标**: 将高度封装的 PoC 项目转化为符合“视觉+记忆+算法工程”要求的交付作业。

## 0. 总体路线图 (Roadmap)

我们将分四个阶段进行“绞杀”替换：

1.  **Phase 1 - 根系重塑 (Environment & Data)**: 替换场景配置，接管数据预处理（Gym Wrappers）。
2.  **Phase 2 - 躯干替换 (Model Architecture)**: 接管神经网络定义，手写 CNN 和 LSTM。
3.  **Phase 3 - 输送养分 (Reward & Training)**: 注入自定义奖励函数，验证 PPO/RNN 收敛性。
4.  **Phase 4 - 果实交付 (Delivery & Evaluation)**: 封装最终评估脚本与 Docker 镜像。

---

## Phase 1: 根系重塑 (Environment & Data)

**目标**：接管 VizDoom 的原始数据流，不再依赖 Sample Factory 的默认预处理。

### 1.1 场景迁移 (Scenario Migration)

- **现状**: 使用 `DefendTheCenter` (默认)。
- **动作**:
  - 将 `src/envs/vizdoom_env.py` 中的配置指向 `health_gathering_supreme.cfg` (迷宫寻宝)。
  - **验证**: 运行 `test_vizdoom_env.py`，确认窗口中显示的是迷宫场景而非圆形竞技场。

### 1.2 预处理接管 (Preprocessing Injection)

- **现状**: 依赖 SF 内部的 `Resize` 和 `GrayScale` (隐式)。
- **动作**:
  - 编写明确的 `ImageCleaningWrapper` (在 `src/envs/wrappers.py`)。
  - **显式实现**:
    - `Crop`: 切除底部状态栏（VizDoom 底部通常有数据条）。
    - `Grayscale`: 使用 OpenCV 或 Numpy 将 RGB 转单通道。
    - `Resize`: 强制缩放至 `84x84`。
    - `Normalize`: 归一化至 `[0, 1]` (可选，通常在 Model 前端做，但 Wrapper 层需转 Float)。
- **绞杀点**: 在 `create_vizdoom_env` 工厂函数中，**显式包裹**这个 Wrapper，并关闭 SF 配置中的默认图像处理参数。
- **验证**: 打印 `env.observation_space`，必须是 `Box(1, 84, 84)`。

---

## Phase 2: 躯干替换 (Model Architecture)

**目标**：接管“大脑”，满足作业对 **CNN** 和 **LSTM** 代码可见性的要求。这是最核心的“绞杀”步骤。

### 2.1 视觉层重写 (CNN Implementation)

- **现状**: SF 默认生成的 Encoder (黑盒)。
- **动作**:
  - 在 `src/models` 下新建模型文件。
  - 继承 SF 的 `ActorCriticEncoder` 接口。
  - **手写 PyTorch 代码**: 实现类 `NatureCNN` 结构 (3 层 Conv2d + ReLU + Flatten)。
- **绞杀点**: 在 `train_custom.py` 中注册自定义模型，替换默认 Encoder。

### 2.2 记忆层注入 (LSTM/GRU Injection)

- **现状**: 仅通过参数 `--use_rnn=True` 开启（黑盒）。
- **动作**:
  - 在自定义模型类的 `forward` 函数中，显式定义 `nn.LSTM` 或 `nn.GRU` 单元。
  - **关键点**: 处理 `hidden_state` 的输入与输出。
  - _注_: SF 框架处理了跨 Episode 的状态重置，我们只需定义网络结构。如果作业要求极高，需在报告中展示这部分 `forward` 代码。

### 2.3 验证 (Model Verification)

- **验证**: 启动训练脚本，观察控制台输出的模型结构 (`print(model)`)。必须看到自定义的 `Conv2d` 和 `LSTM` 层，而非 SF 的默认层命名。

---

## Phase 3: 输送养分 (Reward & Algo)

**目标**：通过 Reward Shaping 让模型在复杂场景（迷宫）中真正学会生存。

### 3.1 奖励函数工程 (Reward Engineering)

- **现状**: `RewardShapingWrapper` 针对的是射击场景。
- **动作**:
  - 针对 `Health Gathering Supreme` 重写 `step` 函数中的奖励逻辑。
  - **设计**:
    - `shaping_reward = 0.0`
    - 如果检测到获得医疗包 (`Medkit`) -> `+10` (稀疏变稠密)。
    - 如果检测到踩毒 (`Poison`) -> `-10`。
    - 存活每一步 -> `+0.01` (鼓励生存)。
    - （可选）探测距离：利用 RayCasting 变量给予避障奖励。

### 3.2 算法参数调优 (Ablation Setup)

- **现状**: 默认 APPO 参数。
- **动作**:
  - 准备两套 Config (通过命令行参数区分)：
    1.  `Baseline`: 关闭 RNN，仅用 CNN。
    2.  `Main`: 开启 CNN + LSTM。
  - **验证**: 在 Tensorboard/WandB 中能看到两条对应的 Loss/Reward 曲线，且 `Main` 曲线在后期表现优于 `Baseline`。

---

## Phase 4: 果实交付 (Delivery & Evaluation)

**目标**：脱离开发环境，构建符合助教验收标准的交付物。

### 4.1 评估脚本标准化 (Evaluation Script)

- **现状**: `run_enjoy_safe.py` (包含补丁和复杂参数)。
- **动作**:
  - 新建根目录下的 `evaluate.py`。
  - **硬编码** 环境配置和模型路径（或自动寻找最新 Checkpoint）。
  - 集成 `vizdoom` 的录制功能或 `gym.wrappers.RecordVideo`，确保运行结束生成 `.mp4`。
  - 移除对 `PYTHONPATH` 的依赖（通过 `sys.path.append` 或相对导入解决）。

### 4.2 Docker 容器化 (Containerization)

- **现状**: `docker-compose` + `main.py` 检查脚本。
- **动作**:
  - 修改 `Dockerfile`。
  - 安装系统级依赖: `ffmpeg` (视频编码必须), `libboost-all-dev`, `cmake`, `libsdl2-dev` 等。
  - **Entrypoint 改造**: 确保 `docker run ... python evaluate.py` 可以直接运行，不需要用户手动启动 Xvfb（使用 `xvfb-run -a python evaluate.py` 作为默认执行方式）。

---

## 风险管理 (Risk Management)

1.  **显存溢出 (OOM)**: 自定义 CNN + LSTM 可能比默认模型大。
    - _对策_: 在 `evaluate.py` 中加入 `torch.cuda.empty_cache()`，并限制 batch size。
2.  **收敛困难**: 迷宫场景可能初期很难学。
    - _对策_: 确保 Reward Shaping 足够稠密（每一步都有微小反馈），并进行超参数搜索（Learning Rate）。
3.  **依赖地狱**: 助教环境可能与开发环境不同。
    - _对策_: 严格依赖 Docker 交付。在构建镜像阶段完成所有编译（VizDoom 安装较慢）。

## 下一步行动 (Next Actions)

1.  **Start Phase 1**: 修改 `src/envs/vizdoom_env.py` 切换场景，并编写 `wrapper`。
2.  不要碰 `src/run_enjoy_safe.py`，直接开始写新的 `evaluate.py` 原型。
