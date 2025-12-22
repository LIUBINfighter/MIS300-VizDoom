# UV (pip+venv) 与 Poetry 兼容迁移说明（开发者指南）

**目的**
本说明介绍我们为了兼容两类包管理方式（Poetry 与传统 pip+venv——简称 UV）而做的改动、如何在本地/容器中使用，以及 CI/验证建议。我们默认把 UV（pip + venv）作为首选安装方式以加快构建与提高兼容性，但依然保留 Poetry 工作流。

---

## 1. 主要改动概览 ✅
- 新增并提交了 `requirements.txt`（一个最小的、已固定版本的依赖列表，用于 pip 安装）。
- 添加跨平台安装脚本：
  - `scripts/install.sh`（Unix）
  - `scripts/install.ps1`（Windows PowerShell）
- 添加 `scripts/export-requirements.*`（用于从 Poetry 导出完整 `requirements.txt`）。
- 添加 `scripts/check_deps.py`：本地检查已安装包是否与 `requirements.txt` 大致匹配。
- 修改 `Dockerfile`：新增构建参数 `INSTALL_METHOD`（取值 `uv` 或 `poetry`），默认 `uv`（pip）。
- 增加文档：`docs/uv_poetry_compat.md` 与本指南（`docs/dev/uv_poetry_migration.md`）。

---

## 2. 本地开发：使用 UV（推荐）
### Windows（PowerShell）
1. 创建并激活虚拟环境（脚本会为你处理）并安装依赖：
   ```powershell
   .\scripts\install.ps1 -method pip -venv venv
   ```
2. 安装完成后，运行基本检查：
   ```powershell
   python scripts/check_deps.py
   python quick_test.py
   ```

### Unix / macOS
1. 使用脚本：
   ```bash
   ./scripts/install.sh pip .venv
   source .venv/bin/activate
   python scripts/check_deps.py
   python quick_test.py
   ```

### 常见问题
- 若缺少 `requirements.txt`，脚本会尝试使用 Poetry（若已安装）导出。否则请先运行 `poetry export -f requirements.txt --without-hashes -o requirements.txt`。
- 若需要 GPU 特定的 `torch` 包，请参考 PyTorch 官网并手动在虚拟环境中安装相应 wheel，然后可把版本或源写入 `requirements.txt`。

---

## 3. 使用 Poetry（备用）
1. 若偏好 Poetry：
   ```bash
   poetry install
   ```
2. 如需从 Poetry lock 文件导出 `requirements.txt`（以便使用 pip/uv）：
   ```bash
   poetry export -f requirements.txt --without-hashes -o requirements.txt
   # or use provided helper
   ./scripts/export-requirements.sh
   ```

---

## 4. Docker 构建与验证（强烈建议做 Smoke Test）
### 构建（默认使用 UV/pip）
- 构建镜像（使用 pip）：
  ```bash
  docker build --build-arg INSTALL_METHOD=uv -t mis300-vizdoom:latest .
  ```
- 构建镜像（使用 Poetry）：
  ```bash
  docker build --build-arg INSTALL_METHOD=poetry -t mis300-vizdoom:poetry .
  ```

### 运行与快速验证
- 运行默认容器以检查入口：
  ```bash
  docker run --rm -it mis300-vizdoom:latest
  ```
- 或直接运行快速 smoke test：
  ```bash
  docker run --rm -it mis300-vizdoom:latest python quick_test.py
  ```

### 注意
- 由于 ViZDoom 依赖系统库，Dockerfile 已包含必要系统包（xvfb、ffmpeg 等）。若构建失败，请查看 Docker build 日志中的第一处错误并根据提示安装缺失系统库或修正依赖版本。

---

## 5. CI 建议（GitHub Actions 或其它 CI）
建议在 CI 中加入至少两条 workflow/job：
- job A (uv): 构建镜像或在 runner 上用 `./scripts/install.sh pip` 安装，运行 `python quick_test.py` 或最小 smoke test。该 job 重点保证 pip 安装路径可用。
- job B (poetry): 使用 `poetry install`，运行同样的 smoke test，保证 Poetry 路径不被回退。 

示例（伪 YAML）片段：
```yaml
jobs:
  smoke_uv:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build with pip
        run: docker build --build-arg INSTALL_METHOD=uv -t mis300-vizdoom:ci .
      - name: Run smoke test
        run: docker run --rm mis300-vizdoom:ci python quick_test.py

  smoke_poetry:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build with poetry
        run: docker build --build-arg INSTALL_METHOD=poetry -t mis300-vizdoom:ci-poetry .
      - name: Run smoke test
        run: docker run --rm mis300-vizdoom:ci-poetry python quick_test.py
```

---

## 6. 依赖更新流程（维护者）
1. 使用 Poetry 修改依赖并更新 `poetry.lock`：
   ```bash
   poetry add <package>    # 或 poetry update
   git add pyproject.toml poetry.lock
   ```
2. 导出 `requirements.txt`（确保 pip 路径同步）：
   ```bash
   poetry export -f requirements.txt --without-hashes -o requirements.txt
   git add requirements.txt
   ```
3. 提交 PR，CI 应包含 uv/poetry 的 smoke tests。

---

## 7. 故障排查小贴士
- 构建失败在 `poetry install` 或 `pip install` 中常见原因：缺少系统级依赖（看日志头几行），或与 Python 版本不兼容的二进制包（尤其是 `torch`）。
- 若 `pip install -r requirements.txt` 失败：优先查看 pip 输出的第一个 ERR，若是二进制 wheel 问题，尝试手动安装合适的 wheel 或在 Dockerfile 中增加必要的 apt 包。

---

## 8. 后续任务建议
- 在 CI 中实现并开启 uv/poetry 的 smoke tests（我可以帮你提交 PR 添加 workflow）。
- 考虑为 `torch` 提供可选的 GPU wheel 安装文档片段或脚本（以便能在 GHA self-hosted 或内部 runner 上简化 GPU 配置）。

---

如果你同意，我接下来可以：
- 在 CI 中添加上述两个 smoke-test jobs 并提交 PR（需你确认是否允许我创建分支并提交）。
- 或先把该文档调整成更短的“快速开始”并放到 `README` 中。