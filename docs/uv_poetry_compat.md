# 兼容说明：Poetry 与 pip (uv)

## 目标
使仓库同时支持 Poetry 和 pip+venv（下文称为 uv），并将 uv 作为首选以减少构建时间并方便大多数开发者。

## 文件与脚本
- `pyproject.toml` / `poetry.lock`：保留，支持 Poetry 用户。
- `requirements.txt`：仓库内提供一个最小、已固定版本的 `requirements.txt`，可直接用于 `pip install -r requirements.txt`。
- `scripts/install.ps1`：Windows 下安装脚本，支持 `-method pip|poetry`。
- `scripts/install.sh`：Unix 下安装脚本，支持 `pip|poetry`。
- `scripts/export-requirements.*`：若需要，可用 Poetry 导出完整的 `requirements.txt`。
- `scripts/check_deps.py`：用于在本地检查已安装包的版本是否与 `requirements.txt` 大致一致。

## Docker 与 CI
- Dockerfile 新增构建参数 `INSTALL_METHOD`：
  - `uv`（默认）: 使用 `pip install -r requirements.txt`。
  - `poetry`: 安装并使用 `poetry install`。

构建示例：

- 使用 pip/uv（默认/推荐）

  docker build --build-arg INSTALL_METHOD=uv -t mis300-vizdoom:latest .

- 使用 Poetry

  docker build --build-arg INSTALL_METHOD=poetry -t mis300-vizdoom:poetry .

验证示例：

  docker run --rm -it mis300-vizdoom:latest python quick_test.py

或运行容器默认命令查看是否能正常启动：

  docker run --rm -it mis300-vizdoom:latest

## 注意事项
- `requirements.txt` 是基于 `pyproject.toml` 的一个最小锁定示例；为了确保与 Poetry 同步，请在更新依赖后运行：

  poetry export -f requirements.txt --without-hashes -o requirements.txt

- 如果你使用 GPU 或与 `torch` 相关的特殊安装，请参考 PyTorch 官方安装说明并在 `requirements.txt` 中加入相应轮子源或手动安装。

## 下一步建议
- 在 CI 中加入对两种安装方式的简单 smoke-test（至少安装成功并运行 `python quick_test.py`）。
