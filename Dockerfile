# 使用 3.10 版本，稳定支持 Sample Factory 和 ViZDoom
FROM python:3.10-slim

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Poetry 配置：不在容器内创建虚拟环境
    POETRY_VIRTUALENVS_CREATE=false

# 1. 安装系统级依赖 (ViZDoom 编译与运行必需)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libsdl2-dev \
    libjpeg-dev \
    libfluidsynth-dev \
    libgme-dev \
    libopenal-dev \
    libwildmidi-dev \
    pkg-config \
    # 虚拟显示与视频处理
    xvfb \
    xauth \
    ffmpeg \
    # 基础工具
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. 可选安装包管理工具（默认使用 uv/pip，使用 poetry 可作为备用）
#    通过构建时参数切换：
#      docker build --build-arg INSTALL_METHOD=uv .    # 使用 pip+requirements.txt（默认）
#      docker build --build-arg INSTALL_METHOD=poetry .# 使用 Poetry
ARG INSTALL_METHOD=uv
RUN if [ "$INSTALL_METHOD" = "poetry" ]; then \
    curl -sSL https://install.python-poetry.org | python3 -; \
  else \
    echo "Skipping Poetry installation (INSTALL_METHOD=$INSTALL_METHOD)"; \
  fi
ENV PATH="/root/.local/bin:$PATH" \
    # 增加超时时间，防止大文件下载失败
    POETRY_HTTP_TIMEOUT=600

# 3. 设置工作目录
WORKDIR /app

# 4. 拷贝依赖描述文件 (利用 Docker 缓存机制)
COPY pyproject.toml poetry.lock* requirements.txt /app/

# 5. 安装依赖 (不安装项目本身，只安装依赖)
#    支持两种方式：uv (pip + requirements.txt) 或 poetry
RUN if [ "$INSTALL_METHOD" = "uv" ]; then \
    python -m pip install --upgrade pip setuptools wheel && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; else echo "requirements.txt missing"; exit 1; fi; \
  else \
    poetry install --no-interaction --no-ansi --no-root; \
  fi

# 6. 拷贝源代码
COPY . /app/

# 7. 启动脚本：默认开启虚拟屏幕渲染
# 这是一个包装器，确保所有 GUI 相关的调用都在 xvfb 中运行
ENTRYPOINT ["xvfb-run", "-s", "-screen 0 640x480x24"]
CMD ["python", "main.py"]
