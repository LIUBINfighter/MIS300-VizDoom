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

# 2. 安装包管理工具 Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# 3. 设置工作目录
WORKDIR /app

# 4. 拷贝依赖描述文件 (利用 Docker 缓存机制)
COPY pyproject.toml poetry.lock* /app/

# 5. 安装依赖 (不安装项目本身，只安装依赖)
# 如果没有 poetry.lock 会自动创建
RUN poetry install --no-interaction --no-ansi --no-root

# 6. 拷贝源代码
COPY . /app/

# 7. 启动脚本：默认开启虚拟屏幕渲染
# 这是一个包装器，确保所有 GUI 相关的调用都在 xvfb 中运行
ENTRYPOINT ["xvfb-run", "-s", "-screen 0 640x480x24"]
CMD ["python", "main.py"]
