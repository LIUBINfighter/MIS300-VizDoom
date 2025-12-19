# MIS300-VizDoom

## Plan

- Env & `docker-compose build`
  - dockerfile, docker-compose.yml
  - pyproject.toml
  - main.py
  - 时间瓶颈1 `RUN apt-get update && apt-get install -y`  大约需要 `120~150s`
  - 时间瓶颈2 `RUN poetry install --no-interaction --no-ansi --no-root` 大约需要 `700~800s`
  - 总用时 `~1000s`

## Documentation 技术上下文

## License

[Mozilla Public License Version 2.0](./LICENSE)
