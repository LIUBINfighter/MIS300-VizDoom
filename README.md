# MIS300-VizDoom

## Plan

- Env & `docker-compose build`
  - dockerfile, docker-compose.yml
  - pyproject.toml
  - main.py
  - 时间瓶颈1 `RUN apt-get update && apt-get install -y`  大约需要 `120~150s`
  - 时间瓶颈2 `RUN poetry install --no-interaction --no-ansi --no-root` 大约需要 `700~800s`
  - 总用时 `~1000s`

## 

```docker
python -m sf_examples.vizdoom.train_vizdoom   --algo=APPO   --env=doom_defend_the_center   --experiment=defend_center_v1   --train_dir=./train_dir   --device=cpu   --num_workers=1   --num_envs_per_worker=2   --train_for_env_steps=500000   --save_every_sec=300   --with_wandb=False
```

```docker
python src/run_enjoy_safe.py     --env=doom_defend_the_center     --experiment=defend_center_v1     --save_video     --video_frames=1500     --max_num_episodes=5

```

## Documentation 技术上下文

## License

[Mozilla Public License Version 2.0](./LICENSE)
