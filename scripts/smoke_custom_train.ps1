# PowerShell smoke test for custom training
param(
  [string]$Env = "custom_doom_basic",
  [string]$TrainDir = "C:\\tmp\\smoke_train",
  [int]$Steps = 1000,
  [int]$NumWorkers = 1,
  [int]$NumEnvs = 2,
  [string]$Device = "cpu"
)

Write-Host "Running smoke custom training: env=$Env steps=$Steps workers=$NumWorkers envs=$NumEnvs device=$Device"
New-Item -ItemType Directory -Force -Path $TrainDir | Out-Null

$env:PYTHONPATH = "."
python src\train_custom.py --env $Env --num_workers $NumWorkers --num_envs_per_worker $NumEnvs --train_for_env_steps $Steps --device $Device --train_dir $TrainDir --save_every_sec 300 --with_wandb False
if ($LASTEXITCODE -eq 0) { Write-Host "Smoke training finished successfully" } else { Write-Host "Smoke training failed with exit code $LASTEXITCODE"; exit $LASTEXITCODE }
