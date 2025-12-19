#!/usr/bin/env python3
# src/convert_checkpoint.py
import sys
import torch
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python src/convert_checkpoint.py <checkpoint.pth> [out.pth]")
    sys.exit(2)

inp = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) >= 3 else inp.with_name(inp.stem + ".weights_only.pth")

print("Loading:", inp)
# 这里显式允许不安全加载旧格式（仅在你信任文件来源时使用）
checkpoint = torch.load(str(inp), map_location="cpu", weights_only=False)

# 常见存放位置为 checkpoint["model"]
if isinstance(checkpoint, dict) and "model" in checkpoint and checkpoint["model"] is not None:
    model_state = checkpoint["model"]
else:
    # 尝试从 dict 中找第一个看起来像 state_dict 的值
    model_state = None
    if isinstance(checkpoint, dict):
        for v in checkpoint.values():
            if isinstance(v, dict):
                model_state = v
                break

if model_state is None:
    print("Failed to extract model state_dict from checkpoint.")
    sys.exit(1)

torch.save({"model": model_state}, str(out))
print("Saved weights-only checkpoint:", out)