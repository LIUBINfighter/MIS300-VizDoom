#!/usr/bin/env python
"""快速诊断脚本 - 不启动游戏引擎"""
import sys
print("1. Python 环境检查...", flush=True)
print(f"   Python 版本: {sys.version}", flush=True)

print("\n2. 导入 ViZDoom...", flush=True)
try:
    import vizdoom as vzd
    print(f"   ✓ ViZDoom 版本: {vzd.__version__}", flush=True)
except Exception as e:
    print(f"   ✗ 导入失败: {e}", flush=True)
    sys.exit(1)

print("\n3. 检查场景路径...", flush=True)
try:
    scenarios_path = vzd.scenarios_path
    print(f"   场景路径: {scenarios_path}", flush=True)
    
    import os
    if os.path.exists(scenarios_path):
        files = os.listdir(scenarios_path)
        cfg_files = [f for f in files if f.endswith('.cfg')]
        print(f"   ✓ 找到 {len(cfg_files)} 个配置文件", flush=True)
        print(f"   配置文件: {cfg_files[:5]}", flush=True)
    else:
        print(f"   ✗ 路径不存在", flush=True)
except Exception as e:
    print(f"   ✗ 检查失败: {e}", flush=True)

print("\n4. 测试创建游戏实例（不初始化）...", flush=True)
try:
    game = vzd.DoomGame()
    print("   ✓ 游戏实例创建成功", flush=True)
    game.close()
except Exception as e:
    print(f"   ✗ 创建失败: {e}", flush=True)
    sys.exit(1)

print("\n=== ✓ 基础诊断通过 ===", flush=True)
