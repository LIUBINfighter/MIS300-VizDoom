#!/usr/bin/env python
"""
ViZDoom 环境测试脚本
测试场景加载、动作执行和渲染是否正常
"""
import vizdoom as vzd
import numpy as np


def test_basic_scenario():
    """测试基础场景加载"""
    print("=== ViZDoom 环境测试开始 ===\n")
    
    # 创建游戏实例
    game = vzd.DoomGame()
    
    # 使用内置的 Defend the Center 场景
    print("1. 加载场景配置...")
    game.load_config(vzd.scenarios_path + "/defend_the_center.cfg")
    
    # 设置窗口不可见（容器环境必须）
    game.set_window_visible(False)
    
    # 设置模式为玩家模式
    game.set_mode(vzd.Mode.PLAYER)
    
    # 初始化游戏
    print("2. 初始化游戏引擎...")
    game.init()
    print("   ✓ 游戏引擎初始化成功")
    
    # 获取可用动作
    available_actions = game.get_available_buttons()
    print(f"\n3. 可用动作按钮数: {len(available_actions)}")
    print(f"   按钮列表: {[str(btn) for btn in available_actions]}")
    
    # 创建简单的动作空间（仅左转、右转、射击）
    actions = [
        [1, 0, 0],  # 左转
        [0, 1, 0],  # 右转
        [0, 0, 1],  # 射击
    ]
    
    # 运行测试回合
    episodes = 3
    total_reward = 0
    
    print(f"\n4. 开始测试运行 {episodes} 个回合...\n")
    
    for episode in range(episodes):
        game.new_episode()
        episode_reward = 0
        
        while not game.is_episode_finished():
            # 获取游戏状态
            state = game.get_state()
            
            if state is None:
                break
            
            # 随机选择动作
            action = actions[np.random.randint(len(actions))]
            
            # 执行动作
            reward = game.make_action(action)
            episode_reward += reward
        
        total_reward += episode_reward
        
        # 获取游戏变量
        kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        ammo = game.get_game_variable(vzd.GameVariable.AMMO2)
        health = game.get_game_variable(vzd.GameVariable.HEALTH)
        
        print(f"   回合 {episode + 1}:")
        print(f"      奖励: {episode_reward:.2f}")
        print(f"      击杀数: {int(kills)}")
        print(f"      剩余弹药: {int(ammo)}")
        print(f"      剩余生命: {int(health)}")
    
    print(f"\n5. 测试完成!")
    print(f"   总奖励: {total_reward:.2f}")
    print(f"   平均奖励: {total_reward / episodes:.2f}")
    
    # 关闭游戏
    game.close()
    print("\n=== ✓ 环境测试通过 ===")
    
    return True


if __name__ == "__main__":
    try:
        test_basic_scenario()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
