import vizdoom as vzd
import os

def check_variables(scenario_name):
    game = vzd.DoomGame()
    # 尝试寻找场景文件
    # ViZDoom 通常在包目录下有 scenarios 文件夹
    scenarios_path = os.path.join(os.path.dirname(vzd.__file__), "scenarios")
    cfg_path = os.path.join(scenarios_path, scenario_name)
    
    if not os.path.exists(cfg_path):
        print(f"Could not find {scenario_name} at {cfg_path}")
        return

    game.load_config(cfg_path)
    game.set_window_visible(False)
    game.init()
    
    print(f"--- Variables in {scenario_name} ---")
    # 获取配置中定义的变量
    # 注意：ViZDoom API 并不直接提供“哪些变量被启用了”的列表，
    # 但我们可以尝试获取常见的变量
    vars_to_check = [
        vzd.GameVariable.FRAGCOUNT,
        vzd.GameVariable.HEALTH,
        vzd.GameVariable.AMMO2,
        vzd.GameVariable.KILLCOUNT,
        vzd.GameVariable.HITCOUNT,
    ]
    
    state = game.get_state()
    for v in vars_to_check:
        try:
            val = game.get_game_variable(v)
            print(f"{v.name}: {val}")
        except:
            print(f"{v.name}: Not available")
            
    game.close()

if __name__ == "__main__":
    check_variables("defend_the_center.cfg")
