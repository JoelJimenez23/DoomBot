# eval/bot_evaluation.py
from __future__ import annotations

import sys
from pathlib import Path

# ------------------------------------------------------------------
# Resolver rutas del proyecto
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent  # .../DoomBot
UTILS_DIR = PROJECT_ROOT / "utils"

# Importante: para poder hacer "from utils.xxx import yyy"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------
# Imports del proyecto
# ------------------------------------------------------------------
from utils.keys import GlobalKeyState, load_yaml, build_action_vector, ask_to_start

# OJO: según tu tree, doom_env/env.py existe; por eso:
from doom_env.env import DoomTaskEnv
from doom_env.tasks.aim_shoot import TaskAimShoot
from stable_baselines3 import PPO

import time

def main():
    print("\n=== DOOM ===")
    if not ask_to_start():
        print("Saliendo.")
        return

    # --------------------------------------------------------------
    # Cargar config desde utils/
    # --------------------------------------------------------------
    config_path = UTILS_DIR / "game_config.yaml"
    try:
        _game_cfg = load_yaml(str(config_path))  # solo para validar que existe y es YAML válido
    except Exception as e:
        print(f"ERROR al cargar config de juego: {e}")
        sys.exit(1)

    # --------------------------------------------------------------
    # Crear env (ajusta según tu DoomTaskEnv)
    # --------------------------------------------------------------
    env = DoomTaskEnv(_game_cfg,TaskAimShoot())  # probablemente necesitas pasarle cfg/task; aquí lo dejaste placeholder
    model = PPO.load("models/ppo_doom_center", device="cpu")

    obs, info = env.reset()
    episode = 0
    ep_r = 0.0
    
    shots = 0
    shot_events = 0
    ammo_spent = 0.0
    prev_ammo = None

    while True:
        print("obs: ",obs)
        action, _ = model.predict(obs, deterministic=True)
        print("action: ",action)
        obs, reward, terminated, truncated, info = env.step(action)
        print("new obs: ",obs,"reward :", reward ,"info: ",info)
        ep_r += float(reward)

        time.sleep(1 / 35.0)

        if info.get("fired", False):
            shots += 1
        if info.get("shot_event", False):
            shot_events += 1
        d_ammo = float(info.get("d_ammo", 0.0))
        if d_ammo < 0:
            ammo_spent += -d_ammo

        done = terminated or truncated
        if done:
            episode += 1
            print(
                f"Episode {episode} | Return={ep_r:+.3f} | "
                f"Health={info.get('health', '?')} | "
                f"Kills={info.get('kills', '?')} | "
                f"GoalPresent={info.get('goal_present', '?')} | "
                f"GoalDist={info.get('goal_dist', '?')} | "
                f"TaskDone={info.get('task_done', False)}"
                f"... | Shots={shots} | ShotEvents={shot_events} | AmmoSpent={ammo_spent:.1f}"
            )
            obs, info = env.reset()
            ep_r = 0.0



if __name__ == "__main__":
    main()