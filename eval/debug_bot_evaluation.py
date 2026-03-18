# eval/bot_evaluation.py
from __future__ import annotations

import sys
from pathlib import Path


import os

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)

def box(title, content_lines, width=70):
    print("┌" + "─"*(width-2) + "┐")
    print(f"│ {title}".ljust(width-1) + "│")
    print("├" + "─"*(width-2) + "┤")

    for line in content_lines:
        print(f"│ {line}".ljust(width-1) + "│")

    print("└" + "─"*(width-2) + "┘")


def format_obs(obs):
    return [
        f"enemy_present : {obs[0]:.2f}",
        f"enemy_x       : {obs[1]:.3f}",
        f"enemy_y       : {obs[2]:.3f}",
        f"enemy_w       : {obs[3]:.3f}",
        f"enemy_h       : {obs[4]:.3f}",
        f"area          : {obs[5]:.4f}",
        f"dx            : {obs[6]:.3f}",
        f"dy            : {obs[7]:.3f}",
        f"ammo_norm     : {obs[8]:.2f}",
        f"health_norm   : {obs[9]:.2f}",
    ]


def format_info(info):
    return [
        f"ammo        : {fmt(info.get('ammo'))}",
        f"health      : {fmt(info.get('health'))}",
        f"kills       : {fmt(info.get('kills'))}",
        f"weapon      : {fmt(info.get('weapon'))}",
        f"enemy_count : {fmt(info.get('enemy_count'))}",
        f"aim_main    : {fmt(info.get('aim_main'))}",
        f"near        : {fmt(info.get('near'))}",
        f"area        : {fmt(info.get('area'), 4)}",
        f"dx          : {fmt(info.get('dx'))}",
    ]




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

    step = 0

    while True:

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        ep_r += float(reward)
        step += 1

        clear()

        box(
            "STEP INFO",
            [
                f"step      : {step}",
                f"action    : {action}",
                f"reward    : {reward:+.4f}",
                f"episode_r : {ep_r:+.3f}",
            ],
        )

        box("OBSERVATION", format_obs(obs))

        box("GAME STATE", format_info(info))

        box(
            "SHOOTING STATS",
            [
                f"shots        : {shots}",
                f"shot_events  : {shot_events}",
                f"ammo_spent   : {ammo_spent:.1f}",
            ],
        )

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
            print("\nEPISODE END")
            print(
                f"Episode {episode} | Return={ep_r:+.3f} | "
                f"Kills={info.get('kills')} | Health={info.get('health')}"
            )

            obs, info = env.reset()
            ep_r = 0.0
            step = 0

if __name__ == "__main__":
    main()