# doom_env/tasks/aim_shoot.py
from __future__ import annotations
from typing import Any, Dict, Tuple, List, Optional
import numpy as np

from doom_env.tasks.base_task import BaseTask
from doom_env.core.state import EnvConfig, StepOutput
from doom_env.core.action_space import ATTACK_IDS
from doom_env.features.common import ENEMY_NAMES
from doom_env.features.enemies import (
    pick_main_enemy, enemy_topk_features, aim_best_from_enemies,
    near_from_area, aim_main_from_dxdy
)
from doom_env.rewards.aim_reward import compute_aim_reward

def aim_main_from_dxdy(dx: float, dy: float, sx: float = 0.35, sy: float = 0.70) -> float:
    dist = float(np.sqrt((dx*dx)/(sx*sx + 1e-9) + (dy*dy)/(sy*sy + 1e-9)))
    return float(np.clip(1.0 - dist, 0.0, 1.0))


class TaskAimShoot(BaseTask):
    def obs_dim(self, cfg: EnvConfig) -> int:
        K = int(getattr(cfg, "topk_enemies", 3))
        return 10 + 3*K + 1

    def make_obs_and_info(self, core, st) -> Tuple[np.ndarray, Dict[str, Any]]:
        screen = st.screen_buffer
        H, W = screen.shape[0], screen.shape[1]
        dets = core.get_detections(st, screen)

        enemies = [d for d in dets if d.get("name","") in ENEMY_NAMES]
        enemy_count = len(enemies)

        top_feats = enemy_topk_features(enemies, W, H, K=int(getattr(core.cfg, "topk_enemies", 3)), core_cfg=core.cfg)
        danger = float(core.recent_damage_timer / max(1, int(getattr(core.cfg, "recent_damage_window", 10))))
        aim_best = aim_best_from_enemies(enemies, W, H)

        enemy = pick_main_enemy(core, dets, W, H)
        
        gv = core.get_vars(st)
        if len(gv) == 5:
            health, armor, kills, weapon, ammo = gv
        else:
            health, kills, weapon, ammo = gv

        ammo_norm = np.clip(ammo / 50.0, 0.0, 1.0)
        health_norm = np.clip(health / 100.0, 0.0, 1.0)

        if enemy is None:
            obs = np.array([0,0,0,0,0,0,0,0, ammo_norm, health_norm, *top_feats, danger], dtype=np.float32)
            info = {
                "enemy_present": 0, "enemy_count": enemy_count,
                "aim_best": float(aim_best), "aim_main": 0.0, "near": 0.0,
                "danger": danger,
            }
            return obs, info

        cx = (enemy["x"] + 0.5*enemy["w"]) / W
        cy = (enemy["y"] + 0.5*enemy["h"]) / H
        bw = enemy["w"]/W
        bh = enemy["h"]/H
        area = (enemy["w"]*enemy["h"]) / (W*H + 1e-9)
        dx = cx - 0.5
        dy = cy - 0.5

        aim_main = aim_main_from_dxdy(dx, dy)
        near = near_from_area(area)

        obs = np.array([1,cx,cy,bw,bh,area,dx,dy, ammo_norm, health_norm, *top_feats, danger], dtype=np.float32)
        info = {
            "enemy_present": 1, "enemy_count": enemy_count,
            "aim_best": float(aim_best),
            "aim_main": float(aim_main),
            "near": float(near),
            "danger": danger,
            "area": float(area),
            "dx": float(dx),
        }
        return obs, info

    def action_override(self, core, action_id: int):
        st0 = core.game.get_state()
        if st0 is None:
            return int(action_id), {"requested_attack": bool(int(action_id) in ATTACK_IDS)}

        screen0 = st0.screen_buffer
        H0, W0 = screen0.shape[0], screen0.shape[1]
        dets0 = core.get_detections(st0, screen0)
        enemy0 = pick_main_enemy(core, dets0, W0, H0)

        requested_attack = bool(int(action_id) in ATTACK_IDS)
        blocked_by_cooldown = False

        if enemy0 is not None:
            # -----------------------------
            # Geometría del bbox
            # -----------------------------
            cx0 = (enemy0["x"] + 0.5 * enemy0["w"]) / W0
            # cy0 = (enemy0["y"] + 0.5 * enemy0["h"]) / H0
            dx0 = cx0 - 0.5
            # dy0 = cy0 - 0.5
            aim_y_frac = float(getattr(core.cfg, "aim_y_frac", 0.35))  # 0.30–0.45
            cy0 = (enemy0["y"] + aim_y_frac * enemy0["h"]) / H0
            dy0 = cy0 - 0.5


            bw0 = float(enemy0["w"]) / (W0 + 1e-9)   # ancho normalizado [0..1]
            bh0 = float(enemy0["h"]) / (H0 + 1e-9)
            area0 = (enemy0["w"] * enemy0["h"]) / (W0 * H0 + 1e-9)
            if "marinechainsawvzd" in enemy0.get("name", ""):
                marine_scale = 3.5
                area0 = min(1.0, area0 * (marine_scale ** 2))
            # -----------------------------
            # Suavizado de dx + lock-on
            # -----------------------------
            alpha = float(getattr(core.cfg, "dx_ema_alpha", 0.6))  # más alto = más suave
            if not hasattr(core, "dx_ema"):
                core.dx_ema = dx0
            core.dx_ema = alpha * core.dx_ema + (1.0 - alpha) * dx0
            dx_use = float(core.dx_ema)

            lock_dx = float(getattr(core.cfg, "lock_dx", 0.08))
            lock_k  = int(getattr(core.cfg, "lock_k", 4))
            if not hasattr(core, "lock_cnt"):
                core.lock_cnt = 0
            if abs(dx_use) <= lock_dx:
                core.lock_cnt += 1
            else:
                core.lock_cnt = 0
            locked = (core.lock_cnt >= lock_k)

            # -----------------------------
            # Aim score
            # -----------------------------

            fire_aim_far  = float(getattr(core.cfg, "fire_aim_far", 0.55))
            fire_aim_near = float(getattr(core.cfg, "fire_aim_near", 0.35))
            turn_dx_th    = float(getattr(core.cfg, "turn_dx_th", 0.06))
            auto_fire     = bool(getattr(core.cfg, "auto_fire", True))

            # vars
            gv0 = core.get_vars(st0)
            if len(gv0) == 5:
                health0, armor0, kills0, weapon0, ammo0 = gv0
            else:
                health0, kills0, weapon0, ammo0 = gv0
            weapon0 = int(weapon0)

            # HITSCAN_WEAPONS = {2, 3, 4}  # pistol/shotgun/chaingun
            ROCKET_WEAPONS  = {5, 6, 7}  # rocket launcher / variants

            # aim0 = aim_main_from_dxdy(dx0, dy0)
            # ✅ si no hay LOOK_UP/DOWN, dy es ruido para hitscan: usa aim 1D en dx
            HITSCAN_WEAPONS = {2, 3, 4}
            if int(weapon0) in HITSCAN_WEAPONS:
                aim0 = max(0.0, 1.0 - abs(dx0) / 0.5)   # mismo que aim_main_from_dx()
            else:
                aim0 = aim_main_from_dxdy(dx0, dy0)


            # -----------------------------
            # Threshold de aim por distancia (via area)
            # -----------------------------
            near0 = near_from_area(area0)
            aim_th = (1.0 - near0) * fire_aim_far + near0 * fire_aim_near

            # ✅ si el target es MUY pequeño/delgado, no seas tan exigente
            # (esto hace que marine y demon “se sientan” igual)
            if (bw0 < 0.08) or (area0 < 0.010):
                aim_th = min(aim_th, 0.55)

            can_shoot = (ammo0 > 0) and (core.attack_cooldown <= 0)

            # -----------------------------
            # ✅ Deadzone adaptativo (clave)
            # -----------------------------
            dz_base = float(getattr(core.cfg, "aim_deadzone", 0.03))

            # deadzone crece con el tamaño del bbox (y se asegura un mínimo decente)
            # - targets pequeños: permite más error angular (no “castiga” al marine)
            # - targets grandes: vuelve a dz_base
            dz_by_w = 0.60 * bw0            # 0.60 es buen punto
            dz_by_area = 0.12 * (area0 ** 0.5)  # suave, evita valores locos
            deadzone = max(dz_base, dz_by_w, dz_by_area)

            # clamp final para evitar extremos
            deadzone = float(np.clip(deadzone, 0.03, 0.12))

            # -----------------------------
            # ROCKETS (igual que antes pero con dx_use)
            # -----------------------------
            if auto_fire and can_shoot and (weapon0 in ROCKET_WEAPONS):
                rocket_area_block = float(getattr(core.cfg, "rocket_area_block", 0.26))
                too_close = (area0 >= rocket_area_block)

                if too_close:
                    action_id = 6  # backward (evita suicidio)
                else:
                    if locked:
                        action_id = 7
                    else:
                        if abs(dx_use) <= turn_dx_th:
                            action_id = 7
                        elif dx_use < 0:
                            action_id = 1
                        else:
                            action_id = 2

            # -----------------------------
            # HITSCAN (tratamiento “igualitario”)
            # -----------------------------
            elif auto_fire and can_shoot and (weapon0 in HITSCAN_WEAPONS):
                # ✅ usa dx_use + deadzone adaptativo
                if (aim0 >= aim_th) and (abs(dx_use) <= deadzone):
                    action_id = 7
                else:
                    # alinear sin disparar
                    if abs(dx_use) <= deadzone:
                        action_id = 0
                    elif dx_use < 0:
                        action_id = 1
                    else:
                        action_id = 2

            # -----------------------------
            # Otros
            # -----------------------------
            elif auto_fire and can_shoot:
                if (aim0 >= aim_th) and (abs(dx_use) <= deadzone):
                    action_id = 7
                else:
                    if abs(dx_use) <= deadzone:
                        action_id = 0
                    elif dx_use < 0:
                        action_id = 1
                    else:
                        action_id = 2

        requested_attack = bool(int(action_id) in ATTACK_IDS)

        if requested_attack and core.attack_cooldown > 0:
            blocked_by_cooldown = True
            action_id = 0

        return int(action_id), {
            "requested_attack": requested_attack,
            "blocked_by_cooldown": blocked_by_cooldown,
        }



    def compute_reward(self, core, step_out: StepOutput, info: Dict[str, Any]) -> float:
        # Base shaping que ya tienes
        r = float(compute_aim_reward(core, step_out, info))

        enemy_present = int(info.get("enemy_present", 0))

        fired = bool(info.get("fired", False))
        shot_event = bool(info.get("shot_event", False))

        # d_ammo < 0 cuando gastas balas
        d_ammo = float(info.get("d_ammo", 0.0))
        ammo_spent = max(0.0, -d_ammo)

        # Knobs (tuneables desde EnvConfig)
        hit_bonus = float(getattr(core.cfg, "rw_hit_bonus", 0.20))
        miss_pen  = float(getattr(core.cfg, "rw_miss_penalty", 0.05))
        ammo_pen  = float(getattr(core.cfg, "rw_ammo_penalty", 0.01))
        fire_no_enemy_pen = float(getattr(core.cfg, "rw_fire_no_enemy_penalty", 0.03))

        # 1) Premio por impacto
        if shot_event:
            r += hit_bonus

        # 2) Castigo por disparar y NO impactar (solo si hay enemigo)
        if enemy_present == 1 and fired and (not shot_event):
            r -= miss_pen

        # 3) Castigo por gastar munición (solo en combate)
        if enemy_present == 1 and ammo_spent > 0:
            r -= ammo_pen * ammo_spent

        # 4) Castigo si dispara sin enemigo visible
        if enemy_present == 0 and fired:
            r -= fire_no_enemy_pen

        return float(r)