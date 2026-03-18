# doom_env/rewards/aim_reward.py
from __future__ import annotations
from doom_env.core.action_space import (
    TURN_IDS, TURN_ATTACK_IDS, STRAFE_IDS, STRAFE_ATTACK_IDS, BACK_IDS, BACK_TURN_IDS,
    FWD_IDS, FWD_ATTACK_IDS, BACK_ATTACK_IDS, ATTACK_IDS
)

# ids típicos en Doom (pueden variar por wad, pero suele ser así)
HITSCAN_WEAPONS = {2, 3, 4}   # pistol, shotgun, chaingun
ROCKET_WEAPONS  = {5}         # rocket launcher


def compute_aim_reward(core, step_out, info):
    step_tics = max(1, int(getattr(core.cfg, "step_tics", 2)))

    ammo = float(info.get("ammo", 0.0))
    health = float(info.get("health", 0.0))

    # weapon (si no está, cae a -1 y se usa comportamiento default)
    weapon = int(float(info.get("weapon", -1)))

    low_ammo = ammo <= 2
    low_health = health <= 20
    panic = low_ammo or low_health

    enemy_count = int(info.get("enemy_count", 0))
    aim_best = float(info.get("aim_best", 0.0))
    aim_main = float(info.get("aim_main", 0.0))
    near = float(info.get("near", 0.0))

    d_ammo = float(step_out.d_ammo)
    d_health = float(step_out.d_health)
    d_kills = float(step_out.d_kills)

    # base + living
    reward = float(step_out.base_reward) + 0.001 * step_tics

    # aim diferencial
    prev = float(getattr(core, "prev_aim_best", 0.0))
    d_aim = aim_best - prev
    reward += 0.01 * d_aim
    core.prev_aim_best = aim_best

    # kills
    if d_kills > 0:
        reward += 1.5 * d_kills

    # daño + evasión
    if d_health < 0:
        reward -= 0.06 * (-d_health)

    shot_event = bool(info.get("shot_event", False))          # en tu env: ammo_spent
    requested_attack = bool(info.get("requested_attack", False))
    blocked_by_cooldown = bool(info.get("blocked_by_cooldown", False))
    action_id = int(info.get("action_id", 0))

    fire_aim_far  = float(getattr(core.cfg, "fire_aim_far", 0.55))
    fire_aim_near = float(getattr(core.cfg, "fire_aim_near", 0.35))

    # hitscan exige más precisión
    if weapon in HITSCAN_WEAPONS:
        fire_aim_far = max(fire_aim_far, 0.75)
        fire_aim_near = max(fire_aim_near, 0.65)

    aim_th = (1.0 - near) * fire_aim_far + near * fire_aim_near

    # ------------- COSTO DE AMMO (REGLA PRINCIPAL) -------------
    # d_ammo < 0 => gastó ammo real (proyectil/bala salió)
    if d_ammo < 0:
        if weapon in HITSCAN_WEAPONS:
            # hitscan: “no falles”
            if aim_main < aim_th:
                reward -= 0.06  # castigo fuerte por gastar bala sin estar alineado
            else:
                reward += 0.004  # pequeño refuerzo por tiro preciso
        elif weapon in ROCKET_WEAPONS:
            # rockets: costo moderado (ya es caro, pero no quieres spam)
            reward -= 0.015
        else:
            # default suave
            reward -= 0.008

    # -----------------------------------------------------------
    # shaping contextual (enemigo / panic / vacío)
    if enemy_count > 0 and not panic:
        reward += (0.006 + 0.010 * near) * aim_main

        if aim_main >= aim_th:
            if not shot_event:
                # si está alineado y NO disparó, castiga (pero si fue cooldown, suave)
                reward -= 0.02 if not blocked_by_cooldown else 0.004
            else:
                # ya dimos bonus por tiro preciso arriba; esto lo dejamos leve
                reward += 0.004
        else:
            if shot_event:
                # ya castigamos duro arriba para hitscan; aquí deja suave para no duplicar demasiado
                reward -= (0.005 * (1.0 - near))

        # regularizadores de movimiento
        if action_id in (TURN_IDS | TURN_ATTACK_IDS):
            reward -= 0.0008
        if action_id in (STRAFE_ATTACK_IDS | FWD_ATTACK_IDS | BACK_ATTACK_IDS):
            reward += 0.002

    elif enemy_count > 0 and panic:
        if requested_attack:
            reward -= 0.015
        if action_id in (BACK_IDS | BACK_TURN_IDS):
            reward += 0.03
        if action_id in STRAFE_IDS:
            reward += 0.01
        if action_id in TURN_IDS:
            reward += 0.005

    else:
        # sin enemigos: no dispares
        if requested_attack:
            reward -= 0.04
        if d_ammo < 0:
            reward -= 0.08
        if action_id in (BACK_IDS | BACK_TURN_IDS):
            reward -= 0.003
        if action_id in FWD_IDS:
            reward += 0.001
        if action_id in TURN_IDS:
            reward += 0.002

    # peligro reciente sin enemigos (buscar/evadir)
    danger = float(info.get("danger", 0.0))
    if danger > 0 and enemy_count == 0:
        if action_id in TURN_IDS:
            reward += 0.01
        if action_id in STRAFE_IDS:
            reward += 0.01
        if action_id == 0:
            reward -= 0.02
        if requested_attack:
            reward -= 0.05

    return float(reward)