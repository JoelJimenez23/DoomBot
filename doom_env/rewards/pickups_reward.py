# doom_env/rewards/pickups_reward.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class PickupsRewardCfg:
    # pickups reales (deltas)
    w_health_pickup: float = 2.0
    w_armor_pickup: float = 1.0
    w_ammo_pickup: float = 0.5

    # denso: acercarse al pickup (dist en pantalla)
    w_progress: float = 0.2

    # regularización
    idle_penalty: float = -0.002
    turn_spam_penalty: float = -0.0002

def compute_pickups_reward(
    d_health: float,
    d_armor: float,
    d_ammo: float,
    prev_dist: Optional[float],
    curr_dist: Optional[float],
    idle_steps: int,
    turn_streak: int,
    cfg: PickupsRewardCfg,
) -> float:
    r = 0.0

    # 1) señal principal: “pickup ocurrió”
    if d_health > 0:
        r += cfg.w_health_pickup
    if d_armor > 0:
        r += cfg.w_armor_pickup
    if d_ammo > 0:
        r += cfg.w_ammo_pickup

    # 2) progreso hacia el target (solo si hay dist válida)
    if prev_dist is not None and curr_dist is not None:
        r += cfg.w_progress * (prev_dist - curr_dist)

    # 3) anti-loop
    if idle_steps >= 10:
        r += cfg.idle_penalty * float(idle_steps)
    if turn_streak >= 10:
        r += cfg.turn_spam_penalty * float(turn_streak)

    return float(r)