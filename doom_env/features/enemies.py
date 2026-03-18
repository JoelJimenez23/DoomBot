# doom_env/features/enemies.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from doom_env.features.common import ENEMY_NAMES, is_enemy

def aim_main_from_dx(dx: float) -> float:
    return max(0.0, 1.0 - abs(dx) / 0.5)

def near_from_area(area: float, area0=0.01, area1=0.05) -> float:
    return float(np.clip((area - area0) / (area1 - area0 + 1e-9), 0.0, 1.0))

def aim_main_from_dxdy(dx: float, dy: float, scale: float = 0.35) -> float:
    """
    aim 2D: 1 cuando el centro del bbox está en el centro de pantalla.
    scale controla qué tan rápido cae. 0.35 funciona bien con 320x240.
    """
    dist = float(np.sqrt(dx*dx + dy*dy))
    return float(np.clip(1.0 - dist / max(1e-6, scale), 0.0, 1.0))

def enemy_features(e, W, H):
    cx = (e["x"] + 0.5*e["w"]) / W
    cy = (e["y"] + 0.5*e["h"]) / H
    area = (e["w"]*e["h"]) / (W*H + 1e-9)
    dx = cx - 0.5
    dy = cy - 0.5
    return dx, dy, area

# def enemy_threat(core_cfg, e, W, H) -> float:
#     name = str(e.get("name","")).strip().lower()
#     dx, dy, area = enemy_features(e, W, H)
#     center = max(0.0, 1.0 - abs(dx) / 0.5)

#     near0 = float(getattr(core_cfg, "near_area0", 0.004))
#     near1 = float(getattr(core_cfg, "near_area1", 0.030))
#     near = np.clip((area - near0) / (near1 - near0 + 1e-9), 0.0, 1.0)

#     if name in {"cacodemon"}:
#         base = 1.00
#     elif name in {"marinechainsawvzd","shotgunguy","chaingunguy"}:
#         base = 0.85
#     elif name in {"demon"}:
#         base = 0.70
#     else:
#         base = 0.60

#     threat = base * (near ** 1.7) * (0.75 + 0.25 * center)
#     return float(np.clip(threat, 0.0, 1.0))
def enemy_threat(core_cfg, e, W, H) -> float:
    name = str(e.get("name","")).strip().lower()
    dx, dy, area = enemy_features(e, W, H)
    center = max(0.0, 1.0 - abs(dx) / 0.5)

    near0 = float(getattr(core_cfg, "near_area0", 0.004))
    near1 = float(getattr(core_cfg, "near_area1", 0.030))
    near = np.clip((area - near0) / (near1 - near0 + 1e-9), 0.0, 1.0)

    # base por clase
    if name in {"cacodemon"}:
        base = 1.00
        exp  = 1.7
    elif name in {"marinechainsawvzd", "shotgunguy", "chaingunguy"}:
        base = 1.10          # 🔥 sube prioridad
        exp  = 1.10          # 🔥 no lo mates por ser "far" (bbox chico)
    elif name in {"demon"}:
        base = 0.75
        exp  = 1.7
    else:
        base = 0.60
        exp  = 1.7

    threat = base * (near ** exp) * (0.75 + 0.25 * center)
    return float(np.clip(threat, 0.0, 1.0))


# def pick_main_enemy(core, dets: List[Dict[str, Any]], W: int, H: int):
#     enemies = [d for d in dets if is_enemy(d.get("name",""))]
#     if not enemies:
#         return None

#     has_marine = any(str(e.get("name","")).lower() == "marinechainsawvzd" for e in enemies)

#     prev_name = getattr(core, "prev_enemy_name", None)
#     prev_cxcy = getattr(core, "prev_enemy_cxcy", None)
#     stickiness = float(getattr(core.cfg, "target_stickiness", 0.25))


#     best, best_score = None, -1e9
#     for d in enemies:
#         th = enemy_threat(core.cfg, d, W, H)
#         conf = float(d.get("conf", 1.0))
#         cx = (d["x"] + 0.5*d["w"]) / W
#         cy = (d["y"] + 0.5*d["h"]) / H
#         dx = cx - 0.5
#         aim = aim_main_from_dx(dx)
#         area = (d["w"]*d["h"]) / (W*H + 1e-9)

#         score = 1.20*th + 0.40*aim + 0.15*np.clip(area/0.03,0,1) + 0.10*conf
#         if prev_name and str(d.get("name","")).lower() == str(prev_name).lower():
#             score += stickiness
#         if prev_cxcy is not None:
#             pcx, pcy = prev_cxcy
#             dd = float(np.sqrt((cx-pcx)**2 + (cy-pcy)**2))
#             score += 0.20 * max(0.0, 1.0 - dd/0.20)

#         if score > best_score:
#             best_score = score
#             best = d

#     if best is not None:
#         cx = (best["x"] + 0.5*best["w"]) / W
#         cy = (best["y"] + 0.5*best["h"]) / H
#         core.prev_enemy_name = str(best.get("name","")).lower()
#         core.prev_enemy_cxcy = (cx, cy)

#     return best


def pick_main_enemy(core, dets: List[Dict[str, Any]], W: int, H: int):
    enemies = [d for d in dets if is_enemy(d.get("name", ""))]
    if not enemies:
        return None

    def _name(d):
        return str(d.get("name", "")).strip().lower()

    # ¿hay marines? ¿cuántos?
    marine_count = sum(1 for e in enemies if _name(e) == "marinechainsawvzd")
    has_marine = marine_count > 0

    # memoria para stickiness
    prev_name = getattr(core, "prev_enemy_name", None)
    prev_cxcy = getattr(core, "prev_enemy_cxcy", None)
    stickiness = float(getattr(core.cfg, "target_stickiness", 0.25))

    # ✅ knobs (tuneables desde EnvConfig)
    marine_target_bonus = float(getattr(core.cfg, "marine_target_bonus", 0.55))
    marine_crowd_bonus  = float(getattr(core.cfg, "marine_crowd_bonus", 0.20))
    marine_area_scale   = float(getattr(core.cfg, "marine_area_scale", 2.0))  # infla solo “área score”
    marine_stickiness_boost = float(getattr(core.cfg, "marine_stickiness_boost", 0.20))

    best, best_score = None, -1e9
    for d in enemies:
        name = _name(d)

        th = enemy_threat(core.cfg, d, W, H)
        conf = float(d.get("conf", 1.0))

        cx = (d["x"] + 0.5 * d["w"]) / W
        cy = (d["y"] + 0.5 * d["h"]) / H
        dx = cx - 0.5
        aim = aim_main_from_dx(dx)

        area = (d["w"] * d["h"]) / (W * H + 1e-9)

        # ✅ reduce sesgo contra bbox delgado:
        # solo para marines, infla el área efectiva que entra al score
        area_eff = area
        if name == "marinechainsawvzd":
            area_eff = min(1.0, area * (marine_area_scale ** 2))

        # score base (igual estructura que tenías)
        score = (
            1.20 * th +
            0.40 * aim +
            0.12 * np.clip(area_eff / 0.03, 0, 1) +   # (0.15→0.12) baja un poco la dominancia del área
            0.10 * conf
        )

        # ✅ prioridad explícita a marines si hay marines presentes
        if has_marine and name == "marinechainsawvzd":
            # bonus base
            score += marine_target_bonus
            # bonus extra si hay varios marines (para que NO los ignore)
            score += marine_crowd_bonus * max(0, marine_count - 1)

        # stickiness (si ya estabas trackeando ese mismo target)
        if prev_name and name == str(prev_name).lower():
            score += stickiness
            # si el target previo era marine, hazlo más pegajoso
            if name == "marinechainsawvzd":
                score += marine_stickiness_boost

        # continuidad espacial
        if prev_cxcy is not None:
            pcx, pcy = prev_cxcy
            dd = float(np.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2))
            score += 0.20 * max(0.0, 1.0 - dd / 0.20)

        if score > best_score:
            best_score = score
            best = d

    if best is not None:
        cx = (best["x"] + 0.5 * best["w"]) / W
        cy = (best["y"] + 0.5 * best["h"]) / H
        core.prev_enemy_name = _name(best)
        core.prev_enemy_cxcy = (cx, cy)

    return best

    

def aim_best_from_enemies(enemies: List[Dict[str,Any]], W: int, H: int) -> float:
    best = 0.0
    for e in enemies:
        cx = (e["x"] + 0.5*e["w"]) / W
        cy = (e["y"] + 0.5*e["h"]) / H
        dx = cx - 0.5
        dy = cy - 0.5
        s = aim_main_from_dxdy(dx, dy)  # <- 2D
        if s > best:
            best = s
    return float(best)

def enemy_topk_features(enemies: List[Dict[str,Any]], W:int, H:int, K:int, core_cfg):
    # sort por threat
    enemies_sorted = sorted(enemies, key=lambda e: enemy_threat(core_cfg,e,W,H), reverse=True)
    top = enemies_sorted[:K]
    feats = []
    for i in range(K):
        if i < len(top):
            th = enemy_threat(core_cfg, top[i], W, H)
            dx, dy, _ = enemy_features(top[i], W, H)
            feats.extend([float(dx), float(dy), float(th)])
        else:
            feats.extend([0.0,0.0,0.0])
    return feats