# doom_env/features/pickups.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

from doom_env.features.items import PICKUP_INFO, PICKUP_NAMES

KIND_TO_ID = {"health": 0, "armor": 1, "ammo": 2, "other": 3}

@dataclass
class Pickup:
    name: str
    kind_id: int
    value: float
    dx: float   # cx/W - 0.5  (≈[-0.5,0.5])
    dy: float   # cy/H - 0.5
    area: float # bbox_area/(W*H)
    score: float

def _clamp(x: float, a: float, b: float) -> float:
    return a if x < a else (b if x > b else x)

def extract_pickups_from_dets(dets: List[Dict], W: int, H: int) -> List[Pickup]:
    out: List[Pickup] = []
    if W <= 0 or H <= 0:
        return out

    for d in dets:
        name = str(d.get("name", ""))
        if name not in PICKUP_NAMES:
            continue

        x, y = float(d.get("x", 0.0)), float(d.get("y", 0.0))
        w, h = float(d.get("w", 0.0)), float(d.get("h", 0.0))
        if w <= 0 or h <= 0:
            continue

        cx = (x + 0.5 * w) / float(W)
        cy = (y + 0.5 * h) / float(H)
        dx = cx - 0.5
        dy = cy - 0.5
        area = (w * h) / float(W * H + 1e-9)
        area = _clamp(area, 0.0, 1.0)

        info = PICKUP_INFO.get(name, {"kind": "other", "value": 1})
        kind = str(info.get("kind", "other"))
        value = float(info.get("value", 1.0))
        kind_id = KIND_TO_ID.get(kind, KIND_TO_ID["other"])

        dist = math.sqrt(dx * dx + dy * dy)          # 0..~0.707
        closeness = 1.0 / (0.05 + dist)              # más cerca => más grande
        score = value * closeness * (0.7 + 0.3*area) # area ayuda un poco

        out.append(Pickup(
            name=name, kind_id=kind_id, value=value,
            dx=float(dx), dy=float(dy), area=float(area),
            score=float(score),
        ))

    out.sort(key=lambda p: p.score, reverse=True)
    return out

def pickups_topk_vector(picks: List[Pickup], K: int) -> Tuple[List[float], Optional[float], int]:
    """
    Vector fijo:
      [has_pickup,
       dx, dy, area, kind_id, value] * K
    y devuelve (vec, best_dist, count)
    """
    vec: List[float] = []
    count = len(picks)
    has = 1.0 if count > 0 else 0.0
    vec.append(has)

    best_dist: Optional[float] = None
    if count > 0:
        p0 = picks[0]
        best_dist = float((p0.dx*p0.dx + p0.dy*p0.dy) ** 0.5)

    for i in range(K):
        if i < count:
            p = picks[i]
            # vec.extend([p.dx, p.dy, p.area, float(p.kind_id), float(p.value)])
            kind_norm = (float(p.kind_id) / 3.0) * 2.0 - 1.0   # 0..3 -> -1..1
            value_norm = float(p.value) / 100.0                # 25 -> 0.25, 50 -> 0.5, 100 -> 1.0 (clamp si quieres)
            if value_norm > 1.0:
                value_norm = 1.0
            vec.extend([p.dx, p.dy, p.area, kind_norm, value_norm])
        else:
            vec.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    return vec, best_dist, count