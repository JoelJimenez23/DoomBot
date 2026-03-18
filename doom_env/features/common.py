# doom_env/features/common.py
import re

ENEMY_NAMES = {
    "zombieman","shotgunguy","doomimp","cacodemon","demon",
    "zombie","formerhuman","former","formersergeant","shotguy","chaingunguy",
    "marinechainsawvzd","marine",
}

IGNORE_NAMES = {
    "doomplayer","bulletpuff","blood","rocketsmoketrail",
    "deadcacodemon","deadzombieman",
    # proyectiles (en A los ignoras; en C los vas a usar)
    "doomimpball"
}

YOLO_ID_TO_NAME = {
    0:"Cacodemon",1:"DoomPlayer",2:"CustomRocket",3:"RocketSmokeTrail",4:"Medikit",
    5:"BulletPuff",6:"DeadCacodemon",7:"Blood",8:"GreenArmor",9:"Zombieman",
    10:"ShotgunGuy",11:"DeadZombieman",12:"Clip",13:"DoomImp",14:"DoomImpBall",
    15:"Rocket",16:"CustomMedikit",17:"MarineChainsawVzd",18:"Demon",
}

_RAW_PICKUP_INFO = {
    "Stimpack":      {"kind": "health", "value": 10},
    "Medikit":       {"kind": "health", "value": 25},
    "HealthBonus":   {"kind": "health", "value": 1},
    "GreenArmor":    {"kind": "armor",  "value": 100},
    "ArmorBonus":    {"kind": "armor",  "value": 1},
    "Clip":          {"kind": "ammo",   "value": 10},
    "AmmoBox":       {"kind": "ammo",   "value": 50},
    "Shells":        {"kind": "ammo",   "value": 4},
    "ShellBox":      {"kind": "ammo",   "value": 20},
}



def norm_name(name: str) -> str:
    if not name:
        return ""
    s = name.strip().replace("-", "_")
    s = re.sub(r"\s+", "", s)
    return s.lower()

def is_enemy(name: str) -> bool:
    return norm_name(name) in ENEMY_NAMES

def is_ignored(name: str) -> bool:
    return norm_name(name) in IGNORE_NAMES


PICKUP_INFO = {norm_name(k): v for k, v in _RAW_PICKUP_INFO.items()}
PICKUP_NAMES = set(PICKUP_INFO.keys())