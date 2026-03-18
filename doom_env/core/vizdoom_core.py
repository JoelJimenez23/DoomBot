# doom_env/core/vizdoom_core.py
from __future__ import annotations
from typing import Any, Tuple, Dict
import vizdoom as vzd
import numpy as np

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from doom_env.core.state import EnvConfig, StepOutput
from doom_env.core.action_space import actions_family_a, actions_family_b ,ATTACK_IDS
from doom_env.features.detectors import SimpleYOLO, get_detections_from_state

class ViZDoomCore:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.game = vzd.DoomGame()
        self.game.set_doom_scenario_path(self.cfg["scenario"]["doom_scenario_path"])
        self.game.set_doom_map(self.cfg["scenario"]["doom_map"])
        # self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)

        self.game.set_depth_buffer_enabled(False)
        self.game.set_labels_buffer_enabled(bool(self.cfg["render"]["use_labels"]))
        self.game.set_objects_info_enabled(False)
        self.game.set_sectors_info_enabled(False)
        self.game.set_automap_buffer_enabled(False)

        self.game.set_render_hud(True)
        self.game.set_render_weapon(True)
        self.game.set_window_visible(bool(self.cfg["render"]["visible_window"]))

        self.buttons = [
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT,
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.MOVE_FORWARD,
            vzd.Button.MOVE_BACKWARD,
            vzd.Button.ATTACK,
        ]
        for b in self.buttons:
            self.game.add_available_button(b)

        # game variables
        self.game.add_available_game_variable(vzd.GameVariable.HEALTH)
        self.game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
        self.game.add_available_game_variable(vzd.GameVariable.SELECTED_WEAPON)
        self.game.add_available_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
        self.game.add_available_game_variable(vzd.GameVariable.ARMOR)
        self.game.add_available_game_variable(vzd.GameVariable.AMMO0)
        self.game.add_available_game_variable(vzd.GameVariable.AMMO1)
        self.game.add_available_game_variable(vzd.GameVariable.AMMO2)
        self.game.add_available_game_variable(vzd.GameVariable.AMMO3)
        self.game.add_available_game_variable(vzd.GameVariable.AMMO4)


        self.game.set_episode_timeout(int(self.cfg["episode_timeout"]))
        self.game.init()

        # self.actions = actions_family_a()
        # self.n_actions = len(self.actions)

        self.yolo = None
        # if not bool(cfg.use_labels):
        #     if not cfg.yolo_model_path:
        #         raise ValueError("use_labels=False pero no diste yolo_model_path")
        #     self.yolo = SimpleYOLO(cfg.yolo_model_path, device=cfg.yolo_device, conf=cfg.yolo_conf)

        # stateful trackers
        self.prev_ammo = None
        self.prev_health = None
        self.prev_kills = None

        self.attack_cooldown = 0
        self.recent_damage_timer = 0

        self.last_info_core: Dict[str, float] = {}

        self.game.set_doom_skill(int(self.cfg["scenario"]["doom_skill"]))

        af = str(getattr(cfg, "action_family", "A")).upper()

        if af in ("B", "NAV"):
            self.actions = actions_family_b(nav_v2=bool(getattr(cfg, "nav_v2", False)))
        elif af in ("AB", "A+B", "MERGE"):
            from doom_env.core.action_space import actions_family_ab
            self.actions = actions_family_ab(nav_v2=bool(getattr(cfg, "nav_v2", False)))
        else:
            self.actions = actions_family_a()

        self.n_actions = len(self.actions)

        self._yolo = None
        self._yolo_last = []
        self._yolo_step = 0

        if not bool(self.cfg["render"]["use_labels"]):
            if not cfg.yolo_model_path:
                raise ValueError("use_labels=False pero yolo_model_path vacío")

            from ultralytics import YOLO
            self._yolo = YOLO(cfg.yolo_model_path)

            # opcional: fuse acelera un poco
            try:
                self._yolo.fuse()
            except Exception:
                pass

        


    def get_ammo(self, st): return self.get_vars(st)[4]
    def get_health(self, st): return self.get_vars(st)[0]

    def close(self):
        try:
            self.game.close()
        except Exception:
            pass

    def reset(self, seed=None):
        self.game.new_episode()
        st = self.game.get_state()

        self.prev_ammo = None
        self.prev_health = None
        self.prev_kills = None
        self.prev_armor = None
        self.attack_cooldown = 0
        self.recent_damage_timer = 0

        self.last_info_core = {}

        if st is not None:
            health, armor, kills, weapon, ammo = self.get_vars(st)
            self.prev_ammo = ammo
            self.prev_health = health
            self.prev_kills = kills
            self.prev_armor = armor

        # print("st.game_variables: ", st.game_variables)
        return st

    def get_vars(self, st):
        gv = st.game_variables

        # Orden EXACTO según los add_available_game_variable que tú tienes ahora:
        # 0 HEALTH
        # 1 KILLCOUNT
        # 2 SELECTED_WEAPON
        # 3 SELECTED_WEAPON_AMMO
        # 4 ARMOR
        # 5 AMMO0
        # 6 AMMO1
        # 7 AMMO2
        # 8 AMMO3
        # 9 AMMO4

        health   = float(gv[0])
        kills    = float(gv[1])
        weapon   = float(gv[2])
        sel_ammo = float(gv[3])
        armor    = float(gv[4])

        ammo_pools = [float(x) for x in gv[5:10]]
        ammo_total = sum(a for a in ammo_pools if a > 0)

        ammo = sel_ammo if sel_ammo >= 0 else ammo_total

        return health, armor, kills, weapon, ammo

    def get_detections(self, st, screen_rgb: np.ndarray):
        # LABELS mode (visible-only)
        if bool(self.cfg["render"]["use_labels"]):
            return self._get_visible_labels_only(st)

        # YOLO mode
        if self._yolo is None:
            return []

        self._yolo_step += 1

        every = int(getattr(self.cfg, "yolo_every_n", 3))      # ✅ YOLO cada 3 steps (default)
        imgsz = int(getattr(self.cfg, "yolo_imgsz", 256))      # ✅ más barato que 320
        conf  = float(getattr(self.cfg, "yolo_conf", 0.20))    # ✅ baja un poco si “pierde”
        iou   = float(getattr(self.cfg, "yolo_iou", 0.45))
        max_det = int(getattr(self.cfg, "yolo_max_det", 15))   # ✅ limita costo cuando hay muchos
        device = str(getattr(self.cfg, "yolo_device", "0"))    # "0" o "cpu"

        # cache: reutiliza detecciones entre frames
        if every > 1 and (self._yolo_step % every != 0):
            return self._yolo_last

        # RGB -> BGR (ultralytics suele aguantar RGB, pero así es más estándar)
        frame_bgr = screen_rgb[..., ::-1]

        # half SOLO si estás en GPU
        use_half = bool(getattr(self.cfg, "yolo_half", True)) and (device != "cpu")

        # Si conoces los IDs de clases, filtra por clases para acelerar muchísimo:
        # Ej: self.cfg.yolo_classes = [0,1,2]  # enemies, armor, medikit
        classes = getattr(self.cfg, "yolo_classes", None)

        results = self._yolo.predict(
            source=frame_bgr,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            half=use_half,
            classes=classes,
            verbose=False,
        )

        dets = []
        r0 = results[0]
        names = r0.names  # dict id->name

        if r0.boxes is not None and len(r0.boxes) > 0:
            xyxy = r0.boxes.xyxy.cpu().numpy()
            confs = r0.boxes.conf.cpu().numpy()
            clss  = r0.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                dets.append({
                    "name": str(names.get(int(k), int(k))),
                    "x": int(x1), "y": int(y1),
                    "w": int(x2 - x1), "h": int(y2 - y1),
                    "conf": float(c),
                    "source": "yolo",
                })

        self._yolo_last = dets
        return dets


    def _get_visible_labels_only(self, st):
        """
        Extrae detecciones SOLO desde st.labels.
        Esto evita detectar enemigos detrás de paredes.
        """
        detections = []

        if st is None:
            return detections

        labels = getattr(st, "labels", None)
        if not labels:
            return detections

        for lb in labels:
            name = getattr(lb, "object_name", None) or getattr(lb, "name", "")
            name = str(name)

            # bounding box (compatibilidad varias versiones)
            x = getattr(lb, "x", None)
            y = getattr(lb, "y", None)
            w = getattr(lb, "width", None)
            h = getattr(lb, "height", None)

            if x is None:
                bb = getattr(lb, "bounding_box", None)
                if bb is not None:
                    x = getattr(bb, "x", None)
                    y = getattr(bb, "y", None)
                    w = getattr(bb, "width", None) or getattr(bb, "w", None)
                    h = getattr(bb, "height", None) or getattr(bb, "h", None)

            if None in (x, y, w, h):
                continue

            if w <= 1 or h <= 1:
                continue

            detections.append({
                "name": name,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "conf": 1.0
            })

        return detections


    def apply_action(self, action_id: int) -> float:
        step_tics = max(1, int(self.cfg["step_tics"]))
        fire_tics = max(1, min(int(self.cfg["fire_tics"]), step_tics))
        action = self.actions[action_id]

        if action_id in ATTACK_IDS:
            r = float(self.game.make_action(action, fire_tics))
            if step_tics > fire_tics and bool(self.cfg["noop_after_fire"]):
                r += float(self.game.make_action([False]*len(action), step_tics - fire_tics))
            return r

        return float(self.game.make_action(action, step_tics))

    def _update_deltas(self, ammo, health, armor, kills):
        if self.prev_ammo is None: self.prev_ammo = ammo
        if self.prev_health is None: self.prev_health = health
        if self.prev_armor is None: self.prev_armor = armor
        if self.prev_kills is None: self.prev_kills = kills

        d_ammo   = ammo - self.prev_ammo
        d_health = health - self.prev_health
        d_armor  = armor - self.prev_armor
        d_kills  = kills - self.prev_kills

        self.prev_ammo, self.prev_health, self.prev_armor, self.prev_kills = ammo, health, armor, kills
        return d_ammo, d_health, d_armor, d_kills

    def step(self, action_id: int) -> StepOutput:
        # tick cooldown
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

        base_r = self.apply_action(action_id)

        if self.game.is_episode_finished():
            info_core = {"done": True}
            # añade últimos vars conocidos si existen
            if self.last_info_core:
                info_core.update(self.last_info_core)
            return StepOutput(state=None, base_reward=base_r, terminated=True, truncated=False, info_core=info_core)

        st = self.game.get_state()
        if st is None:
            return StepOutput(state=None, base_reward=base_r, terminated=False, truncated=False, info_core={"warning":"state_none"})

        
        health, armor, kills, weapon, ammo = self.get_vars(st)
        d_ammo, d_health, d_armor, d_kills = self._update_deltas(ammo, health, armor, kills)

        if int(weapon) in (5,6,7):
            print("WEAPON", weapon, "AMMO", ammo, "RAW", list(st.game_variables))
        # danger timer
        if d_health < 0:
            self.recent_damage_timer = int(getattr(self.cfg, "recent_damage_window", 10))
        else:
            self.recent_damage_timer = max(0, self.recent_damage_timer - 1)

        info_core = {
            "ammo": ammo, "health": health, "armor": armor, "kills": kills,
            "d_ammo": float(d_ammo), "d_health": float(d_health), "d_armor": float(d_armor), "d_kills": float(d_kills),
            "danger": float(self.recent_damage_timer / max(1, int(getattr(self.cfg, "recent_damage_window", 10)))),
        }
        info_core["weapon"] = float(weapon)
        
        self.last_info_core = dict(info_core)
        return StepOutput(
            state=st, 
            base_reward=base_r,
            terminated=False, 
            truncated=False,
            d_ammo=float(d_ammo), 
            d_health=float(d_health), 
            d_armor=float(d_armor),
            d_kills=float(d_kills),
            info_core=info_core
        )