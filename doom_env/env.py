#doom_env/env.py
from __future__ import annotations
from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


from doom_env.core.state import EnvConfig
from doom_env.tasks.base_task import BaseTask
from doom_env.core.vizdoom_core import ViZDoomCore
from doom_env.core.action_space import ATTACK_IDS, FWD_IDS, TURN_IDS, STRAFE_IDS

import cv2
import time

class DoomTaskEnv(gym.Env):
    def __init__(self,cfg: EnvConfig,task: BaseTask):
        super().__init__()
        self.cfg = cfg
        self.core = ViZDoomCore(cfg)
        self.task = task

        self.action_space = spaces.Discrete(self.core.n_actions)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.task.obs_dim(cfg),),
            dtype=np.float32,
        )
        self._last_obs = None
        # -----------------------------
        # Exploración visual (audit-safe)
        # -----------------------------
        self._prev_gray_small = None   # frame gris reducido del paso anterior
        self._visits = {}              # hash -> conteo de visitas
        self._stuck_counter = 0
        self._unstuck_steps_left = 0
        self._unstuck_mode = 0         # 0: turn, 1: strafe (alternado)


    def close(self):
        if getattr(self.cfg, "debug_draw", False):
            cv2.destroyAllWindows()
        self.core.close()
    
    def reset(self,seed=None,option=None):
        super().reset(seed=seed)
        st = self.core.reset(seed=seed)
        self.core.prev_aim_best = 0.0

        obs, info = self.task.make_obs_and_info(self.core, st)
        if getattr(self.cfg, "debug_draw", False) and st is not None:
            dets = self.core.get_detections(st, st.screen_buffer)
            self._debug_draw(st, dets, main_enemy=None)

        self._prev_gray_small = None
        self._visits.clear()
        self._stuck_counter = 0
        self._unstuck_steps_left = 0
        self._unstuck_mode = 0

        self._last_obs = obs
        return obs, info
    
    def step(self, action_id: int):
        action_id = int(action_id)

        # 1) override (auto-fire + cooldown block)
        new_action_id, override_info = self.task.action_override(self.core, action_id)
        new_action_id = int(new_action_id)

        # 1.5) override anti-stuck (solo por visión, sin API interno)
        # Si estamos en modo 'despegue', forzamos giro/strafe unos pasos.
        if self._unstuck_steps_left > 0:
            self._unstuck_steps_left -= 1
            # alterna entre girar y strafe para salir de esquinas
            if self._unstuck_mode == 0:
                new_action_id = 2  # turn_right (familia A)
            else:
                new_action_id = 4  # strafe_right (familia A)
            self._unstuck_mode = 1 - self._unstuck_mode

        # 2) step core
        step_out = self.core.step(new_action_id)

        # 3) info base
        info: Dict[str, Any] = {}
        info.update(override_info)
        if step_out.info_core:
            info.update(step_out.info_core)

        # 4) shot_event (esto es CLAVE para tu reward)
        requested_attack = bool(info.get("requested_attack", new_action_id in ATTACK_IDS))
        fired = (new_action_id in ATTACK_IDS)
        ammo_spent = (step_out.d_ammo < 0.0)

        info["action_id"] = int(new_action_id)
        info["fired"] = bool(fired)
        info["ammo_spent"] = bool(ammo_spent)
        info["shot_event"] = bool(ammo_spent)
        info["requested_attack"] = bool(requested_attack)

        # 5) aplica cooldown si realmente disparó (y no estaba bloqueado)
        if ammo_spent and not bool(info.get("blocked_by_cooldown", False)):
            cd = int(getattr(self.cfg, "attack_cooldown_steps", 0))
            if cd > 0:
                self.core.attack_cooldown = cd

        # 6) obs/info si hay state
        if step_out.state is None:
            obs = self.task.zero_obs(self.cfg)
            rew = float(step_out.base_reward)
            terminated = bool(step_out.terminated)
            truncated = bool(step_out.truncated)

            if hasattr(self, "_last_task_info"):
                info.update(self._last_task_info)

            return obs, rew, terminated, truncated, info

        if getattr(self.cfg, "debug_draw", False):
            st = step_out.state
            dets = self.core.get_detections(st, st.screen_buffer)
            self._debug_draw(st, dets, main_enemy=None)

        obs, task_info = self.task.make_obs_and_info(self.core, step_out.state)
        info.update(task_info)
        self._last_task_info = dict(task_info)

        terminated = bool(step_out.terminated)
        truncated  = bool(step_out.truncated)

        # permitir que la task termine el episodio (victoria)
        if bool(info.get("task_done", False)):
            terminated = True

        # 7) reward final: base_reward + task_reward + intrinsic exploration
        task_r = float(self.task.compute_reward(self.core, step_out, info))
        expl_r = float(self._exploration_reward(step_out.state, int(new_action_id), info))
        rew = float(step_out.base_reward) + task_r + expl_r

        self._last_obs = obs
        return obs, rew, terminated, truncated, info

    # -----------------------------
    # Exploración visual (audit-safe)
    # -----------------------------
    def _downscale_gray(self, rgb: np.ndarray, size: Tuple[int, int] = (32, 24)) -> np.ndarray:
        """Convierte RGB->gris y lo reduce a size (w,h). Devuelve uint8."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        return small.astype(np.uint8)

    def _ahash(self, gray_small: np.ndarray) -> bytes:
        """Average-hash simple: produce bytes compactos para usar como key."""
        h = cv2.resize(gray_small, (16, 12), interpolation=cv2.INTER_AREA)
        m = int(h.mean())
        bits = (h > m).astype(np.uint8).reshape(-1)
        packed = np.packbits(bits)
        return packed.tobytes()

    def _mean_abs_diff(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))

    def _exploration_reward(self, st, action_id: int, info: Dict[str, Any]) -> float:
        """
        Reward intrínseco para exploración sin usar posiciones internas.
        - novelty por hash visual
        - castigo por quedarse 'stuck' (poco cambio visual)
        - castigo por 'muro' (forward sin cambio)
        - activa unstuck override si queda pegado muchos pasos
        """
        rgb = getattr(st, "screen_buffer", None)
        if rgb is None:
            return 0.0

        # hiperparámetros (ajústalos si quieres)
        r_new = float(getattr(self.cfg, "explore_new_reward", 0.02))
        r_rep = float(getattr(self.cfg, "explore_repeat_penalty", 0.002))
        eps = float(getattr(self.cfg, "stuck_diff_eps", 2.0))
        k = int(getattr(self.cfg, "stuck_k", 10))
        r_stuck = float(getattr(self.cfg, "stuck_penalty", 0.01))
        r_wall = float(getattr(self.cfg, "wall_penalty", 0.01))
        unstuck_steps = int(getattr(self.cfg, "unstuck_steps", 3))

        gray_small = self._downscale_gray(rgb, (32, 24))

        # 1) novelty hash
        key = self._ahash(gray_small)
        n = self._visits.get(key, 0) + 1
        self._visits[key] = n
        novelty = r_new if n == 1 else -r_rep * float(np.log1p(n))

        # 2) motion proxy (stuck)
        if self._prev_gray_small is None:
            diff = 999.0
        else:
            diff = self._mean_abs_diff(gray_small, self._prev_gray_small)

        is_static = (diff < eps)
        if is_static:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0

        stuck_pen = (-r_stuck) if self._stuck_counter >= k else 0.0

        # 3) wall proxy: forward sin cambio (solo si no hay combate)
        enemy_present = bool(info.get("enemy_present", False))
        wall_pen = 0.0
        if (not enemy_present) and (action_id in FWD_IDS) and is_static:
            wall_pen = -r_wall

        # 4) activa unstuck si está pegado y no hay enemigo
        if (not enemy_present) and (self._stuck_counter >= k) and (self._unstuck_steps_left <= 0):
            self._unstuck_steps_left = max(1, unstuck_steps)
            self._unstuck_mode = 0

        self._prev_gray_small = gray_small

        # si hay enemigo, baja el impacto para no molestar el aiming
        if enemy_present:
            novelty *= 0.3
            stuck_pen *= 0.3
            wall_pen = 0.0

        return float(novelty + stuck_pen + wall_pen)

    def _debug_draw(self, st, dets, main_enemy=None):
        if not getattr(self.cfg, "debug_draw", False):
            return
        if st is None:
            return

        img = st.screen_buffer  # RGB
        if img is None:
            return

        # OpenCV espera BGR
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # dibujar todas las detecciones
        for d in dets:
            x, y, w, h = int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"])
            name = str(d.get("name", ""))
            conf = d.get("conf", None)
            label = f"{name}" if conf is None else f"{name} {conf:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(frame, label, (x, max(0, y - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # resaltar main enemy si lo pasas
        if main_enemy is not None:
            x, y, w, h = int(main_enemy["x"]), int(main_enemy["y"]), int(main_enemy["w"]), int(main_enemy["h"])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # escala para ver mejor
        scale = int(getattr(self.cfg, "debug_scale", 2))
        if scale > 1:
            frame = cv2.resize(frame, (frame.shape[1]*scale, frame.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("debug", frame)
        cv2.waitKey(1)