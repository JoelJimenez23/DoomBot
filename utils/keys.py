# keys.py
from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Set

import numpy as np
import yaml
from pynput import keyboard


# -----------------------------
# Utils YAML / FS
# -----------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encuentra el archivo YAML: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"El YAML debe ser un mapeo (dict): {path}")
    return data


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Teclado
# -----------------------------
KEY_ALIASES = {
    "ctrl_l": "CTRL",
    "ctrl_r": "CTRL",
    "ctrl": "CTRL",
    "shift_l": "SHIFT",
    "shift_r": "SHIFT",
    "shift": "SHIFT",
    "space": "SPACE",
    "esc": "ESCAPE",
    "enter": "ENTER",
    "return": "ENTER",
    "up": "UP",
    "down": "DOWN",
    "left": "LEFT",
    "right": "RIGHT",
}


def normalize_key(key: keyboard.Key | keyboard.KeyCode) -> Optional[str]:
    """
    Convierte lo que reporta pynput en un string consistente:
      - letras -> 'W'
      - espacio -> 'SPACE'
      - flechas -> 'UP', etc.
    """
    try:
        if isinstance(key, keyboard.KeyCode):
            ch = key.char
            if ch is None:
                return None
            if len(ch) == 1:
                return ch.upper() if ch.isalpha() else ch
            return ch.upper()
        else:
            name = str(key).split(".")[-1].lower()
            return KEY_ALIASES.get(name, name.upper())
    except Exception:
        return None


class GlobalKeyState:
    """
    Listener global para capturar teclas presionadas y un flag de salida (ESC).
    """
    def __init__(self) -> None:
        self._pressed: Set[str] = set()
        self._lock = threading.Lock()
        self._exit_requested = False
        self._listener: Optional[keyboard.Listener] = None

    def _on_press(self, key) -> None:
        k = normalize_key(key)
        if k is None:
            return
        with self._lock:
            self._pressed.add(k)
            if k == "ESCAPE":
                self._exit_requested = True

    def _on_release(self, key) -> None:
        k = normalize_key(key)
        if k is None:
            return
        with self._lock:
            self._pressed.discard(k)

    def start(self) -> None:
        if self._listener is not None:
            return
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def snapshot(self) -> Set[str]:
        with self._lock:
            return set(self._pressed)

    def exit_requested(self) -> bool:
        with self._lock:
            return bool(self._exit_requested)

    def clear_exit(self) -> None:
        with self._lock:
            self._exit_requested = False


# -----------------------------
# Mapping teclado -> botones
# -----------------------------
def build_action_vector(
    button_names: List[str],
    keymap: Dict[str, Any],
    pressed: Set[str],
) -> np.ndarray:
    """
    Convierte teclas presionadas (ej: {'W','A'}) a vector multi-binario
    alineado con `button_names` (ej: ['MOVE_FORWARD','MOVE_LEFT','ATTACK']).

    Espera que keymap tenga:
      keymap['keyboard'] = {'W': 'MOVE_FORWARD', 'A': 'MOVE_LEFT', ...}
    """
    K = len(button_names)
    vec = np.zeros((K,), dtype=np.int32)

    keyboard_map: Dict[str, str] = {k.upper(): v for k, v in (keymap.get("keyboard", {}) or {}).items()}

    # Para acelerar: nombre->index
    name_to_idx = {name: i for i, name in enumerate(button_names)}

    for key in pressed:
        btn = keyboard_map.get(key)
        if btn is None or btn == "QUIT":
            continue
        j = name_to_idx.get(btn)
        if j is not None:
            vec[j] = 1

    return vec


def ask_to_start() -> bool:
    try:
        ans = input("¿Deseas comenzar? [s/N]: ").strip().lower()
        return ans in ("s", "si", "sí", "y", "yes")
    except KeyboardInterrupt:
        return False
