# eval/manual_evaluation.py

from __future__ import annotations
import time
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Resolver rutas del proyecto
# ------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
UTILS_DIR = PROJECT_ROOT / "utils"

sys.path.append(str(UTILS_DIR))

# ------------------------------------------------------------------
# Imports del proyecto
# ------------------------------------------------------------------

from keys import GlobalKeyState, load_yaml, build_action_vector, ask_to_start
from doom_controller import DoomController

def main():
    print("\n=== DOOM ===")
    if not ask_to_start():
        print("Saliendo.")
        return

    # --------------------------------------------------------------
    # Cargar archivos desde utils/
    # --------------------------------------------------------------
    keymap_path = UTILS_DIR / "keymap.yaml"
    config_path = UTILS_DIR / "game_config.yaml"

    # Cargar configuraciones con manejo de error
    try:
        game_cfg = load_yaml(str(config_path))  # (opcional) útil para validar que existe y es YAML válido
    except Exception as e:
        print(f"ERROR al cargar config de juego: {e}")
        sys.exit(1)

    try:
        keymap = load_yaml(str(keymap_path))
    except Exception as e:
        print(f"ERROR al cargar keymap: {e}")
        sys.exit(1)

    # --------------------------------------------------------------
    # Iniciar controlador
    # --------------------------------------------------------------
    try:
        controller = DoomController(str(config_path))
    except Exception as e:
        print(f"ERROR al crear DoomController con config '{config_path}':\n{e}")
        sys.exit(1)

    # --------------------------------------------------------------
    # Captura de teclado
    # --------------------------------------------------------------
    keys = GlobalKeyState()
    keys.start()
    print("Controles activos. Pulsa ESC para salir.")

    # --------------------------------------------------------------
    # Reset y metadatos
    # --------------------------------------------------------------
    obs = controller.reset()

    # Soportar ambos nombres por si tu DoomController varía
    button_names = getattr(controller, "button_names", None) or getattr(controller, "_button_names", None)
    gv_names = getattr(controller, "game_variable_names", None) or getattr(controller, "_game_variable_names", None)

    if button_names is None:
        # fallback: por si tu clase tiene método
        if hasattr(controller, "get_button_names"):
            button_names = controller.get_button_names()
        else:
            raise AttributeError("No encuentro button_names/_button_names ni get_button_names() en DoomController")

    if gv_names is None:
        gv_names = []

    # --------------------------------------------------------------
    # Bucle a 35 Hz con sincronización real
    # --------------------------------------------------------------
    target_hz = 35.0
    period = 1.0 / target_hz
    next_t = time.perf_counter()
    t_index = 0
    cumulative_reward = 0.0

    terminal_reason: Optional[str] = None

    try:
        while True:
            if keys.exit_requested():
                terminal_reason = "user"
                break

            pressed = keys.snapshot()
            action_vec = build_action_vector(button_names, keymap, pressed)

            # Intentar usar la firma "step(action, repeat=1)" si existe;
            # si no existe, caer a step(action) sin reward/terminated/truncated.
            try:
                obs, r, terminated, truncated, info = controller.step(action_vec, repeat=1)
                cumulative_reward += float(r)

                if terminated or truncated:
                    if truncated:
                        terminal_reason = "timeout"
                    else:
                        # intentar detectar muerte del jugador
                        is_dead = False
                        try:
                            game = getattr(controller, "game", None)
                            if game is not None and hasattr(game, "is_player_dead"):
                                is_dead = bool(game.is_player_dead())
                        except Exception:
                            is_dead = False

                        if is_dead:
                            terminal_reason = "death"
                        else:
                            # heurística por HEALTH si existe
                            health_val = None
                            gv = None
                            if isinstance(obs, dict):
                                gv = obs.get("gamevariables", None)

                            if isinstance(gv, np.ndarray) and ("HEALTH" in gv_names):
                                try:
                                    health_val = float(gv[gv_names.index("HEALTH")])
                                except Exception:
                                    health_val = None

                            terminal_reason = "death" if (health_val is not None and health_val <= 0) else "success"
                    break

            except TypeError:
                # Firma vieja/simple: solo step(action)
                controller.step(action_vec)

            t_index += 1

            # Sincronizar a 35 Hz
            next_t += period
            now = time.perf_counter()
            delay = next_t - now
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = now

    except KeyboardInterrupt:
        terminal_reason = "user"

    finally:
        msg = {
            "user": "Sesión terminada por el usuario.",
            "death": "Perdiste! :c (muerte del jugador).",
            "success": "Ganaste! :D (Objetivo alcanzado).",
            "timeout": "Sesión terminada por límite de tiempo.",
            None: "Sesión finalizada por X motivo.",
        }.get(terminal_reason, "Sesión finalizada.")

        print("\n" + "=" * 60)
        print(f"Motivo de cierre: {terminal_reason}")
        print(msg)
        print(f"Reward acumulado: {cumulative_reward:.3f}")
        print(f"Steps: {t_index}")
        print("=" * 60 + "\n")

        time.sleep(3.0)

        keys.stop()
        controller.close()


if __name__ == "__main__":
    main()
