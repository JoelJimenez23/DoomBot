# doom_env/core/state.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# Sub-configs alineadas con el YAML
# ============================================================

@dataclass
class ScenarioConfig:
    doom_scenario_path: str = "scenarios/defend_the_center.wad"
    doom_map: str = "map01"
    doom_skill: int = 3
    episode_timeout_tics: int = 0
    episode_start_tics: int = 10
    seed: int = 0


@dataclass
class RenderConfig:
    visible_window: bool = True
    screen_resolution: str = "RES_320X240"
    screen_format: str = "RGB24"
    render_hud: bool = True
    render_crosshair: bool = True
    render_decals: bool = True
    render_particles: bool = True

    automap_buffer_enabled: bool = False
    depth_buffer_enabled: bool = True
    labels_buffer_enabled: bool = True

    audio_enabled: bool = False


@dataclass
class ControlsConfig:
    buttons: List[str] = field(default_factory=lambda: [
        "MOVE_FORWARD",
        "MOVE_BACKWARD",
        "TURN_LEFT",
        "TURN_RIGHT",
        "ATTACK",
        "USE",
        "SPEED",
        "SELECT_WEAPON1",
        "SELECT_WEAPON2",
        "SELECT_WEAPON3",
        "SELECT_WEAPON4",
        "SELECT_WEAPON5",
        "SELECT_WEAPON6",
        "SELECT_WEAPON7",
    ])

    game_variables: List[str] = field(default_factory=lambda: [
        "HEALTH",
        "ARMOR",
        "KILLCOUNT",
        "SELECTED_WEAPON",
        "SELECTED_WEAPON_AMMO",
        "AMMO1",
        "AMMO2",
        "AMMO3",
        "AMMO4",
        "WEAPON1",
        "WEAPON2",
        "WEAPON3",
        "WEAPON4",
        "WEAPON5",
        "WEAPON6",
        "WEAPON7",
    ])


@dataclass
class TimingConfig:
    frame_skip: int = 1
    realtime_lock: bool = True
    target_hz: float = 35.0


@dataclass
class RewardConfig:
    living_reward: float = 0.0
    death_penalty: float = 0.0


@dataclass
class RecordingConfig:
    enable_lmp: bool = False
    output_dir: str = "recordings"
    dump_obs_every_k_tics: int = 1
    enable_png_frames: bool = False

    video_backend: str = "ffmpeg"
    video_container: str = "mkv"
    video_codec: str = "libx264"
    video_crf: int = 4
    video_preset: str = "veryfast"
    video_fps: float = 35.0

    chunk_size: int = 350
    queue_maxsize: int = 256


@dataclass
class SafetyConfig:
    max_episode_seconds: int = 0
    action_timeout_ms: int = 0


# ============================================================
# Extras propios de tu proyecto RL / YOLO / Task logic
# ============================================================

@dataclass
class YoloConfig:
    enabled: bool = False
    model_path: str = ""
    device: str = "0"
    conf: float = 0.20
    iou: float = 0.45
    imgsz: int = 256
    every_n: int = 3
    half: bool = True
    max_det: int = 15


@dataclass
class DebugConfig:
    debug_draw: bool = False
    debug_window_name: str = "doom_dbg"
    debug_scale: int = 3
    debug_show_fps: bool = True


@dataclass
class CombatConfig:
    step_tics: int = 2
    fire_tics: int = 1
    noop_after_fire: bool = True

    topk_enemies: int = 3
    recent_damage_window: int = 10
    attack_cooldown_steps: int = 0

    auto_fire: bool = True
    fire_aim_far: float = 0.55
    fire_aim_near: float = 0.35
    fire_area_min: float = 0.012

    aim_deadzone: float = 0.02
    turn_dx_th: float = 0.06
    aim_y_frac: float = 0.35

    near_area0: float = 0.004
    near_area1: float = 0.030
    target_stickiness: float = 0.25


@dataclass
class NavigationConfig:
    action_family: str = "A"   # "A" o "B"
    nav_v2: bool = False


@dataclass
class RewardShapingConfig:
    rw_hit_bonus: float = 0.25
    rw_miss_penalty: float = 0.06
    rw_ammo_penalty: float = 0.01
    rw_fire_no_enemy_penalty: float = 0.03


# ============================================================
# Config principal
# ============================================================

@dataclass
class EnvConfig:
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    controls: ControlsConfig = field(default_factory=ControlsConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    # extras de tu proyecto
    yolo: YoloConfig = field(default_factory=YoloConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    combat: CombatConfig = field(default_factory=CombatConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    reward_shaping: RewardShapingConfig = field(default_factory=RewardShapingConfig)

    # --------------------------------------------------------
    # Compatibilidad con tu código viejo
    # --------------------------------------------------------
    @property
    def scenario_wad(self) -> str:
        return self.scenario.doom_scenario_path

    @property
    def map_name(self) -> str:
        return self.scenario.doom_map

    @property
    def doom_skill(self) -> int:
        return self.scenario.doom_skill

    @property
    def episode_timeout(self) -> int:
        return self.scenario.episode_timeout_tics

    @property
    def render_enabled(self) -> bool:
        return self.render.visible_window

    @property
    def use_labels(self) -> bool:
        return self.render.labels_buffer_enabled

    @property
    def button_names(self) -> List[str]:
        return self.controls.buttons

    @property
    def game_variable_names(self) -> List[str]:
        return self.controls.game_variables


# ============================================================
# Salida de step
# ============================================================

@dataclass
class StepOutput:
    state: Any
    base_reward: float
    terminated: bool
    truncated: bool

    d_ammo: float = 0.0
    d_health: float = 0.0
    d_armor: float = 0.0
    d_kills: float = 0.0

    info_core: Dict[str, Any] = field(default_factory=dict)