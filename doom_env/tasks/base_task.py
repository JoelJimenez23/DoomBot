# doom_env/tasks/base_task.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np
from doom_env.core.state import EnvConfig, StepOutput

class BaseTask(ABC):
    @abstractmethod
    def obs_dim(self, cfg: EnvConfig) -> int: ...

    def zero_obs(self, cfg: EnvConfig) -> np.ndarray:
        return np.zeros((self.obs_dim(cfg),), dtype=np.float32)

    @abstractmethod
    def make_obs_and_info(self, core, st) -> Tuple[np.ndarray, Dict[str, Any]]: ...

    @abstractmethod
    def compute_reward(self, core, step_out: StepOutput, info: Dict[str, Any]) -> float: ...

    def action_override(self, core, action_id: int):
        # return (action_id, extra_info)
        return int(action_id), {}