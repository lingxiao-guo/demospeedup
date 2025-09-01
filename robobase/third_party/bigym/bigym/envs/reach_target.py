"""Set of reach target tasks."""
from abc import ABC
from dataclasses import dataclass, field

import numpy as np
from gymnasium import spaces
from mojo import Mojo
from mojo.elements import Body, Geom
from mojo.elements.consts import GeomType

from bigym.bigym_env import BiGymEnv
from bigym.const import HandSide
from bigym.robots.robot import Robot


@dataclass
class TargetConfig:
    """Target Config."""

    target_hands: list[HandSide]
    reset_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    size: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.05]))
    color_default: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 1]))
    color_highlight: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 1]))


class Target:
    """Target sphere."""

    def __init__(self, mojo: Mojo, robot: Robot, config: TargetConfig):
        """Init."""
        self._robot = robot
        self._config = config
        self.body = Body.create(mojo)
        self.geom: Geom = Geom.create(
            mojo,
            parent=self.body,
            geom_type=GeomType.SPHERE,
            size=config.size,
            color=config.color_default,
            mass=0,
        )
        self.geom.set_collidable(False)

    def reset_position(self, offset: np.ndarray = np.zeros(3)):
        """Reset position of the target."""
        self.body.set_position(self._config.reset_position + offset)

    def distance(self, pos: np.ndarray) -> float:
        """Get distance to target."""
        return float(np.linalg.norm(self.body.get_position() - pos))

    def is_reached(self, tolerance: float) -> bool:
        """Check if target is reached."""
        for side in self._config.target_hands:
            if side not in self._robot.grippers:
                continue
            hand_pos = self._robot.get_hand_pos(side)
            is_reached = self.distance(hand_pos) <= tolerance
            self.set_highlight(is_reached)
            if is_reached:
                return True
        return False

    def set_highlight(self, highlight: bool):
        """Toggle target highlight."""
        self.geom.set_color(
            self._config.color_highlight if highlight else self._config.color_default
        )


class _ReachTargetEnv(BiGymEnv, ABC):
    """Base reach target environment."""

    TARGET_CONFIGS = [
        TargetConfig(
            target_hands=[HandSide.LEFT, HandSide.RIGHT],
            reset_position=np.array([0.5, 0, 1]),
            color_default=np.array([0.3, 0, 0, 1]),
            color_highlight=np.array([1, 0, 0, 1]),
        )
    ]

    POSITION_BOUNDS = np.array([0.1, 0.1, 0.1])
    TOLERANCE = 0.1

    def _initialize_env(self):
        self.targets: list[Target] = []
        for config in self.TARGET_CONFIGS:
            self.targets.append(Target(self._mojo, self.robot, config))

    def _on_reset(self):
        for target in self.targets:
            offset = np.random.uniform(-self.POSITION_BOUNDS, self.POSITION_BOUNDS)
            target.reset_position(offset)
            target.set_highlight(False)

    def _success(self) -> bool:
        for target in self.targets:
            if not target.is_reached(self.TOLERANCE):
                return False
        return True

    def _on_step(self):
        """Highlight spheres even in fast mode."""
        self._success()


class ReachTarget(_ReachTargetEnv):
    """Reach the target with either left or right wrist."""

    def _get_task_privileged_obs_space(self):
        return {
            "target_position": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            )
        }

    def _get_task_privileged_obs(self):
        return {
            "target_position": np.array(
                self.targets[0].body.get_position(), np.float32
            ).copy()
        }


class ReachTargetSingle(ReachTarget):
    """Reach the target with specific wrist."""

    TARGET_CONFIGS = [
        TargetConfig(
            target_hands=[HandSide.LEFT],
            reset_position=np.array([0.5, 0, 1]),
            color_default=np.array([0.3, 0, 0, 1]),
            color_highlight=np.array([1, 0, 0, 1]),
        )
    ]


class ReachTargetDual(_ReachTargetEnv):
    """Reach 2 targets, one with each arm."""

    TARGET_CONFIGS = [
        TargetConfig(
            target_hands=[HandSide.LEFT],
            reset_position=np.array([0.5, 0.2, 1]),
            color_default=np.array([0.3, 0, 0, 1]),
            color_highlight=np.array([1, 0, 0, 1]),
        ),
        TargetConfig(
            target_hands=[HandSide.RIGHT],
            reset_position=np.array([0.5, -0.2, 1]),
            color_default=np.array([0, 0.3, 0, 1]),
            color_highlight=np.array([0, 1, 0, 1]),
        ),
    ]

    def _get_task_privileged_obs_space(self):
        return {
            "target_position_left": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
            "target_position_right": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
        }

    def _get_task_privileged_obs(self):
        return {
            "target_position_left": np.array(
                self.targets[0].body.get_position(), np.float32
            ).copy(),
            "target_position_right": np.array(
                self.targets[1].body.get_position(), np.float32
            ).copy(),
        }
