"""Set of plate moving tasks."""
from abc import ABC

import numpy as np
from gymnasium import spaces
from pyquaternion import Quaternion

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH
from bigym.envs.props.holders import DishDrainer
from bigym.envs.props.kitchenware import Plate
from bigym.envs.props.tables import Table
from bigym.utils.physics_utils import distance


RACK_BOUNDS = np.array([0.05, 0.05, 0])
RACK_POSITION_LEFT = np.array([0.7, 0.3, 0.95])
RACK_POSITION_RIGHT = np.array([0.7, -0.3, 0.95])

PLATE_OFFSET_POS = np.array([0, 0.01, 0])
PLATE_OFFSET_ROT = Quaternion(axis=[1, 0, 0], degrees=-5).elements


class _MovePlatesEnv(BiGymEnv, ABC):
    """Base plates environment."""

    _PRESET_PATH = PRESETS_PATH / "move_plates.yaml"

    _SUCCESSFUL_DIST = 0.05
    _SUCCESS_ROT = np.deg2rad(20)

    _PLATES_COUNT = 1

    def _initialize_env(self):
        self.table = self._preset.get_props(Table)[0]
        self.rack_start = self._preset.get_props(DishDrainer)[0]
        self.rack_target = self._preset.get_props(DishDrainer)[1]
        self.plates = [Plate(self._mojo) for _ in range(self._PLATES_COUNT)]

    def _success(self) -> bool:
        up = np.array([0, 0, 1])
        right = np.array([0, -1, 0])
        for plate in self.plates:
            if np.all(
                [
                    distance(plate.body, site) > self._SUCCESSFUL_DIST
                    for site in self.rack_target.sites
                ]
            ):
                return False
            plate_up = Quaternion(plate.body.get_quaternion()).rotate(up)
            angle = np.arccos(np.clip(np.dot(plate_up, right), -1.0, 1.0))
            if angle > self._SUCCESS_ROT:
                return False
            if not plate.is_colliding(self.rack_target):
                return False
            if plate.is_colliding(self.table):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(plate, side):
                    return False
        return True

    def _fail(self) -> bool:
        if super()._fail():
            return True
        for plate in self.plates:
            if plate.is_colliding(self.floor):
                return True
        return False

    def _on_reset(self):
        offset = np.random.uniform(-RACK_BOUNDS, RACK_BOUNDS)
        self.rack_start.body.set_position(RACK_POSITION_LEFT + offset)
        offset = np.random.uniform(-RACK_BOUNDS, RACK_BOUNDS)
        self.rack_target.body.set_position(RACK_POSITION_RIGHT + offset)

        sites = np.array(self.rack_start.sites)
        sites = np.random.choice(sites, size=len(self.plates), replace=False)

        for site, plate in zip(sites, self.plates):
            plate.body.set_position(site.get_position() + PLATE_OFFSET_POS, True)
            quat = Quaternion(site.get_quaternion())
            quat *= PLATE_OFFSET_ROT
            plate.body.set_quaternion(quat.elements, True)


class MovePlate(_MovePlatesEnv):
    """Move one plate from one rack to another."""

    def _get_task_privileged_obs_space(self):
        return {
            "rack_pose": spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            ),
            "plate_pose": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
        }

    def _get_task_privileged_obs(self):
        return {
            "rack_pose": np.array(self.rack_target.get_pose(), np.float32).flatten(),
            "plate_pose": np.array(self.plates[0].get_pose(), np.float32).flatten(),
        }


class MoveTwoPlates(_MovePlatesEnv):
    """Move two plates from one rack to another."""

    _PLATES_COUNT = 2

    def _get_task_privileged_obs_space(self):
        return {}

    def _get_task_privileged_obs(self):
        return {}
