"""Manipulation tasks."""
from abc import ABC

import numpy as np
from mojo.elements import Body, Geom
from mojo.elements.consts import GeomType
from pyquaternion import Quaternion

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH
from bigym.envs.props.cabintets import BaseCabinet
from bigym.envs.props.cutlery import Spoon
from bigym.envs.props.items import Cube
from bigym.envs.props.kitchenware import Mug
from bigym.robots.configs.h1 import H1FineManipulation
from bigym.utils.env_utils import get_random_points_on_plane


class _ManipulationEnv(BiGymEnv, ABC):
    """Base manipulation environment."""

    _PRESET_PATH = PRESETS_PATH / "cabinet.yaml"

    def _initialize_env(self):
        self.cabinet = self._preset.get_props(BaseCabinet)[0]


class FlipCup(_ManipulationEnv):
    """Flip cup upside-up task."""

    _CUP_POS = np.array([0.8, 0, 1])
    _CUP_ROT_X = np.deg2rad(180)
    _CUP_ROT_Z = np.deg2rad(180)
    _CUP_STEP = 0.1
    _CUP_POS_EXTENTS = np.array([0.1, 0.25])
    _CUP_POS_BOUNDS = np.array([0.03, 0.03, 0])
    _CUP_ROT_BOUNDS = np.deg2rad(30)

    _TOLERANCE = np.deg2rad(5)

    def _initialize_env(self):
        super()._initialize_env()
        self.cup = Mug(self._mojo)

    def _success(self) -> bool:
        up = np.array([0, 0, 1])
        cup_up = Quaternion(self.cup.body.get_quaternion()).rotate(up)
        angle_to_up = np.arccos(np.clip(np.dot(cup_up, up), -1.0, 1.0))
        if angle_to_up > self._TOLERANCE:
            return False
        if not self.cup.is_colliding(self.cabinet.counter):
            return False
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.cup, side):
                return False
        return True

    def _on_reset(self):
        spawn_point = get_random_points_on_plane(
            1,
            self._CUP_POS,
            self._CUP_POS_EXTENTS,
            self._CUP_STEP,
            self._CUP_POS_BOUNDS,
        )[0]
        self.cup.body.set_position(spawn_point, True)
        quat = Quaternion(axis=[1, 0, 0], angle=self._CUP_ROT_X)
        angle = np.random.uniform(-self._CUP_ROT_BOUNDS, self._CUP_ROT_BOUNDS)
        quat *= Quaternion(axis=[0, 0, 1], angle=self._CUP_ROT_Z + angle)
        self.cup.body.set_quaternion(quat.elements, True)


class FlipCutlery(_ManipulationEnv):
    """Flip cutlery item task."""

    DEFAULT_ROBOT = H1FineManipulation

    _CUP_POS = np.array([0.8, 0, 0.86])
    _CUP_ROT_Z = np.deg2rad(180)
    _CUP_STEP = 0.1
    _CUP_POS_EXTENTS = np.array([0.1, 0.25])
    _CUP_POS_BOUNDS = np.array([0.03, 0.03, 0])
    _CUP_ROT_BOUNDS = np.deg2rad(180)

    _SPOON_OFFSET = np.array([0, 0, 0.15])
    _SPOON_QUAT = Quaternion(axis=[1, 0, 0], degrees=90)

    _TOLERANCE = np.deg2rad(50)

    def _initialize_env(self):
        super()._initialize_env()
        self.cup = Mug(self._mojo, kinematic=False)
        self.spoon = Spoon(self._mojo)

    def _success(self) -> bool:
        down = np.array([0, 0, -1])
        fwd = np.array([0, 1, 0])
        spoon_fwd = Quaternion(self.spoon.body.get_quaternion()).rotate(fwd)
        angle_to_up = np.arccos(np.clip(np.dot(spoon_fwd, down), -1.0, 1.0))
        if angle_to_up > self._TOLERANCE:
            return False
        if not self.spoon.is_colliding(self.cup):
            return False
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.cup, side):
                return False
        return True

    def _on_reset(self):
        spawn_point = get_random_points_on_plane(
            1,
            self._CUP_POS,
            self._CUP_POS_EXTENTS,
            self._CUP_STEP,
            self._CUP_POS_BOUNDS,
        )[0]
        self.cup.body.set_position(spawn_point)
        angle = np.random.uniform(-self._CUP_ROT_BOUNDS, self._CUP_ROT_BOUNDS)
        quat = Quaternion(axis=[0, 0, 1], angle=self._CUP_ROT_Z + angle)
        self.cup.body.set_quaternion(quat.elements)

        self.spoon.body.set_position(spawn_point + self._SPOON_OFFSET, True)
        self.spoon.body.set_quaternion(self._SPOON_QUAT.elements, True)


class StackBlocks(BiGymEnv):
    """Stack blocks in the correct area of the table."""

    _PRESET_PATH = PRESETS_PATH / "stack_blocks.yaml"

    _NUM_BLOCKS = 3
    _BLOCKS_POS = np.array([0.7, 0, 1])
    _BLOCKS_POS_EXTENTS = np.array([0.2, 0.5])
    _BLOCKS_STEP = 0.15
    _BLOCKS_POS_BOUNDS = np.array([0.05, 0.05, 0])
    _BLOCKS_ROT_BOUNDS = np.deg2rad([0, 0, 180])

    _TARGET_SIZE = np.array([0.05, 0.05, 0.001])
    _TARGET_COLOR = np.array([0.3, 0.8, 0.3, 1.0])
    _TARGET_POS = np.array([1.4, 0, 0.95])
    _TARGET_POS_BOUNDS = np.array([0.05, 0.2, 0.0])

    def _initialize_env(self):
        self.blocks = [Cube(self._mojo) for _ in range(self._NUM_BLOCKS)]
        self.target: Body = Body.create(self._mojo)
        self.target_collider = Geom.create(
            self._mojo,
            parent=self.target,
            geom_type=GeomType.BOX,
            size=self._TARGET_SIZE,
            color=self._TARGET_COLOR,
        )

    def _on_reset(self):
        points = get_random_points_on_plane(
            len(self.blocks),
            self._BLOCKS_POS,
            self._BLOCKS_POS_EXTENTS,
            self._BLOCKS_STEP,
        )
        for block, point in zip(self.blocks, points):
            block.set_pose(
                point,
                position_bounds=self._BLOCKS_POS_BOUNDS,
                rotation_bounds=self._BLOCKS_ROT_BOUNDS,
            )
        offset = np.random.uniform(-self._TARGET_POS_BOUNDS, self._TARGET_POS_BOUNDS)
        self.target.set_position(self._TARGET_POS + offset)

    def _success(self) -> bool:
        blocks_sorted = sorted(self.blocks, key=lambda b: b.body.get_position()[2])
        if not blocks_sorted[0].is_colliding(self.target_collider):
            return False
        if not blocks_sorted[1].is_colliding(blocks_sorted[0]):
            return False
        if not blocks_sorted[2].is_colliding(blocks_sorted[1]):
            return False
        for block in self.blocks:
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(block, side):
                    return False
        return True

    def _fail(self) -> bool:
        if super()._fail():
            return True
        for block in self.blocks:
            if block.is_colliding(self.floor):
                return True
        return False
