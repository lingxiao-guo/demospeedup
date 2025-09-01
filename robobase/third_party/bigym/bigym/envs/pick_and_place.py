"""Pick and place tasks."""

import numpy as np
from pyquaternion import Quaternion

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH
from bigym.envs.props.cabintets import BaseCabinet, WallCabinet
from bigym.envs.props.cutlery import Spatula
from bigym.envs.props.items import Box, Sandwich
from bigym.envs.props.prop import Prop
from bigym.envs.props.kitchenware import Mug, Saucepan, Pan, ChoppingBoard
from bigym.robots.configs.h1 import H1FineManipulation
from bigym.utils.env_utils import get_random_points_on_plane


class PutCups(BiGymEnv):
    """Put cups in the wall cabinet."""

    _PRESET_PATH = PRESETS_PATH / "counter_base_wall_1x1.yaml"

    _CUPS_COUNT = 2

    _CUPS_POS = np.array([0.8, 0, 1])
    _CUPS_ROT = np.deg2rad(180)
    _CUPS_STEP = 0.15
    _CUPS_POS_EXTENTS = np.array([0.1, 0.25])
    _CUPS_POS_BOUNDS = np.array([0.03, 0.03, 0])
    _CUPS_ROT_BOUNDS = np.deg2rad(30)

    def _initialize_env(self):
        self.cabinet_base = self._preset.get_props(BaseCabinet)[0]
        self.cabinet_wall = self._preset.get_props(WallCabinet)[0]
        self.cups = [Mug(self._mojo) for _ in range(self._CUPS_COUNT)]

    def _success(self) -> bool:
        for cup in self.cups:
            if not cup.is_colliding(self.cabinet_wall.shelf_bottom):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(cup, side):
                    return False
        return True

    def _on_reset(self):
        points = get_random_points_on_plane(
            len(self.cups),
            self._CUPS_POS,
            self._CUPS_POS_EXTENTS,
            self._CUPS_STEP,
            self._CUPS_POS_BOUNDS,
        )
        for cup, point in zip(self.cups, points):
            cup.body.set_position(point)
            angle = np.random.uniform(-self._CUPS_ROT_BOUNDS, self._CUPS_ROT_BOUNDS)
            cup.body.set_quaternion(
                Quaternion(axis=[0, 0, 1], angle=self._CUPS_ROT + angle).elements
            )


class TakeCups(PutCups):
    """Take cups from the wall cupboard."""

    _CUPS_POS = np.array([1.05, 0, 1.5])

    def _success(self) -> bool:
        for cup in self.cups:
            if not cup.is_colliding(self.cabinet_base.counter):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(cup, side):
                    return False
        return True


class StoreBox(BiGymEnv):
    """Put box in the cupboard task."""

    _PRESET_PATH = PRESETS_PATH / "cabinet_door.yaml"

    _BOX_POS = np.array([0.8, 0, 1])
    _BOX_POS_BOUNDS = np.array([0.03, 0.03, 0])
    _BOX_ROT_BOUNDS = np.deg2rad(180)

    def _initialize_env(self):
        self.cabinet_base = self._preset.get_props(BaseCabinet)[0]
        self.box = Box(self._mojo, True)

    def _on_reset(self):
        offset = np.random.uniform(-self._BOX_POS_BOUNDS, self._BOX_POS_BOUNDS)
        self.box.body.set_position(self._BOX_POS + offset, True)
        angle = np.random.uniform(-self._BOX_ROT_BOUNDS, self._BOX_ROT_BOUNDS)
        self.box.body.set_quaternion(Quaternion(axis=[0, 0, 1], angle=angle).elements)

    def _success(self) -> bool:
        if not self.box.is_colliding(self.cabinet_base.shelf):
            return False
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.box, side):
                return False
        return True


class PickBox(StoreBox):
    """Pick up box from and place it on the counter task."""

    _BOX_POS = np.array([0.8, -1, 0.2])
    _BOX_QUAT = Quaternion(axis=[0, 1, 0], degrees=90)

    def _success(self) -> bool:
        if not self.box.is_colliding(self.cabinet_base.counter):
            return False
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.box, side):
                return False
        return True

    def _on_reset(self):
        offset = np.random.uniform(-self._BOX_POS_BOUNDS, self._BOX_POS_BOUNDS)
        self.box.body.set_position(self._BOX_POS + offset, True)
        angle = np.random.uniform(-self._BOX_ROT_BOUNDS, self._BOX_ROT_BOUNDS)
        quat = self._BOX_QUAT
        quat *= Quaternion(axis=[1, 0, 0], angle=angle)
        self.box.body.set_quaternion(quat.elements)


class SaucepanToHob(BiGymEnv):
    """Take saucepan from cabinet and place it to hob."""

    _PRESET_PATH = PRESETS_PATH / "cabinet_hob.yaml"

    _SAUCEPAN_POS = np.array([0.85, 0.1, 0.5])
    _SAUCEPAN_QUAT = Quaternion(axis=[0, 0, 1], degrees=90)
    _SAUCEPAN_POS_BOUNDS = np.array([0.05, 0.05, 0])
    _SAUCEPAN_ROT_BOUNDS = np.deg2rad([0, 0, 20])

    def _initialize_env(self):
        self.cabinet_base = self._preset.get_props(BaseCabinet)[0]
        self.saucepan = Saucepan(self._mojo)

    def _success(self) -> bool:
        if not self.saucepan.is_colliding(self.cabinet_base.hob):
            return False
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.saucepan, side):
                return False
        return True

    def _on_reset(self):
        self.saucepan.set_pose(
            self._SAUCEPAN_POS,
            self._SAUCEPAN_QUAT.elements,
            self._SAUCEPAN_POS_BOUNDS,
            self._SAUCEPAN_ROT_BOUNDS,
        )


class StoreKitchenware(BiGymEnv):
    """Put all kitchenware to cupboard."""

    _PRESET_PATH = PRESETS_PATH / "cabinet_hob.yaml"

    _ITEMS = [Saucepan, Pan]
    _ITEMS_QUAT = Quaternion(axis=[0, 0, 1], degrees=15)
    _ITEMS_POS_BOUNDS = np.array([0.02, 0.02, 0])
    _ITEMS_ROT_BOUNDS = np.deg2rad([0, 0, 30])

    def _initialize_env(self):
        self.cabinet_base = self._preset.get_props(BaseCabinet)[0]
        self.items: list[Prop] = [item(self._mojo) for item in self._ITEMS]

    def _success(self) -> bool:
        for item in self.items:
            if not item.is_colliding(self.cabinet_base.shelf):
                return False
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(item, side):
                    return False
        return True

    def _on_reset(self):
        sites = [self.cabinet_base.sites[0], self.cabinet_base.sites[2]]
        np.random.shuffle(sites)
        for item, site in zip(self.items, sites):
            item.set_pose(
                site.get_position(),
                self._ITEMS_QUAT.elements,
                self._ITEMS_POS_BOUNDS,
                self._ITEMS_ROT_BOUNDS,
            )


class ToastSandwich(BiGymEnv):
    """Move sandwich on the frying pan."""

    DEFAULT_ROBOT = H1FineManipulation

    _PRESET_PATH = PRESETS_PATH / "counter_base_2_hob.yaml"

    _PAN_QUAT = Quaternion(axis=[0, 0, 1], degrees=-90)
    _PAN_POS_BOUNDS = np.array([0.02, 0.02, 0])
    _PAN_ROT_BOUNDS = np.deg2rad([0, 0, 30])

    _SPATULA_OFFSET = np.array([-0.02, 0.02, 0.08])
    _SPATULA_QUAT = Quaternion(axis=[0, 0, 1], degrees=90)

    _BOARD_POS = np.array([0.7, -0.6, 0.88])
    _BOARD_ROT_BOUNDS = np.deg2rad([0, 0, 5])

    _TOLERANCE = np.deg2rad(10)
    _TOASTED = False
    _ROUNDED = False

    _SANDWICH_OFFSET = np.array([0, 0, 0.05])
    _SANDWICH_POS_BOUNDS = np.array([0.05, 0.05, 0])
    _SANDWICH_ROT_BOUNDS = np.deg2rad([0, 0, 180])

    @property
    def _sandwich_anchor(self) -> Prop:
        return self.board

    def _initialize_env(self):
        self.cabinet_base = self._preset.get_props(BaseCabinet)[0]
        self.pan = Pan(self._mojo)
        self.spatula = Spatula(self._mojo)
        self.board = ChoppingBoard(self._mojo)
        self.sandwich = Sandwich(
            self._mojo, toasted=self._TOASTED, rounded_collider=self._ROUNDED
        )

    def _on_reset(self):
        site = self.cabinet_base.sites[0]
        self.pan.set_pose(
            site.get_position(),
            self._PAN_QUAT.elements,
            self._PAN_POS_BOUNDS,
            self._PAN_ROT_BOUNDS,
        )
        self.spatula.set_pose(
            self.pan.body.get_position() + self._SPATULA_OFFSET,
            self._SPATULA_QUAT.elements,
        )
        self.board.set_pose(self._BOARD_POS, rotation_bounds=self._BOARD_ROT_BOUNDS)
        self.sandwich.set_pose(
            self._sandwich_anchor.body.get_position() + self._SANDWICH_OFFSET,
            position_bounds=self._SANDWICH_POS_BOUNDS,
            rotation_bounds=self._SANDWICH_ROT_BOUNDS,
        )

    def _success(self) -> bool:
        up = np.array([0, 0, 1])
        sandwich_up = Quaternion(self.sandwich.body.get_quaternion()).rotate(up)
        angle_to_up = np.arccos(np.clip(np.dot(sandwich_up, up), -1.0, 1.0))
        angle_to_down = np.arccos(np.clip(np.dot(sandwich_up, -up), -1.0, 1.0))
        if not (angle_to_up <= self._TOLERANCE or angle_to_down <= self._TOLERANCE):
            return False
        if not self.pan.is_colliding(self.cabinet_base.hob):
            return False
        if not self.sandwich.is_colliding(self.pan):
            return False
        return True

    def _fail(self) -> bool:
        if super()._fail():
            return True
        for side in self.robot.grippers:
            if self.robot.is_gripper_holding_object(self.sandwich, side):
                return True
        return False


class FlipSandwich(ToastSandwich):
    """Flip sandwich using spatula."""

    _SANDWICH_OFFSET = np.array([0, 0, 0.04])
    _SANDWICH_POS_BOUNDS = np.array([0.01, 0.01, 0])
    _SANDWICH_ROT_BOUNDS = np.deg2rad([0, 0, 180])

    _ROUNDED = True

    @property
    def _sandwich_anchor(self) -> Prop:
        return self.pan

    def _success(self) -> bool:
        up = np.array([0, 0, 1])
        sandwich_up = Quaternion(self.sandwich.body.get_quaternion()).rotate(up)
        angle_to_down = np.arccos(np.clip(np.dot(sandwich_up, -up), -1.0, 1.0))
        if angle_to_down > self._TOLERANCE:
            return False
        if not self.pan.is_colliding(self.cabinet_base.hob):
            return False
        if not self.sandwich.is_colliding(self.pan):
            return False
        return True


class RemoveSandwich(FlipSandwich):
    """Remove sandwich from the frying pan."""

    _TOASTED = True

    def _success(self) -> bool:
        up = np.array([0, 0, 1])
        sandwich_up = Quaternion(self.sandwich.body.get_quaternion()).rotate(up)
        angle_to_up = np.arccos(np.clip(np.dot(sandwich_up, up), -1.0, 1.0))
        angle_to_down = np.arccos(np.clip(np.dot(sandwich_up, -up), -1.0, 1.0))
        if not (angle_to_up <= self._TOLERANCE or angle_to_down <= self._TOLERANCE):
            return False
        if not self.sandwich.is_colliding(self.board):
            return False
        return True
