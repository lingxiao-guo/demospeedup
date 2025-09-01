"""Groceries tasks."""


import numpy as np

from bigym.bigym_env import BiGymEnv
from bigym.const import PRESETS_PATH
from bigym.envs.props.cabintets import BaseCabinet, WallCabinet, OpenShelf
from bigym.envs.props.items import Wine, Beer, Soap, Cereals, Ketchup, Mustard, Soda
from bigym.envs.props.prop import Prop
from bigym.utils.env_utils import get_random_points_on_plane


class GroceriesStoreLower(BiGymEnv):
    """Put groceries to lower cabinets tasks."""

    _PRESET_PATH = PRESETS_PATH / "counter_base_2.yaml"

    _PROP_TYPES = [Wine, Soap, Beer, Cereals, Ketchup, Mustard, Soda]

    _PROPS_PER_CATEGORY = 1
    _PROPS_COUNT = 4

    _PROPS_POS = np.array([0.7, -0.3, 0.9])
    _PROPS_STEP = 0.2
    _PROPS_POS_EXTENTS = np.array([0.1, 0.5])
    _PROPS_POS_BOUNDS = np.array([0.1, 0.05, 0])
    _PROPS_ROT_BOUNDS = np.deg2rad([0, 0, 180])

    def _initialize_env(self):
        self.cabinet_1 = self._preset.get_props(BaseCabinet)[0]
        self.cabinet_2 = self._preset.get_props(BaseCabinet)[1]
        self.props: list[Prop] = []
        self.selected_props: list[Prop] = []
        for prop_type in self._PROP_TYPES:
            self.props.extend(
                [prop_type(self._mojo) for _ in range(self._PROPS_PER_CATEGORY)]
            )

    def _on_reset(self):
        for prop in self.props:
            prop.disable()

        self.selected_props = np.random.choice(
            np.array(self.props), size=self._PROPS_COUNT, replace=False
        )
        points = get_random_points_on_plane(
            len(self.selected_props),
            self._PROPS_POS,
            self._PROPS_POS_EXTENTS,
            self._PROPS_STEP,
        )
        for prop, point in zip(self.selected_props, points):
            prop.enable()
            prop.set_pose(
                point,
                position_bounds=self._PROPS_POS_BOUNDS,
                rotation_bounds=self._PROPS_ROT_BOUNDS,
            )

    def _success(self) -> bool:
        for prop in self.selected_props:
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(prop, side):
                    return False
            if not (
                prop.is_colliding(self.cabinet_1.shelf)
                or prop.is_colliding(self.cabinet_1.shelf_bottom)
                or prop.is_colliding(self.cabinet_2.shelf_bottom)
            ):
                return False
        return True


class GroceriesStoreUpper(GroceriesStoreLower):
    """Put groceries to upper cabinets tasks."""

    _PRESET_PATH = PRESETS_PATH / "counter_base_wall_2x2.yaml"

    def _initialize_env(self):
        super()._initialize_env()
        self.cabinet_wall = self._preset.get_props(WallCabinet)[0]
        self.shelf = self._preset.get_props(OpenShelf)[0]

    def _success(self) -> bool:
        for prop in self.selected_props:
            for side in self.robot.grippers:
                if self.robot.is_gripper_holding_object(prop, side):
                    return False
            if not (
                prop.is_colliding(self.cabinet_wall.shelf)
                or prop.is_colliding(self.cabinet_wall.shelf_bottom)
                or prop.is_colliding(self.shelf.shelf)
            ):
                return False
        return True
