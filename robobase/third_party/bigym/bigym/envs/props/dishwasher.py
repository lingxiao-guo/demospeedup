"""Dishwasher."""
from pathlib import Path
from typing import Optional

import numpy as np
from mojo import Mojo
from mojo.elements import Joint, Site, Body, Geom

from bigym.const import ASSETS_PATH
from bigym.envs.props.prop import Prop
from bigym.utils.physics_utils import get_joint_position, set_joint_position


class DishwasherPart:
    """Part of the dishwasher."""

    def __init__(
        self,
        mojo: Mojo,
        dishwasher_body: Body,
        body_name: Optional[str] = None,
        joint_name: Optional[str] = None,
        site_sets: Optional[list[tuple[str, int]]] = None,
    ):
        """Init."""
        self._mojo = mojo
        self.body: Optional[Body] = (
            Body.get(self._mojo, body_name, dishwasher_body) if body_name else None
        )
        self.joint: Optional[Joint] = (
            Joint.get(self._mojo, joint_name, dishwasher_body) if joint_name else None
        )
        self.site_sets: list[list[Site]] = []
        if site_sets:
            for sites_name, sites_count in site_sets:
                sites_set: list[Site] = []
                for i in range(sites_count):
                    site = Site.get(
                        self._mojo, f"{sites_name}_{i + 1}", dishwasher_body
                    )
                    sites_set.append(site)
                self.site_sets.append(sites_set)
        self.colliders: list[Geom] = Prop.get_body_colliders(self.body)


class Dishwasher(Prop):
    """Dishwasher."""

    DOOR_BODY = "dishwasher/door"
    DOOR_JOINT = "dishwasher/door_hinge"

    TRAY_BOTTOM_BODY = "dishwasher/tray_bottom"
    TRAY_BOTTOM_JOINT = "dishwasher/tray_bottom_linear"
    TRAY_BOTTOM_SITES = [
        ("dishwasher/tray_bottom_holder_1", 11),
        ("dishwasher/tray_bottom_holder_2", 11),
    ]

    TRAY_MIDDLE_BODY = "dishwasher/tray_mid"
    TRAY_MIDDLE_JOINT = "dishwasher/tray_mid_linear"
    TRAY_MIDDLE_SITES = [
        ("dishwasher/tray_mid_holder_sites_1", 11),
        ("dishwasher/tray_mid_holder_sites_2", 11),
    ]

    BASKET_BODY = "dishwasher/cuttlery_basket"
    BASKET_SITES = [("dishwasher/cuttlery_basket", 6)]

    @property
    def _model_path(self) -> Path:
        return ASSETS_PATH / "props/dishwasher/dishwasher.xml"

    def _post_init(self):
        """Init."""
        self.door = DishwasherPart(
            self._mojo, self.body, self.DOOR_BODY, self.DOOR_JOINT
        )
        self.tray_bottom = DishwasherPart(
            self._mojo,
            self.body,
            self.TRAY_BOTTOM_BODY,
            self.TRAY_BOTTOM_JOINT,
            self.TRAY_BOTTOM_SITES,
        )
        self.tray_middle = DishwasherPart(
            self._mojo,
            self.body,
            self.TRAY_MIDDLE_BODY,
            self.TRAY_MIDDLE_JOINT,
            self.TRAY_MIDDLE_SITES,
        )
        self.basket = DishwasherPart(
            self._mojo, self.body, self.BASKET_BODY, site_sets=self.BASKET_SITES
        )

    def set_state(self, door: float, bottom_tray: float, middle_tray: float):
        """Set state of dishwasher joints."""
        set_joint_position(self.door.joint, door, True)
        set_joint_position(self.tray_bottom.joint, bottom_tray, True)
        set_joint_position(self.tray_middle.joint, middle_tray, True)

    def get_state(self) -> np.ndarray:
        """Get state of dishwasher joints."""
        return np.array(
            [
                get_joint_position(self.door.joint, True),
                get_joint_position(self.tray_bottom.joint, True),
                get_joint_position(self.tray_middle.joint, True),
            ]
        )
