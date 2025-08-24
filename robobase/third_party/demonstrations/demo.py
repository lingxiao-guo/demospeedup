"""Class to hold demo."""
from __future__ import annotations
import logging
import numpy as np
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from safetensors import safe_open
from safetensors.numpy import save_file
from typing import Optional, Any, Union, Iterable

from gymnasium.core import ActType

from bigym.bigym_env import BiGymEnv, CONTROL_FREQUENCY_MAX

from demonstrations.utils import (
    Metadata,
    ObservationMode,
)

from demonstrations.const import (
    ACTION_KEY,
    VISUAL_OBSERVATIONS_PREFIX,
    SAFETENSORS_SUFFIX,
    SAFETENSORS_OBSERVATION_PREFIX,
    SAFETENSORS_INFO_PREFIX,
    GYM_OBSERVATION_KEY,
    GYM_REWARD_KEY,
    GYM_TERMINATIION_KEY,
    GYM_TRUNCATION_KEY,
    GYM_INFO_KEY,
)

# Additional steps after termination/truncation
TERMINATION_STEPS = CONTROL_FREQUENCY_MAX * 2


@dataclass
class DemoStep:
    """Class to hold a time step."""

    observation: dict[str, Any]
    reward: float
    termination: bool
    truncation: bool
    info: dict[str, Any]

    def __init__(self, observation, reward, termination, truncation, info, action):
        """Init.

        Args:
            observation: A dictionary containing time step information.
            reward: The reward of the time step.
            termination: Whether the episode terminated.
            truncation: Whether the episode was truncated.
            info: A dictionary containing additional information.
            action: The action taken to reach the time step.
        """
        self.observation = observation
        self.reward = reward
        self.termination = termination
        self.truncation = truncation
        self.info = info
        self.set_executed_action(action)

    @property
    def executed_action(self) -> ActType:
        """Executed action."""
        return self.info.get(ACTION_KEY, None)

    def set_executed_action(self, action):
        """Set the executed action."""
        # Convert to float64 to reduce compounding errors
        self.info[ACTION_KEY] = np.float64(action)

    @property
    def has_visual_observations(self) -> bool:
        """Check if this timestep has visual observations."""
        return any(
            key.lower().startswith(VISUAL_OBSERVATIONS_PREFIX)
            for key in self.observation
        )

    @property
    def visual_observations(self) -> dict[str, np.ndarray]:
        """Get all visual observations of the current timestep."""
        visual_observations: dict[str, np.ndarray] = {}
        for key, observation in self.observation.items():
            if key.lower().startswith(VISUAL_OBSERVATIONS_PREFIX):
                visual_observations[key] = observation
        return visual_observations


class Demo:
    """Class to hold demo."""

    def __init__(
        self,
        metadata: Metadata,
        timesteps: Optional[list[DemoStep]] = None,
    ):
        """Init.

        Args:
            metadata: Metadata demo information.
            timesteps: A list of time steps.
        """
        self._metadata: Metadata = metadata
        if timesteps is not None:
            self._steps: list[DemoStep] = timesteps
        else:
            self._steps: list[DemoStep] = []

    @classmethod
    def from_safetensors(
        cls, demo_path: Path, override_metadata: Optional[Metadata] = None
    ) -> Optional[Demo]:
        """Load demo from a safetensors file.

        Args:
            demo_path(Path): Path to safetensors file.
            override_metadata(Metadata): Optional metadata override.

        Returns:
            A Demo object.
        """
        if isinstance(demo_path, str):
            demo_path = Path(demo_path)
        if not demo_path.suffix == SAFETENSORS_SUFFIX:
            demo_path = demo_path.with_suffix(SAFETENSORS_SUFFIX)
        if not demo_path.exists():
            logging.error(f"File {demo_path} does not exist.")
            return None
        metadata = override_metadata or Metadata.from_safetensors(demo_path)
        if metadata.observation_mode == ObservationMode.Lightweight:
            return LightweightDemo.from_safetensors(demo_path, override_metadata)
        demo = cls.load_timesteps_from_safetensors(demo_path)
        timesteps = [DemoStep(*step, step[-1][ACTION_KEY]) for step in demo]
        return cls(
            metadata=metadata,
            timesteps=timesteps,
        )

    @classmethod
    def from_env(cls, env: BiGymEnv) -> Demo:
        """Create a demo from an environment.

        Args:
            env: The environment to record.

        Returns:
            A Demo object.
        """
        return cls(
            metadata=Metadata.from_env(env),
        )

    @staticmethod
    def load_timesteps_from_safetensors(
        demo_path: Path,
    ):
        """Load timesteps from a safetensors file.

        Args:
            demo_path(Path): Path to safetensors file.

        Returns:
            List[Tuple(Dict[str, np.ndarray])]: a list of time steps.
        """
        demo_dict = {
            GYM_OBSERVATION_KEY: {},
            GYM_REWARD_KEY: None,
            GYM_TERMINATIION_KEY: None,
            GYM_TRUNCATION_KEY: None,
            GYM_INFO_KEY: {},
        }
        demo_path = Path(demo_path)
        logging.debug(f"Processing demo {demo_path}")
        with safe_open(demo_path, framework="np", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                t = f.get_tensor(key)
                if key.startswith(SAFETENSORS_OBSERVATION_PREFIX):
                    demo_dict[GYM_OBSERVATION_KEY][
                        key.removeprefix(SAFETENSORS_OBSERVATION_PREFIX)
                    ] = t
                elif key.startswith(SAFETENSORS_INFO_PREFIX):
                    demo_dict[GYM_INFO_KEY][
                        key.removeprefix(SAFETENSORS_INFO_PREFIX)
                    ] = t
                elif key in demo_dict:
                    demo_dict[key] = t
                else:
                    demo_dict[GYM_INFO_KEY][key] = t

        demo_length = len(demo_dict[GYM_INFO_KEY][ACTION_KEY])

        # Convert demo_dict
        #   from:   Dict[Dict[str, List[np.ndarray]]]
        #   to:     List[Tuple(Dict[str, np.ndarray])]
        demo = []

        def is_iterable(variable):
            return isinstance(variable, Iterable) and not isinstance(variable, str)

        for step_id in range(demo_length):
            demo_step_dict = {}
            for key, value in demo_dict.items():
                if isinstance(value, dict):
                    sub_dict = {}
                    for sub_key, sub_value in value.items():
                        sub_dict[sub_key] = (
                            sub_value[step_id]
                            if is_iterable(sub_value) and len(sub_value) == demo_length
                            else None
                        )
                    demo_step_dict[key] = sub_dict
                elif is_iterable(value) and len(value) > 0:
                    demo_step_dict[key] = value[step_id]
                else:
                    demo_step_dict[key] = value
            demo.append(tuple(demo_step_dict.values()))
        return demo

    @property
    def _saving_format(self):
        """Saving format.

        Returns:
            A dictionary containing timesteps ready to be saved.
        """
        to_save = {
            GYM_OBSERVATION_KEY: defaultdict(list),
            GYM_REWARD_KEY: [],
            GYM_TERMINATIION_KEY: [],
            GYM_TRUNCATION_KEY: [],
            GYM_INFO_KEY: defaultdict(list),
        }
        for step in self._steps:
            for key, val in step.observation.items():
                to_save[GYM_OBSERVATION_KEY][key].append(val)
            to_save[GYM_REWARD_KEY].append(step.reward)
            to_save[GYM_TERMINATIION_KEY].append(step.termination)
            to_save[GYM_TRUNCATION_KEY].append(step.truncation)
            for key, val in step.info.items():
                to_save[GYM_INFO_KEY][key].append(val)
        return to_save

    def save(self, path: Union[str, Path]) -> Path:
        """Save a gymnasium demo to a file.

        Args:
            path: The path to the file.

        Returns:
            The path to the file.
        """
        if isinstance(path, str):
            path = Path(path)
        float_dtype = np.float64
        path.parent.mkdir(exist_ok=True, parents=True)
        timesteps = self._saving_format
        demo_dict = {
            f"{SAFETENSORS_OBSERVATION_PREFIX}{key}": (
                np.asarray(val, dtype=float_dtype)
                if np.issubdtype(np.asarray(val).dtype, np.floating)
                else np.asarray(val)
            )
            for key, val in timesteps[GYM_OBSERVATION_KEY].items()
        }
        demo_dict[GYM_REWARD_KEY] = np.asarray(
            timesteps[GYM_REWARD_KEY], dtype=float_dtype
        )
        demo_dict[GYM_TERMINATIION_KEY] = np.asarray(timesteps[GYM_TERMINATIION_KEY])
        demo_dict[GYM_TRUNCATION_KEY] = np.asarray(timesteps[GYM_TRUNCATION_KEY])
        demo_dict |= {
            f"{SAFETENSORS_INFO_PREFIX}{key}": (
                np.asarray(val, dtype=float_dtype)
                if np.issubdtype(np.asarray(val).dtype, np.floating)
                else np.asarray(val)
            )
            for key, val in timesteps[GYM_INFO_KEY].items()
        }

        save_file(demo_dict, path, self.safetensor_metadata)
        logging.info(f"Saved {path}")
        return path

    @property
    def timesteps(self) -> list[DemoStep]:
        """Time steps."""
        return self._steps.copy()

    @property
    def duration(self) -> int:
        """Amount of demo steps."""
        return len(self._steps)

    @property
    def seed(self):
        """Seed of demo."""
        return self._metadata.seed

    @property
    def metadata(self) -> Metadata:
        """Metadata."""
        return self._metadata

    @property
    def uuid(self):
        """UUID."""
        return self._metadata.uuid

    @property
    def safetensor_metadata(self):
        """Metadata in safetensors format."""
        return self.metadata.ready_for_safetensors()

    def add_timestep(self, observation, reward, termination, truncation, info, action):
        """Add a time step to the recording.

        Args:
            observation: A dictionary containing time step information.
            reward: The reward of the time step.
            termination: Whether the episode terminated.
            truncation: Whether the episode was truncated.
            info: A dictionary containing additional information.
            action: The action taken to reach the time step.
        """
        timestep = DemoStep(observation, reward, termination, truncation, info, action)
        self._steps.append(timestep)

    def add_termination_steps(self, steps_count: int):
        """Duplicate last step multiple times."""
        steps = [deepcopy(self._steps[-1])] * steps_count
        self._steps.extend(steps)


class LightweightDemo(Demo):
    """Class to hold a lightweight demo."""

    def __init__(
        self,
        metadata: Metadata,
        timesteps=None,
    ):
        """Init.

        Args:
            metadata(dict): Metadata demo information.
            timesteps(list): A list of time steps.
        """
        super().__init__(metadata, timesteps)
        self._metadata.observation_mode = ObservationMode.Lightweight
        if timesteps is not None:
            self._steps: list[DemoStep] = LightweightDemo.lighten_timesteps(timesteps)
        else:
            self._steps: list[DemoStep] = []

    @classmethod
    def from_safetensors(
        cls, demo_path: Path, override_metadata: Optional[Metadata] = None
    ) -> Optional[Demo]:
        """Load demo from a safetensors file.

        Args:
            demo_path(Path): Path to safetensors file.
            override_metadata(Metadata): Optional metadata override.

        Returns:
            A Demo object.
        """
        if isinstance(demo_path, str):
            demo_path = Path(demo_path)
        if not demo_path.suffix == SAFETENSORS_SUFFIX:
            demo_path = demo_path.with_suffix(SAFETENSORS_SUFFIX)
        if not demo_path.exists():
            logging.error(f"File {demo_path} does not exist.")
            return None
        metadata = override_metadata or Metadata.from_safetensors(demo_path)
        if not metadata.observation_mode == ObservationMode.Lightweight:
            raise ValueError(
                f"Demo {demo_path} is not a lightweight demo. "
                "Use `Demo.from_safetensors` instead."
            )
        demo = cls.load_timesteps_from_safetensors(demo_path)
        timesteps = [DemoStep(*step, step[-1][ACTION_KEY]) for step in demo]
        return cls(
            metadata=metadata,
            timesteps=timesteps,
        )

    @classmethod
    def from_demo(cls, demo: Demo) -> LightweightDemo:
        """Create a lightweight demo from a demo.

        Args:
            demo: The demo to lighten.

        Returns:
            A LightweightDemo object.
        """
        return cls(
            metadata=deepcopy(demo.metadata),
            timesteps=deepcopy(demo.timesteps),
        )

    @classmethod
    def from_env(cls, env: BiGymEnv) -> Demo:
        """Create a demo from an environment.

        Args:
            env: The environment to record.

        Returns:
            A Demo object.
        """
        return cls(
            metadata=Metadata.from_env(env),
        )

    @property
    def _saving_format(self):
        """Saving format.

        Returns:
            A dictionary containing timesteps ready to be saved.
        """
        to_save = {
            GYM_OBSERVATION_KEY: defaultdict(list),
            GYM_REWARD_KEY: [],
            GYM_TERMINATIION_KEY: [],
            GYM_TRUNCATION_KEY: [],
            GYM_INFO_KEY: defaultdict(list),
        }
        for step in self._steps:
            to_save[GYM_TERMINATIION_KEY].append(step.termination)
            to_save[GYM_TRUNCATION_KEY].append(step.truncation)
            to_save[GYM_INFO_KEY][ACTION_KEY].append(step.executed_action)
        return to_save

    def add_timestep(self, observation, reward, termination, truncation, info, action):
        """Add a time step to the recording.

        Args:
            observation: A dictionary containing time step information.
            reward: The reward of the time step.
            termination: Whether the episode terminated.
            truncation: Whether the episode was truncated.
            info: A dictionary containing additional information.
            action: The action taken to reach the time step.
        """
        timestep = DemoStep({}, None, termination, truncation, {}, action)
        self._steps.append(timestep)

    @staticmethod
    def lighten_timesteps(timesteps: list[DemoStep]) -> list[DemoStep]:
        """Lighten the timesteps.

        Args:
            timesteps(list): A list of time steps.

        Returns:
            A list of time steps.
        """
        return [
            DemoStep(
                {}, None, step.termination, step.truncation, {}, step.executed_action
            )
            for step in timesteps
        ]
