"""Helper class for converting between different action representations."""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np
from tqdm import tqdm

from bigym.robots.robot import Robot
from demonstrations.demo import Demo, DemoStep
from demonstrations.utils import Metadata

from bigym.bigym_env import BiGymEnv, CONTROL_FREQUENCY_MAX


class DemoConverter:
    """Class to convert demonstrations."""

    @staticmethod
    def absolute_to_delta(demo: Demo) -> Demo:
        """Converts a demonstration from absolute to delta actions.

        :param demo: The demonstration to convert (in absolute joint positions).
        :return: The converted demonstration (in delta joint positions).
        """

        def get_delta_action(
            prev_action: np.ndarray,
            action: np.ndarray,
            base_dof_count: int,
            grippers_count: int,
        ) -> np.ndarray:
            delta = action - prev_action
            delta[:base_dof_count] = action[:base_dof_count]
            delta[-grippers_count:] = action[-grippers_count:]
            return delta

        timesteps = deepcopy(demo.timesteps)
        if demo.metadata.environment_data.action_mode_absolute:
            demo.metadata.environment_data.action_mode_absolute = False

        # Cache environment info
        robot = demo.metadata.get_robot()
        action_space = robot.action_mode.action_space(1)
        floating_dof_count = len(robot.action_mode.floating_dofs)
        grippers_count = len(robot.grippers)

        overhead = np.zeros_like(action_space.sample())
        last_action = np.zeros_like(action_space.sample())
        for timestep in timesteps:
            absolute_action = timestep.executed_action + overhead
            delta_action = get_delta_action(
                last_action, absolute_action, floating_dof_count, grippers_count
            )
            clipped_action = np.clip(delta_action, action_space.low, action_space.high)
            overhead = delta_action - clipped_action
            if not np.allclose(overhead, 0):
                timestep.set_executed_action(clipped_action)
                last_action = absolute_action - overhead
            else:
                overhead *= 0
                timestep.set_executed_action(delta_action)
                last_action = absolute_action
        if demo.metadata.environment_data.action_mode_absolute:
            demo.metadata.environment_data.action_mode_absolute = False
        return Demo(demo.metadata, timesteps)

    @staticmethod
    def clip_actions(demo: Demo, action_scale: float = 1) -> Demo:
        """Clip demo actions to action space."""
        timesteps = deepcopy(demo.timesteps)
        action_space = demo.metadata.get_action_space(action_scale)
        overhead = np.zeros_like(action_space.sample())
        for timestep in timesteps:
            action = timestep.executed_action + overhead
            clipped_action = np.clip(action, action_space.low, action_space.high)
            overhead = action - clipped_action
            timestep.set_executed_action(clipped_action)
        return Demo(demo.metadata, timesteps)

    @staticmethod
    def decimate(
        demo: Demo,
        target_freq: int,
        original_freq: int = CONTROL_FREQUENCY_MAX,
        robot: Optional[Robot] = None,
    ) -> Demo:
        """Decimate provided demo at certain rate.

        :param demo: Original demonstration.
        :param target_freq: Control frequency of the new demo.
        :param original_freq: Control frequency of the original demo.
        :param robot: Optional existing robot instance to speed-up decimation.
        """
        if original_freq != CONTROL_FREQUENCY_MAX:
            raise RuntimeError(
                f"Demonstrations with frequency != {CONTROL_FREQUENCY_MAX} "
                f"can't be decimated."
            )

        decimation_rate = int(np.round(original_freq / target_freq))
        robot = robot or demo.metadata.get_robot()
        action_space = robot.action_mode.action_space(decimation_rate)
        grippers_count = len(robot.grippers)

        original_timesteps = deepcopy(demo.timesteps)
        decimated_timesteps: list[DemoStep] = []

        action = np.zeros_like(action_space.sample())
        overhead = np.zeros_like(action_space.sample())

        # Repeat final actions to ensure success
        if 0 < len(original_timesteps) % decimation_rate < decimation_rate:
            steps_count = decimation_rate - len(original_timesteps) % decimation_rate
            original_timesteps.extend([deepcopy(original_timesteps[-1])] * steps_count)

        actions_counter = 0
        for timestep in original_timesteps:
            timestep = deepcopy(timestep)
            original_action = timestep.executed_action.copy()
            action += original_action + overhead
            overhead *= 0
            actions_counter += 1
            if actions_counter % decimation_rate == 0:
                if demo.metadata.environment_data.action_mode_absolute:
                    floating_base_actions = demo.metadata.floating_dof_count
                    action[floating_base_actions:] = (
                        action[floating_base_actions:] / decimation_rate
                    )
                action[-grippers_count:] = original_action[-grippers_count:]
                clipped_action = np.clip(action, action_space.low, action_space.high)
                timestep.set_executed_action(clipped_action)
                decimated_timesteps.append(timestep)
                overhead = action - clipped_action
                action = np.zeros_like(action)
        return Demo(demo.metadata, decimated_timesteps)

    @staticmethod
    def create_demo_in_new_env(
        demo: Demo,
        env: BiGymEnv,
    ) -> Demo:
        """Create a new demonstration in a new environment.

        :param demo: The demonstration to convert.
        :param env: The environment to collect the new demonstration in (action
            mode must match the demonstration).

        :return: The new demonstration.
        """
        observation, _ = env.reset(seed=demo.seed)
        metadata = Metadata.from_env(env)
        metadata.uuid = demo.metadata.uuid
        new_demo = Demo(metadata)
        t = 0
        floating_base_actions = demo.metadata.floating_dof_count
        with tqdm(
            total=len(demo.timesteps),
            desc="Creating Demo",
            unit="step",
            leave=False,
        ) as pbar:
            for timestep in demo.timesteps:
                action = timestep.executed_action
                if t==0:
                    action[:floating_base_actions]+=observation["proprioception_floating_base"]
                else:
                    action[:floating_base_actions] += last_floating_base_actions
                last_floating_base_actions = action[:floating_base_actions]
                observation, reward, term, trunc, info = env.step(action)
                new_demo.add_timestep(
                    observation,
                    reward,
                    term,
                    trunc,
                    info,
                    action,
                )
                t += 1
                pbar.update()

        return new_demo
