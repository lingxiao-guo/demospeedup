"""Tool to replay demonstrations."""
from __future__ import annotations

from typing import Optional

from bigym.bigym_env import CONTROL_FREQUENCY_MAX, BiGymEnv
from demonstrations.demo import Demo, DemoStep
from demonstrations.demo_converter import DemoConverter


class DemoPlayer:
    """Tool to replay demonstrations."""

    @staticmethod
    def replay(
        demo: Demo,
        control_frequency: int,
        demo_frequency: int = CONTROL_FREQUENCY_MAX,
        render_mode: Optional[str] = None,
    ):
        """Replay demonstration in original environment."""
        env = demo.metadata.get_env(control_frequency, render_mode=render_mode)
        DemoPlayer.replay_in_env(demo, env, demo_frequency)

    @staticmethod
    def replay_in_env(
        demo: Demo, env: BiGymEnv, demo_frequency: int = CONTROL_FREQUENCY_MAX
    ):
        """Replay demonstration in environment."""
        timesteps = DemoPlayer._get_timesteps_for_replay(demo, env, demo_frequency)
        env.reset(seed=demo.seed)
        for step in timesteps:
            action = step.executed_action
            env.step(action, fast=True)
            if env.render_mode:
                env.render()
        env.close()

    @staticmethod
    def validate(
        demo: Demo,
        control_frequency: int,
        demo_frequency: int = CONTROL_FREQUENCY_MAX,
    ) -> bool:
        """Replay demonstration in original environment."""
        env = demo.metadata.get_env(control_frequency)
        return DemoPlayer.validate_in_env(demo, env, demo_frequency)

    @staticmethod
    def validate_in_env(
        demo: Demo, env: BiGymEnv, demo_frequency: int = CONTROL_FREQUENCY_MAX
    ) -> bool:
        """Check if demonstration is successful in environment."""
        timesteps = DemoPlayer._get_timesteps_for_replay(demo, env, demo_frequency)
        env.reset(seed=demo.seed)
        is_successful = False
        for step in timesteps:
            action = step.executed_action
            env.step(action, fast=True)
            if env.reward > 0:
                is_successful = True
                break
        env.close()
        return is_successful

    @staticmethod
    def _get_timesteps_for_replay(
        demo: Demo, env: BiGymEnv, demo_frequency: int = CONTROL_FREQUENCY_MAX
    ) -> list[DemoStep]:
        if env.control_frequency != demo_frequency:
            timesteps = DemoConverter.decimate(
                demo,
                target_freq=env.control_frequency,
                original_freq=demo_frequency,
                robot=env.robot,
            ).timesteps
        else:
            timesteps = demo.timesteps
        return timesteps
