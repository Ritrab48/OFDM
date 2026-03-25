"""Smoke tests for the pure-Python OFDM-ISAC reference implementation."""

from __future__ import annotations

import unittest

from ofdm_isac_rl import (
    DQNAgent,
    OfdmIsacEnv,
    SystemConfig,
    TrainingConfig,
    build_baseline_policies,
)


class ReferenceModelTests(unittest.TestCase):
    def test_environment_step_returns_finite_metrics(self) -> None:
        env = OfdmIsacEnv(SystemConfig(episode_length=4, seed=3))
        state = env.reset()
        self.assertEqual(len(state), 7)
        next_state, reward, done, info = env.step(0)
        self.assertEqual(len(next_state), 7)
        self.assertIsInstance(done, bool)
        self.assertGreaterEqual(info["num_pilots"], 1.0)
        self.assertTrue(reward == reward)

    def test_ici_power_uses_data_power_instead_of_total_power(self) -> None:
        config = SystemConfig(episode_length=4, seed=3)
        env = OfdmIsacEnv(config)
        env.reset()
        _, _, _, info = env.step(1)
        expected_observed_ici_power = (
            config.ici_constant
            * info["residual_cfo_power_est"]
            * info["data_power"]
            * info["channel_power_est"]
        )
        expected_observed_sinr = (
            info["data_power"]
            * info["channel_power_est"]
            / (config.noise_variance + expected_observed_ici_power)
        )
        expected_true_ici_power = (
            config.ici_constant
            * (info["residual_cfo_hz"] ** 2)
            * info["data_power"]
            * info["channel_power_true"]
        )
        expected_true_sinr = (
            info["data_power"]
            * info["channel_power_true"]
            / (config.noise_variance + expected_true_ici_power)
        )
        self.assertAlmostEqual(info["ici_power"], expected_observed_ici_power, places=12)
        self.assertAlmostEqual(info["sinr"], expected_observed_sinr, places=12)
        self.assertAlmostEqual(info["ici_power_true"], expected_true_ici_power, places=12)
        self.assertAlmostEqual(info["sinr_true"], expected_true_sinr, places=12)

    def test_agent_training_step_runs(self) -> None:
        system = SystemConfig(episode_length=4, seed=5)
        training = TrainingConfig(
            episodes=1,
            replay_capacity=64,
            batch_size=8,
            warmup_steps=8,
            target_update_interval=4,
            seed=6,
        )
        env = OfdmIsacEnv(system)
        agent = DQNAgent(system, training)
        state = env.reset()
        for _ in range(12):
            action = agent.select_action(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            if done:
                state = env.reset()
        self.assertGreater(agent.total_steps, 0)

    def test_baseline_policies_cover_fixed_random_and_heuristic(self) -> None:
        system = SystemConfig()
        policies = build_baseline_policies(system, seed=9)
        self.assertEqual(len(policies), len(system.action_space) + 2)
        policy_names = {policy.name for policy in policies}
        self.assertIn("random", policy_names)
        self.assertIn("heuristic", policy_names)


if __name__ == "__main__":
    unittest.main()
