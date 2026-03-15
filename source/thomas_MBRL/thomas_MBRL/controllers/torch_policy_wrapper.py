"""
torch_policy_wrapper.py

Small wrapper around a TorchScript locomotion policy for Isaac Lab.

This module keeps policy loading and observation extraction separate from the
MPPI runner so the runner stays focused on command optimization.
"""

from __future__ import annotations

from typing import Any

import torch


class TorchVelocityPolicy:
    """
    Wrapper for a TorchScript velocity policy.

    The policy is expected to accept a batch of observations and return a batch
    of actions with shape (num_envs, action_dim).
    """

    def __init__(self, checkpoint_path: str, device: str) -> None:
        """
        Load the TorchScript policy and move it to the requested device.

        Args:
            checkpoint_path: Path to the TorchScript policy file.
            device: Torch device string such as "cuda" or "cpu".
        """
        self.device = torch.device(device)
        self.policy = torch.jit.load(checkpoint_path, map_location=self.device)
        self.policy.eval()

    def extract_policy_obs(self, obs: Any) -> torch.Tensor:
        """
        Extract the policy observation tensor from a Gym/Isaac Lab observation.

        Isaac Lab commonly returns either a tensor directly or a dictionary with
        keys such as "policy" or "obs". This helper handles the common cases.

        Args:
            obs: Observation object returned by the environment or observation
                manager.

        Returns:
            Batched observation tensor on the correct device.

        Raises:
            TypeError: If the observation format is unsupported.
            KeyError: If the observation is a dictionary but no usable tensor is
                found.
        """
        if isinstance(obs, torch.Tensor):
            return obs.to(self.device)

        if isinstance(obs, dict):
            preferred_keys = ("policy", "obs", "observation", "actor")
            for key in preferred_keys:
                if key in obs and isinstance(obs[key], torch.Tensor):
                    return obs[key].to(self.device)

            for value in obs.values():
                if isinstance(value, torch.Tensor):
                    return value.to(self.device)

            raise KeyError("No tensor-valued observation found in observation dictionary.")

        raise TypeError(f"Unsupported observation type: {type(obs)}")

    @torch.no_grad()
    def __call__(self, obs: Any) -> torch.Tensor:
        """
        Run the policy on a batch of observations.

        Args:
            obs: Environment observation or observation dictionary.

        Returns:
            Batched action tensor with shape (num_envs, action_dim).
        """
        policy_obs = self.extract_policy_obs(obs)
        actions = self.policy(policy_obs)

        if not isinstance(actions, torch.Tensor):
            raise TypeError("Policy output must be a torch.Tensor.")

        return actions.to(self.device)