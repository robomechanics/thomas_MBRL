from __future__ import annotations

import torch

from .models import DynamicsEnsemble


class TrajectoryPlanner:
    """Shared rollout evaluation utilities for action-sequence planners."""

    def __init__(
        self,
        model: DynamicsEnsemble,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        horizon: int = 30,
        candidates: int = 512,
        discount: float = 0.99,
        temperature: float = 0.5,
    ):
        self.model = model
        self.action_low = action_low
        self.action_high = action_high
        self.horizon = horizon
        self.candidates = candidates
        self.discount = discount
        self.temperature = temperature
        self.action_dim = action_low.numel()
        self._prev_mean: torch.Tensor | None = None

    def _warm_start_mean(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        shape = (batch_size, self.horizon, self.action_dim)
        if self._prev_mean is None or self._prev_mean.shape != shape or self._prev_mean.device != obs.device:
            return torch.zeros(shape, device=obs.device, dtype=obs.dtype)

        shifted = torch.zeros_like(self._prev_mean)
        shifted[:, :-1, :] = self._prev_mean[:, 1:, :]
        return shifted

    def _clip_action_sequences(self, action_sequences: torch.Tensor) -> torch.Tensor:
        return torch.max(
            torch.min(action_sequences, self.action_high.view(1, 1, 1, -1)),
            self.action_low.view(1, 1, 1, -1),
        )

    @torch.no_grad()
    def evaluate_sequences(self, obs: torch.Tensor, action_sequences: torch.Tensor) -> torch.Tensor:
        batch_size, candidates, _, action_dim = action_sequences.shape
        states = obs.unsqueeze(1).expand(-1, candidates, -1).reshape(batch_size * candidates, -1)
        returns = torch.zeros(batch_size * candidates, device=obs.device, dtype=obs.dtype)
        discounts = torch.ones_like(returns)
        alive = torch.ones_like(returns)

        for t in range(self.horizon):
            actions_t = action_sequences[:, :, t, :].reshape(batch_size * candidates, action_dim)
            preds = self.model.predict(states, actions_t)
            reward = preds.rewards.squeeze(-1)
            continue_prob = preds.continue_logits.sigmoid().squeeze(-1)
            returns = returns + discounts * alive * reward
            alive = alive * continue_prob
            discounts = discounts * self.discount
            states = states + preds.delta_obs

        return returns.view(batch_size, candidates)


class CEMPlanner(TrajectoryPlanner):
    """Cross-entropy planner over learned dynamics."""

    def __init__(
        self,
        model: DynamicsEnsemble,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        horizon: int = 30,
        candidates: int = 512,
        elites: int = 64,
        iterations: int = 5,
        discount: float = 0.99,
        temperature: float = 0.5,
    ):
        super().__init__(
            model=model,
            action_low=action_low,
            action_high=action_high,
            horizon=horizon,
            candidates=candidates,
            discount=discount,
            temperature=temperature,
        )
        self.elites = elites
        self.iterations = iterations

    def plan(self, obs: torch.Tensor) -> torch.Tensor:
        mean = self._warm_start_mean(obs)
        std = torch.full_like(mean, self.temperature)

        for _ in range(self.iterations):
            noise = torch.randn(
                (obs.shape[0], self.candidates, self.horizon, self.action_dim), device=obs.device, dtype=obs.dtype
            )
            action_sequences = self._clip_action_sequences(mean.unsqueeze(1) + std.unsqueeze(1) * noise)

            returns = self.evaluate_sequences(obs, action_sequences)
            elite_indices = returns.topk(self.elites, dim=1).indices
            elite_actions = action_sequences.gather(
                1, elite_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.horizon, self.action_dim)
            )
            mean = elite_actions.mean(dim=1)
            std = elite_actions.std(dim=1).clamp_min(1e-3)

        self._prev_mean = mean.detach()
        return mean[:, 0, :]


class MPPIPlanner(TrajectoryPlanner):
    """Model predictive path integral planner over learned dynamics."""

    def __init__(
        self,
        model: DynamicsEnsemble,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        horizon: int = 30,
        candidates: int = 512,
        iterations: int = 5,
        discount: float = 0.99,
        temperature: float = 0.5,
        lambda_: float = 1.0,
    ):
        super().__init__(
            model=model,
            action_low=action_low,
            action_high=action_high,
            horizon=horizon,
            candidates=candidates,
            discount=discount,
            temperature=temperature,
        )
        self.iterations = iterations
        self.lambda_ = lambda_

    def plan(self, obs: torch.Tensor) -> torch.Tensor:
        mean = self._warm_start_mean(obs)
        std = torch.full_like(mean, self.temperature)

        for _ in range(self.iterations):
            noise = torch.randn(
                (obs.shape[0], self.candidates, self.horizon, self.action_dim), device=obs.device, dtype=obs.dtype
            )
            action_sequences = self._clip_action_sequences(mean.unsqueeze(1) + std.unsqueeze(1) * noise)
            returns = self.evaluate_sequences(obs, action_sequences)

            shifted_returns = returns - returns.max(dim=1, keepdim=True).values
            weights = torch.softmax(shifted_returns / max(self.lambda_, 1e-6), dim=1)
            mean = (weights.unsqueeze(-1).unsqueeze(-1) * action_sequences).sum(dim=1)
            centered = action_sequences - mean.unsqueeze(1)
            variance = (weights.unsqueeze(-1).unsqueeze(-1) * centered.square()).sum(dim=1)
            std = variance.sqrt().clamp_min(1e-3)

        self._prev_mean = mean.detach()
        return mean[:, 0, :]


def build_planner(
    planner_name: str,
    model: DynamicsEnsemble,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    horizon: int,
    candidates: int,
    elites: int,
    iterations: int,
    discount: float,
    temperature: float,
    lambda_: float,
) -> TrajectoryPlanner:
    if planner_name == "mppi":
        return MPPIPlanner(
            model=model,
            action_low=action_low,
            action_high=action_high,
            horizon=horizon,
            candidates=candidates,
            iterations=iterations,
            discount=discount,
            temperature=temperature,
            lambda_=lambda_,
        )
    if planner_name == "cem":
        return CEMPlanner(
            model=model,
            action_low=action_low,
            action_high=action_high,
            horizon=horizon,
            candidates=candidates,
            elites=elites,
            iterations=iterations,
            discount=discount,
            temperature=temperature,
        )
    raise ValueError(f"Unsupported planner: {planner_name}")
