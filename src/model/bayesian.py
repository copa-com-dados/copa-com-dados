"""Bayesian Dixon–Coles model backed by Stan posterior draws (.npz)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.constants import MAX_GOALS
from src.model.base import BaseDixonColesMatchModel
from src.model.utils import score_probability_matrix


class BayesianDixonColesModel(BaseDixonColesMatchModel):
    """Dixon-Coles model parameterized by Stan posterior draws.

    Point-estimate methods (get_attack, match_probs, …) use posterior means.
    Access the full posterior via .draws for vectorized tournament simulation.

    Stan parameterization (log-space):
        home_goals ~ Poisson(exp(atk_home - dfn_away + eta))
        away_goals ~ Poisson(exp(atk_away - dfn_home + eta))
    where eta is a global expected-goals offset (not an asymmetric home effect).
    """

    def __init__(self, draws_path: str | Path) -> None:
        loaded = np.load(draws_path)
        self.teams: list[str] = list(loaded["teams"]) if "teams" in loaded.files else []
        self._attack_draws: NDArray = loaded["attack"]  # (n_draws, n_teams)
        self._defense_draws: NDArray = loaded["defense"]  # (n_draws, n_teams)
        self._eta_draws: NDArray = loaded["eta"]  # (n_draws,)
        self._rho_draws: NDArray | None = (
            loaded["rho"] if "rho" in loaded.files else None
        )

        self._beta_home_draws: NDArray | None = (
            loaded["beta_home"] if "beta_home" in loaded.files else None
        )

        self._attack_mean = self._attack_draws.mean(axis=0)
        self._defense_mean = self._defense_draws.mean(axis=0)
        self._eta_mean = float(self._eta_draws.mean())
        self._rho_mean = (
            float(self._rho_draws.mean()) if self._rho_draws is not None else 0.0
        )
        self._beta_home_mean = (
            float(self._beta_home_draws.mean())
            if self._beta_home_draws is not None
            else 0.0
        )
        self._team_idx: dict[str, int] = {t: i for i, t in enumerate(self.teams)}

    @property
    def n_draws(self) -> int:
        return int(self._attack_draws.shape[0])

    @property
    def draws(self) -> dict[str, NDArray]:
        """Full posterior draw arrays for vectorized simulation."""
        out: dict[str, NDArray] = {
            "attack": self._attack_draws,
            "defense": self._defense_draws,
            "eta": self._eta_draws,
        }
        if self._rho_draws is not None:
            out["rho"] = self._rho_draws
        if self._beta_home_draws is not None:
            out["beta_home"] = self._beta_home_draws
        return out

    # ── BaseDixonColesMatchModel interface ───────────────────────────────────

    def get_attack(self, team: str) -> float:
        """Posterior mean attack strength (exp of log-space mean)."""
        return float(np.exp(self._attack_mean[self._team_idx[team]]))

    def get_defense(self, team: str) -> float:
        """Posterior mean defense strength (inverted to match freq convention)."""
        return float(np.exp(-self._defense_mean[self._team_idx[team]]))

    def get_rho(self) -> float:
        return self._rho_mean

    def get_home_effect(self) -> float:
        """Home advantage as a multiplicative factor on expected goals."""
        return float(np.exp(self._beta_home_mean))

    def match_probs(
        self,
        home: str,
        away: str,
        neutral: bool = True,
        max_goals: int = MAX_GOALS,
        lambda_scale: float = 1.0,
        home_boost: float = 0.0,
    ) -> NDArray[np.floating]:
        """Score-probability matrix using posterior-mean Stan parameterization."""
        hi = self._team_idx[home]
        ai = self._team_idx[away]
        home_offset = 0.0
        if not neutral:
            home_offset = self._beta_home_mean
        elif home_boost > 0:
            home_offset = self._beta_home_mean * home_boost
        hl = (
            float(
                np.exp(
                    self._attack_mean[hi]
                    - self._defense_mean[ai]
                    + self._eta_mean
                    + home_offset
                )
            )
            * lambda_scale
        )
        al = (
            float(
                np.exp(self._attack_mean[ai] - self._defense_mean[hi] + self._eta_mean)
            )
            * lambda_scale
        )
        return score_probability_matrix(hl, al, self._rho_mean, max_goals)


def load_draws(path: str | Path) -> dict[str, NDArray]:
    """Load Stan posterior draws from a ``.npz`` file into a plain dict."""
    loaded = np.load(path)
    return {key: loaded[key] for key in loaded.files}
