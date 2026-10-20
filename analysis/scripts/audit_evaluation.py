"""Evaluation of saved cutoff-specific World Cup forecast snapshots.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit

from .audit_snapshots import (
    DEFAULT_START_DATE,
    GOAL_MARGINAL_MAX,
    GRID_MAX,
    SCHEMA_VERSION,
    STAGE_ORDER,
    _available_snapshot_ids,
    _copy_if_changed,
    _mkdir,
    _read_snapshot_table,
    _require_full_dependencies,
    actual_world_cup_results,
    analysis_root,
    ensure_project_imports,
    find_project_root,
    read_calendar,
    snapshot_plan,
    write_json,
)

__all__ = [
    "attach_baselines",
    "backtest_preflight",
    "bracket_hit_rates",
    "build_match_predictions_from_saved",
    "calibration_tables",
    "draw_analysis",
    "evaluate_from_saved",
    "goal_level_tables",
    "match_level_metrics",
    "paired_bootstrap",
    "run_project_backtests",
    "score_prediction_tables",
    "snapshot_information_gain",
    "title_outcome_evaluation",
    "tournament_level_metrics",
]

def _outcome_label(home_score: float, away_score: float) -> str:
    if home_score > away_score:
        return "home_win"
    if home_score == away_score:
        return "draw"
    return "away_win"


def _safe_probability(value: float) -> float:
    return float(np.clip(value, 1e-12, 1.0 - 1e-12))


def _snapshot_before_match(
    match_date: pd.Timestamp,
    plan: pd.DataFrame,
    available_ids: set[str],
) -> str | None:
    """Use a strictly earlier date because match-level timestamps are unavailable."""
    eligible = plan[
        (pd.to_datetime(plan["cutoff_date"]) < pd.Timestamp(match_date).normalize())
        & plan["snapshot_id"].isin(available_ids)
    ]
    if eligible.empty:
        return None
    return str(eligible.iloc[-1]["snapshot_id"])


def _summary_lookup(summary: pd.DataFrame) -> dict[tuple[str, str, str], pd.Series]:
    return {
        (str(row.snapshot_id), str(row.home_team_display), str(row.away_team_display)): row
        for _, row in summary.iterrows()
    }


def _scoreline_lookup(scorelines: pd.DataFrame) -> dict[tuple[str, str, str, int, int], float]:
    return {
        (
            str(row.snapshot_id),
            str(row.home_team_display),
            str(row.away_team_display),
            int(row.home_goals),
            int(row.away_goals),
        ): float(row.probability)
        for row in scorelines.itertuples(index=False)
    }


def build_match_predictions_from_saved(project_root: str | Path = ".") -> pd.DataFrame:
    """Match each observed result to its latest strictly pre-match snapshot.

    The function reads saved snapshot tables only; it performs no refitting or
    posterior sampling. A pre-tournament counterpart is kept for every match
    so that update value can be evaluated on identical observations.
    """
    root = ensure_project_imports(project_root)
    from src.constants import TEAM_MAP_EN_TO_PT

    available = _available_snapshot_ids(root)
    if not available:
        raise FileNotFoundError(
            "No complete snapshot tables were found. Run Notebook 1 in FULL mode first."
        )
    plan = snapshot_plan(root)
    ordered_available = plan[plan["snapshot_id"].isin(available)]
    summary = pd.concat(
        [_read_snapshot_table(root, snapshot_id, "matchup_summary") for snapshot_id in available],
        ignore_index=True,
    )
    scorelines = pd.concat(
        [_read_snapshot_table(root, snapshot_id, "scoreline_probabilities") for snapshot_id in available],
        ignore_index=True,
    )
    lookup = _summary_lookup(summary)
    score_lookup = _scoreline_lookup(scorelines)
    pre_candidates = ordered_available[
        ordered_available["snapshot_id"].isin({"pre_tournament_post_fifa", "pre_fifa"})
    ]
    if pre_candidates.empty:
        raise FileNotFoundError("A pre-tournament snapshot is required for update evaluation.")
    pre_snapshot = str(pre_candidates.iloc[-1]["snapshot_id"])

    actual = actual_world_cup_results(root)
    actual = actual[actual["score_observed"]].copy()
    rows: list[dict[str, Any]] = []
    for row in actual.itertuples(index=False):
        match_id = f"{pd.Timestamp(row.date).date()}::{row.home_team}::{row.away_team}"
        asof_snapshot = _snapshot_before_match(row.date, plan, set(available))
        for prediction_set, snapshot_id in (
            ("as_of", asof_snapshot),
            ("pre_tournament", pre_snapshot),
        ):
            if snapshot_id is None:
                continue
            home_display = TEAM_MAP_EN_TO_PT.get(row.home_team, row.home_team)
            away_display = TEAM_MAP_EN_TO_PT.get(row.away_team, row.away_team)
            direct = lookup.get((snapshot_id, home_display, away_display))
            reverse = lookup.get((snapshot_id, away_display, home_display))
            if direct is None and reverse is None:
                continue
            reversed_orientation = direct is None
            source = reverse if reversed_orientation else direct
            assert source is not None
            if reversed_orientation:
                p_home = float(source.away_win_probability)
                p_draw = float(source.draw_probability)
                p_away = float(source.home_win_probability)
                expected_home = float(source.expected_away_goals)
                expected_away = float(source.expected_home_goals)
                score_key = (
                    snapshot_id,
                    away_display,
                    home_display,
                    int(row.away_score),
                    int(row.home_score),
                )
            else:
                p_home = float(source.home_win_probability)
                p_draw = float(source.draw_probability)
                p_away = float(source.away_win_probability)
                expected_home = float(source.expected_home_goals)
                expected_away = float(source.expected_away_goals)
                score_key = (
                    snapshot_id,
                    home_display,
                    away_display,
                    int(row.home_score),
                    int(row.away_score),
                )
            within_grid = row.home_score <= GRID_MAX and row.away_score <= GRID_MAX
            score_probability = (
                score_lookup.get(score_key, np.nan)
                if within_grid
                else float(source.residual_probability_mass)
            )
            rows.append(
                {
                    "match_id": match_id,
                    "prediction_set": prediction_set,
                    "snapshot_id": snapshot_id,
                    "cutoff_date": str(source.cutoff_date),
                    "match_date": pd.Timestamp(row.date),
                    "calendar_phase": row.calendar_phase,
                    "calendar_round": row.calendar_round,
                    "home_team_en": row.home_team,
                    "away_team_en": row.away_team,
                    "home_team_display": home_display,
                    "away_team_display": away_display,
                    "home_score": int(row.home_score),
                    "away_score": int(row.away_score),
                    "observed_outcome": _outcome_label(row.home_score, row.away_score),
                    "home_win_probability": p_home,
                    "draw_probability": p_draw,
                    "away_win_probability": p_away,
                    "expected_home_goals": expected_home,
                    "expected_away_goals": expected_away,
                    "scoreline_probability": score_probability,
                    "scoreline_is_censored": not within_grid,
                    "orientation_reversed_for_lookup": reversed_orientation,
                }
            )
    return pd.DataFrame(rows).sort_values(["match_date", "match_id", "prediction_set"]).reset_index(drop=True)


def _ranking_logistic_parameters(
    project_root: Path,
    cutoff_date: str,
) -> tuple[dict[str, float], float, float, tuple[float, float, float]]:
    """Fit a transparent static-FIFA-ranking three-outcome baseline as of cutoff."""
    ensure_project_imports(project_root)
    from src.data import load_ranking_priors, prepare_cycle_data

    training, teams, _ = prepare_cycle_data(
        str(project_root / "data" / "results.csv"),
        DEFAULT_START_DATE,
        end_date=cutoff_date,
        apply_decay=True,
    )
    strengths = load_ranking_priors(project_root / "data" / "raw" / "fifa_ranking_2022.csv", teams)
    strength = dict(zip(teams, np.asarray(strengths, dtype=float), strict=True))
    home = training["home_score"].to_numpy(dtype=float)
    away = training["away_score"].to_numpy(dtype=float)
    draw_rate = float(np.mean(home == away))
    outcome_rates = (
        float(np.mean(home > away)),
        draw_rate,
        float(np.mean(home < away)),
    )
    non_draw = home != away
    deltas = np.array(
        [
            strength.get(h, 0.0) - strength.get(a, 0.0)
            for h, a in zip(training["home_team"], training["away_team"], strict=True)
        ],
        dtype=float,
    )[non_draw]
    outcomes = (home[non_draw] > away[non_draw]).astype(float)
    if len(deltas) == 0 or np.allclose(deltas, 0.0):
        slope = 0.0
    else:
        def objective(value: float) -> float:
            probability = np.clip(expit(value * deltas), 1e-12, 1.0 - 1e-12)
            return -float(np.sum(outcomes * np.log(probability) + (1.0 - outcomes) * np.log(1.0 - probability)))

        slope = float(minimize_scalar(objective, bounds=(-10.0, 10.0), method="bounded").x)
    return strength, slope, draw_rate, outcome_rates


def attach_baselines(predictions: pd.DataFrame, project_root: str | Path = ".") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach uniform, historical-frequency and static-ranking-logistic baselines."""
    root = find_project_root(project_root)
    out = predictions.copy()
    parameter_rows: list[dict[str, Any]] = []
    cache: dict[str, tuple[dict[str, float], float, float, tuple[float, float, float]]] = {}
    for cutoff in sorted(out["cutoff_date"].dropna().unique()):
        if cutoff not in cache:
            cache[cutoff] = _ranking_logistic_parameters(root, str(cutoff))
        strength, slope, draw_rate, historical = cache[cutoff]
        mask = out["cutoff_date"] == cutoff
        deltas = out.loc[mask, "home_team_en"].map(strength).fillna(0.0) - out.loc[
            mask, "away_team_en"
        ].map(strength).fillna(0.0)
        conditional_home = expit(slope * deltas.to_numpy(dtype=float))
        out.loc[mask, "uniform_home_win"] = 1.0 / 3.0
        out.loc[mask, "uniform_draw"] = 1.0 / 3.0
        out.loc[mask, "uniform_away_win"] = 1.0 / 3.0
        out.loc[mask, "historical_home_win"] = historical[0]
        out.loc[mask, "historical_draw"] = historical[1]
        out.loc[mask, "historical_away_win"] = historical[2]
        out.loc[mask, "ranking_logistic_draw"] = draw_rate
        out.loc[mask, "ranking_logistic_home_win"] = (1.0 - draw_rate) * conditional_home
        out.loc[mask, "ranking_logistic_away_win"] = (1.0 - draw_rate) * (1.0 - conditional_home)
        parameter_rows.append(
            {
                "cutoff_date": cutoff,
                "baseline": "static_fifa_ranking_logistic",
                "slope": slope,
                "draw_rate": draw_rate,
                "historical_home_win_rate": historical[0],
                "historical_draw_rate": historical[1],
                "historical_away_win_rate": historical[2],
            }
        )
    return out, pd.DataFrame(parameter_rows)


def _metric_components(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute proper-score contributions for one three-outcome forecast system."""
    outcome_columns = ("home_win", "draw", "away_win")
    probabilities = np.column_stack(
        [frame[f"{prefix}_{column}"].to_numpy(dtype=float) for column in outcome_columns]
    )
    observed = np.column_stack(
        [(frame["observed_outcome"] == column).to_numpy(dtype=float) for column in outcome_columns]
    )
    out = pd.DataFrame(index=frame.index)
    out["brier"] = np.sum((probabilities - observed) ** 2, axis=1)
    # RPS is evaluated in the ordered Away--Draw--Home convention.
    ordered_probabilities = probabilities[:, [2, 1, 0]]
    ordered_observed = observed[:, [2, 1, 0]]
    out["rps"] = np.sum(
        (np.cumsum(ordered_probabilities, axis=1)[:, :-1] - np.cumsum(ordered_observed, axis=1)[:, :-1]) ** 2,
        axis=1,
    ) / 2.0
    selected = np.sum(probabilities * observed, axis=1)
    out["log_score"] = -np.log(np.clip(selected, 1e-12, 1.0))
    for column, index in zip(outcome_columns, range(3), strict=True):
        out[f"brier_{column}"] = (probabilities[:, index] - observed[:, index]) ** 2
    return out


def match_level_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aggregate proper scores and per-match contributions."""
    systems = {
        "model": "",
        "uniform": "uniform",
        "historical_frequency": "historical",
        "static_fifa_ranking_logistic": "ranking_logistic",
    }
    detailed_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for prediction_set, group in predictions.groupby("prediction_set", sort=False):
        for system, prefix in systems.items():
            if system == "model":
                temp = group.copy()
                temp = temp.rename(
                    columns={
                        "home_win_probability": "_home_win",
                        "draw_probability": "_draw",
                        "away_win_probability": "_away_win",
                    }
                )
                components = _metric_components(
                    temp,
                    "",
                )
            else:
                components = _metric_components(group, prefix)
            detailed = group[["match_id", "prediction_set", "observed_outcome"]].copy()
            detailed["system"] = system
            detailed = pd.concat([detailed.reset_index(drop=True), components.reset_index(drop=True)], axis=1)
            detailed_frames.append(detailed)
            for metric in ("brier", "rps", "log_score"):
                metric_rows.append(
                    {
                        "record_type": "metric",
                        "prediction_set": prediction_set,
                        "system": system,
                        "metric": metric,
                        "n_matches": int(len(detailed)),
                        "mean": float(detailed[metric].mean()),
                        "standard_error": float(detailed[metric].sem(ddof=1)),
                    }
                )
    detailed_all = pd.concat(detailed_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    model_metrics = metrics[metrics["system"] == "model"].set_index(["prediction_set", "metric"])
    skill_rows: list[dict[str, Any]] = []
    for row in metrics[metrics["system"] != "model"].itertuples(index=False):
        baseline_mean = float(row.mean)
        model_mean = float(model_metrics.loc[(row.prediction_set, row.metric), "mean"])
        skill_rows.append(
            {
                "record_type": "skill",
                "prediction_set": row.prediction_set,
                "system": "model",
                "baseline": row.system,
                "metric": row.metric,
                "skill_score": 1.0 - model_mean / baseline_mean if baseline_mean > 0 else np.nan,
            }
        )
    return pd.concat([metrics, pd.DataFrame(skill_rows)], ignore_index=True, sort=False), detailed_all


def calibration_tables(predictions: pd.DataFrame, n_bins: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute reliability bins and explicit expected calibration error."""
    rows: list[dict[str, Any]] = []
    ece_rows: list[dict[str, Any]] = []
    for prediction_set, group in predictions.groupby("prediction_set", sort=False):
        for outcome, column in (
            ("home_win", "home_win_probability"),
            ("draw", "draw_probability"),
            ("away_win", "away_win_probability"),
        ):
            probabilities = group[column].to_numpy(dtype=float)
            observed = (group["observed_outcome"] == outcome).to_numpy(dtype=float)
            bins = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
            weighted_error = 0.0
            for bin_index in range(n_bins):
                mask = bins == bin_index
                if not mask.any():
                    continue
                predicted_mean = float(probabilities[mask].mean())
                observed_rate = float(observed[mask].mean())
                count = int(mask.sum())
                gap = abs(predicted_mean - observed_rate)
                weighted_error += count / len(group) * gap
                rows.append(
                    {
                        "prediction_set": prediction_set,
                        "outcome": outcome,
                        "bin": bin_index,
                        "bin_lower": bin_index / n_bins,
                        "bin_upper": (bin_index + 1) / n_bins,
                        "n_matches": count,
                        "mean_predicted_probability": predicted_mean,
                        "observed_frequency": observed_rate,
                        "absolute_gap": gap,
                    }
                )
            ece_rows.append(
                {
                    "prediction_set": prediction_set,
                    "outcome": outcome,
                    "n_bins": n_bins,
                    "expected_calibration_error": weighted_error,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(ece_rows)


def draw_analysis(predictions: pd.DataFrame, match_contributions: pd.DataFrame) -> pd.DataFrame:
    """Summarise draw calibration and score contributions by realised outcome."""
    model = match_contributions[match_contributions["system"] == "model"]
    merged = predictions.merge(
        model[["match_id", "prediction_set", "brier", "brier_draw"]],
        on=["match_id", "prediction_set"],
        how="inner",
    )
    rows: list[dict[str, Any]] = []
    for (prediction_set, outcome), group in merged.groupby(
        ["prediction_set", "observed_outcome"], sort=False
    ):
        rows.append(
            {
                "prediction_set": prediction_set,
                "observed_outcome": outcome,
                "n_matches": int(len(group)),
                "observed_draw_frequency": float(np.mean(group["observed_outcome"] == "draw")),
                "mean_predicted_draw_probability": float(group["draw_probability"].mean()),
                "mean_multiclass_brier": float(group["brier"].mean()),
                "mean_draw_component_of_brier": float(group["brier_draw"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _marginal_lookup(goal_marginals: pd.DataFrame) -> dict[tuple[str, str, str, str], dict[str, float]]:
    output: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for keys, group in goal_marginals.groupby(
        ["snapshot_id", "home_team_display", "away_team_display", "side"], sort=False
    ):
        output[tuple(str(value) for value in keys)] = dict(
            zip(group["category"].astype(str), group["probability"], strict=True)
        )
    return output


def _randomized_pit(observed: int, probabilities: Mapping[str, float], rng: np.random.Generator) -> tuple[float, bool]:
    """Randomised PIT for a saved discrete marginal; tail values are censored."""
    if observed <= GOAL_MARGINAL_MAX:
        previous = sum(probabilities.get(str(goal), 0.0) for goal in range(observed))
        mass = probabilities.get(str(observed), 0.0)
        return previous + rng.uniform() * mass, False
    previous = sum(probabilities.get(str(goal), 0.0) for goal in range(GOAL_MARGINAL_MAX + 1))
    return previous + rng.uniform() * probabilities.get(f">{GOAL_MARGINAL_MAX}", 0.0), True


def goal_level_tables(
    predictions: pd.DataFrame,
    project_root: str | Path = ".",
    seed: int = 20261014,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate score-grid likelihood, xG errors, PIT and goal dispersion."""
    root = find_project_root(project_root)
    available = _available_snapshot_ids(root)
    marginal_data = pd.concat(
        [_read_snapshot_table(root, snapshot_id, "goal_marginals") for snapshot_id in available],
        ignore_index=True,
    )
    scoreline_data = pd.concat(
        [_read_snapshot_table(root, snapshot_id, "scoreline_probabilities") for snapshot_id in available],
        ignore_index=True,
    )
    lookup = _marginal_lookup(marginal_data)
    rng = np.random.default_rng(seed)
    detailed = predictions.copy()
    detailed["censored_scoreline_log_score"] = -np.log(
        detailed["scoreline_probability"].clip(lower=1e-12)
    )
    detailed["absolute_home_goal_error"] = np.abs(
        detailed["home_score"] - detailed["expected_home_goals"]
    )
    detailed["absolute_away_goal_error"] = np.abs(
        detailed["away_score"] - detailed["expected_away_goals"]
    )
    home_pit: list[float] = []
    away_pit: list[float] = []
    censored: list[bool] = []
    home_marginal_variance: list[float] = []
    away_marginal_variance: list[float] = []
    for row in detailed.itertuples(index=False):
        direct_home = lookup.get(
            (str(row.snapshot_id), str(row.home_team_display), str(row.away_team_display), "home")
        )
        direct_away = lookup.get(
            (str(row.snapshot_id), str(row.home_team_display), str(row.away_team_display), "away")
        )
        if direct_home is None or direct_away is None:
            reverse_home = lookup.get(
                (str(row.snapshot_id), str(row.away_team_display), str(row.home_team_display), "away")
            )
            reverse_away = lookup.get(
                (str(row.snapshot_id), str(row.away_team_display), str(row.home_team_display), "home")
            )
            if reverse_home is not None and reverse_away is not None:
                direct_home, direct_away = reverse_home, reverse_away
        if direct_home is None or direct_away is None:
            home_pit.append(np.nan)
            away_pit.append(np.nan)
            censored.append(True)
            home_marginal_variance.append(np.nan)
            away_marginal_variance.append(np.nan)
            continue
        hp, hc = _randomized_pit(int(row.home_score), direct_home, rng)
        ap, ac = _randomized_pit(int(row.away_score), direct_away, rng)
        home_pit.append(hp)
        away_pit.append(ap)
        censored.append(hc or ac)
        support = np.arange(GOAL_MARGINAL_MAX + 2, dtype=float)
        home_mass = np.array(
            [direct_home.get(str(goal), 0.0) for goal in range(GOAL_MARGINAL_MAX + 1)]
            + [direct_home.get(f">{GOAL_MARGINAL_MAX}", 0.0)]
        )
        away_mass = np.array(
            [direct_away.get(str(goal), 0.0) for goal in range(GOAL_MARGINAL_MAX + 1)]
            + [direct_away.get(f">{GOAL_MARGINAL_MAX}", 0.0)]
        )
        home_mean = float(np.sum(support * home_mass))
        away_mean = float(np.sum(support * away_mass))
        home_marginal_variance.append(float(np.sum((support - home_mean) ** 2 * home_mass)))
        away_marginal_variance.append(float(np.sum((support - away_mean) ** 2 * away_mass)))
    detailed["home_randomized_pit"] = home_pit
    detailed["away_randomized_pit"] = away_pit
    detailed["pit_tail_censored"] = censored
    detailed["home_predictive_variance_lower_bound"] = home_marginal_variance
    detailed["away_predictive_variance_lower_bound"] = away_marginal_variance

    score_frequency_rows: list[dict[str, Any]] = []
    for prediction_set, group in detailed.groupby("prediction_set", sort=False):
        category = np.where(
            (group["home_score"] <= GRID_MAX) & (group["away_score"] <= GRID_MAX),
            group["home_score"].astype(str) + "-" + group["away_score"].astype(str),
            f"outside_0_{GRID_MAX}",
        )
        observed_counts = pd.Series(category).value_counts()
        expected_counts = {
            f"{home}-{away}": 0.0
            for home in range(GRID_MAX + 1)
            for away in range(GRID_MAX + 1)
        }
        expected_counts[f"outside_0_{GRID_MAX}"] = 0.0
        for row in group.itertuples(index=False):
            subset = scoreline_data[
                (scoreline_data["snapshot_id"] == row.snapshot_id)
                & (scoreline_data["home_team_display"] == row.home_team_display)
                & (scoreline_data["away_team_display"] == row.away_team_display)
            ]
            reverse = bool(row.orientation_reversed_for_lookup)
            if subset.empty:
                subset = scoreline_data[
                    (scoreline_data["snapshot_id"] == row.snapshot_id)
                    & (scoreline_data["home_team_display"] == row.away_team_display)
                    & (scoreline_data["away_team_display"] == row.home_team_display)
                ]
                reverse = True
            for score_row in subset.itertuples(index=False):
                home_goals = int(score_row.away_goals) if reverse else int(score_row.home_goals)
                away_goals = int(score_row.home_goals) if reverse else int(score_row.away_goals)
                expected_counts[f"{home_goals}-{away_goals}"] += float(score_row.probability)
            expected_counts[f"outside_0_{GRID_MAX}"] += max(
                0.0, 1.0 - float(subset["probability"].sum())
            )
        for label in sorted(set(observed_counts.index).union(expected_counts)):
            count = int(observed_counts.get(label, 0))
            score_frequency_rows.append(
                {
                    "prediction_set": prediction_set,
                    "scoreline_category": label,
                    "observed_count": count,
                    "observed_frequency": float(count / len(group)),
                    "expected_count": float(expected_counts.get(label, 0.0)),
                    "expected_frequency": float(expected_counts.get(label, 0.0) / len(group)),
                }
            )

    dispersion_rows: list[dict[str, Any]] = []
    for prediction_set, group in detailed.groupby("prediction_set", sort=False):
        observed_goals = np.concatenate(
            [group["home_score"].to_numpy(dtype=float), group["away_score"].to_numpy(dtype=float)]
        )
        expected_goals = np.concatenate(
            [
                group["expected_home_goals"].to_numpy(dtype=float),
                group["expected_away_goals"].to_numpy(dtype=float),
            ]
        )
        predicted_variance = np.concatenate(
            [
                group["home_predictive_variance_lower_bound"].to_numpy(dtype=float),
                group["away_predictive_variance_lower_bound"].to_numpy(dtype=float),
            ]
        )
        dispersion_rows.append(
            {
                "prediction_set": prediction_set,
                "n_goal_observations": int(len(observed_goals)),
                "observed_mean": float(np.mean(observed_goals)),
                "observed_variance": float(np.var(observed_goals, ddof=1)),
                "observed_dispersion_index": float(np.var(observed_goals, ddof=1) / np.mean(observed_goals)),
                "mean_predicted_goal_rate": float(np.mean(expected_goals)),
                "mean_predictive_variance_lower_bound": float(np.nanmean(predicted_variance)),
                "predictive_dispersion_index_lower_bound": float(
                    np.nanmean(predicted_variance) / np.mean(expected_goals)
                ),
                "mean_absolute_goal_error": float(
                    np.mean(
                        np.concatenate(
                            [
                                group["absolute_home_goal_error"].to_numpy(),
                                group["absolute_away_goal_error"].to_numpy(),
                            ]
                        )
                    )
                ),
                "mean_censored_scoreline_log_score": float(group["censored_scoreline_log_score"].mean()),
                "n_scoreline_tail_censored": int(group["scoreline_is_censored"].sum()),
                "n_pit_tail_censored": int(group["pit_tail_censored"].sum()),
            }
        )
    return detailed, pd.DataFrame(score_frequency_rows), pd.DataFrame(dispersion_rows)


def _modal_scoreline_lookup(
    scorelines: pd.DataFrame,
) -> dict[tuple[str, str, str], tuple[int, int, float]]:
    """Return the most-probable published scoreline for each oriented pairing.

    The public export contains exact cells from 0--0 through 4--4 plus a
    residual tail.  The modal scoreline is therefore explicitly a modal
    \\emph{published-grid} scoreline, not an assertion that the full predictive
    distribution has no score above four goals.
    """
    output: dict[tuple[str, str, str], tuple[int, int, float]] = {}
    groups = scorelines.groupby(
        ["snapshot_id", "home_team_display", "away_team_display"], sort=False
    )
    for keys, group in groups:
        ordered = group.sort_values(
            ["probability", "home_goals", "away_goals"],
            ascending=[False, True, True],
        )
        selected = ordered.iloc[0]
        output[tuple(str(value) for value in keys)] = (
            int(selected.home_goals),
            int(selected.away_goals),
            float(selected.probability),
        )
    return output


def score_prediction_tables(
    predictions: pd.DataFrame,
    project_root: str | Path = ".",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate expected and modal score forecasts against realised scores.

    Proper three-way outcome scores remain the primary comparison because they
    assess the full probability distribution.  These diagnostics answer the
    complementary descriptive question: how close were expected goals and the
    most-probable displayed scoreline to the realised score?  They do not turn
    a probabilistic score forecast into a point-forecast selection criterion.
    """
    root = find_project_root(project_root)
    available = _available_snapshot_ids(root)
    scoreline_data = pd.concat(
        [_read_snapshot_table(root, snapshot_id, "scoreline_probabilities") for snapshot_id in available],
        ignore_index=True,
    )
    mode_lookup = _modal_scoreline_lookup(scoreline_data)
    detailed = predictions.copy()
    modal_home: list[float] = []
    modal_away: list[float] = []
    modal_probability: list[float] = []
    for row in detailed.itertuples(index=False):
        direct = mode_lookup.get(
            (str(row.snapshot_id), str(row.home_team_display), str(row.away_team_display))
        )
        reverse = False
        if direct is None:
            direct = mode_lookup.get(
                (str(row.snapshot_id), str(row.away_team_display), str(row.home_team_display))
            )
            reverse = True
        if direct is None:
            modal_home.append(np.nan)
            modal_away.append(np.nan)
            modal_probability.append(np.nan)
            continue
        home_goals, away_goals, probability = direct
        if reverse:
            home_goals, away_goals = away_goals, home_goals
        modal_home.append(float(home_goals))
        modal_away.append(float(away_goals))
        modal_probability.append(probability)

    detailed["modal_home_goals"] = modal_home
    detailed["modal_away_goals"] = modal_away
    detailed["modal_scoreline_probability"] = modal_probability
    detailed["expected_home_goal_abs_error"] = np.abs(
        detailed["home_score"] - detailed["expected_home_goals"]
    )
    detailed["expected_away_goal_abs_error"] = np.abs(
        detailed["away_score"] - detailed["expected_away_goals"]
    )
    detailed["expected_total_goal_abs_error"] = np.abs(
        (detailed["home_score"] + detailed["away_score"])
        - (detailed["expected_home_goals"] + detailed["expected_away_goals"])
    )
    detailed["expected_goal_difference_abs_error"] = np.abs(
        (detailed["home_score"] - detailed["away_score"])
        - (detailed["expected_home_goals"] - detailed["expected_away_goals"])
    )
    detailed["modal_home_goal_abs_error"] = np.abs(
        detailed["home_score"] - detailed["modal_home_goals"]
    )
    detailed["modal_away_goal_abs_error"] = np.abs(
        detailed["away_score"] - detailed["modal_away_goals"]
    )
    detailed["modal_total_goal_abs_error"] = np.abs(
        (detailed["home_score"] + detailed["away_score"])
        - (detailed["modal_home_goals"] + detailed["modal_away_goals"])
    )
    detailed["modal_goal_difference_abs_error"] = np.abs(
        (detailed["home_score"] - detailed["away_score"])
        - (detailed["modal_home_goals"] - detailed["modal_away_goals"])
    )
    detailed["modal_exact_scoreline"] = (
        (detailed["home_score"] == detailed["modal_home_goals"])
        & (detailed["away_score"] == detailed["modal_away_goals"])
    )
    detailed["modal_both_teams_within_one_goal"] = (
        (detailed["modal_home_goal_abs_error"] <= 1.0)
        & (detailed["modal_away_goal_abs_error"] <= 1.0)
    )

    summary_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    for prediction_set, group in detailed.groupby("prediction_set", sort=False):
        summary_rows.append(
            {
                "prediction_set": prediction_set,
                "n_matches": int(len(group)),
                "expected_team_goal_mae": float(
                    group[["expected_home_goal_abs_error", "expected_away_goal_abs_error"]]
                    .to_numpy(dtype=float)
                    .mean()
                ),
                "expected_total_goals_mae": float(group["expected_total_goal_abs_error"].mean()),
                "expected_goal_difference_mae": float(
                    group["expected_goal_difference_abs_error"].mean()
                ),
                "modal_team_goal_mae": float(
                    group[["modal_home_goal_abs_error", "modal_away_goal_abs_error"]]
                    .to_numpy(dtype=float)
                    .mean()
                ),
                "modal_total_goals_mae": float(group["modal_total_goal_abs_error"].mean()),
                "modal_goal_difference_mae": float(
                    group["modal_goal_difference_abs_error"].mean()
                ),
                "modal_exact_scoreline_rate": float(group["modal_exact_scoreline"].mean()),
                "modal_both_teams_within_one_goal_rate": float(
                    group["modal_both_teams_within_one_goal"].mean()
                ),
                "note": (
                    "Modal metrics use the most-probable 0--4 grid cell; expected metrics use "
                    "the model's uncensored expected-goal outputs."
                ),
            }
        )
        for phase, phase_group in group.groupby("calendar_phase", sort=False):
            phase_rows.append(
                {
                    "prediction_set": prediction_set,
                    "calendar_phase": phase,
                    "n_matches": int(len(phase_group)),
                    "modal_team_goal_mae": float(
                        phase_group[["modal_home_goal_abs_error", "modal_away_goal_abs_error"]]
                        .to_numpy(dtype=float)
                        .mean()
                    ),
                    "modal_exact_scoreline_rate": float(
                        phase_group["modal_exact_scoreline"].mean()
                    ),
                }
            )
    return detailed, pd.DataFrame(summary_rows), pd.DataFrame(phase_rows)


def actual_stage_targets(project_root: str | Path = ".") -> pd.DataFrame:
    """Derive realised stage membership from the committed 2026 calendar.

    Membership in a scheduled stage is observable from the calendar. Champion
    status is filled only when the final winner is verifiable from the supplied
    result and tiebreak files.
    """
    root = ensure_project_imports(project_root)
    from src.constants import GROUPS

    calendar = read_calendar(root)
    stage_map = {
        "Fase de 32": "round_of_32",
        "Oitavas de final": "round_of_16",
        "Quartas de final": "quarterfinals",
        "Semifinal": "semifinals",
        "Final": "final",
    }
    teams = [team for group in GROUPS.values() for team in group]
    knockout = _observed_knockout_winners(root)
    final = knockout[knockout["calendar_phase"] == "Final"]
    champion = None if final.empty else final.iloc[0]["observed_winner_en"]
    rows: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        if stage == "champion":
            for team in teams:
                observed = np.nan if champion is None or pd.isna(champion) else float(team == champion)
                rows.append({"team_en": team, "stage": stage, "observed": observed})
            continue
        scheduled = calendar[calendar["fase"].map(stage_map) == stage]
        participants = set(scheduled["home_team_en"]).union(scheduled["away_team_en"])
        for team in teams:
            rows.append(
                {"team_en": team, "stage": stage, "observed": float(team in participants)}
            )
    return pd.DataFrame(rows)


def tournament_level_metrics(project_root: str | Path = ".") -> pd.DataFrame:
    """Evaluate saved stage-reach probabilities without refitting any model."""
    root = find_project_root(project_root)
    available = _available_snapshot_ids(root)
    targets = actual_stage_targets(root)
    rows: list[dict[str, Any]] = []
    for snapshot_id in available:
        probabilities = _read_snapshot_table(root, snapshot_id, "stage_probabilities")
        merged = probabilities.merge(targets, on=["team_en", "stage"], how="left")
        for stage, group in merged.groupby("stage", sort=False):
            observed = group["observed"].dropna()
            if observed.empty:
                rows.append(
                    {
                        "snapshot_id": snapshot_id,
                        "stage": stage,
                        "n_teams": 0,
                        "stage_brier": np.nan,
                        "stage_log_loss": np.nan,
                        "expected_number_of_teams": np.nan,
                        "realized_number_of_teams": np.nan,
                        "note": "Outcome not verifiable from supplied files.",
                    }
                )
                continue
            evaluated = group[group["observed"].notna()].copy()
            p = evaluated["probability"].to_numpy(dtype=float)
            y = evaluated["observed"].to_numpy(dtype=float)
            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "stage": stage,
                    "n_teams": int(len(evaluated)),
                    "stage_brier": float(np.mean((p - y) ** 2)),
                    "stage_log_loss": float(
                        np.mean(-(y * np.log(np.clip(p, 1e-12, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p, 1e-12, 1.0))))
                    ),
                    "expected_number_of_teams": float(p.sum()),
                    "realized_number_of_teams": float(y.sum()),
                    "note": "Champion is evaluated when the final winner is verifiable from source files.",
                }
            )
    return pd.DataFrame(rows)


def title_outcome_evaluation(project_root: str | Path = ".") -> pd.DataFrame:
    """Report champion probability and rank only when the source verifies a winner."""
    root = find_project_root(project_root)
    knockout = _observed_knockout_winners(root)
    final = knockout[knockout["calendar_phase"] == "Final"]
    winner = None if final.empty else final.iloc[0]["observed_winner_en"]
    if winner is None or pd.isna(winner):
        return pd.DataFrame(
            [
                {
                    "snapshot_id": np.nan,
                    "status": "not_evaluable",
                    "champion_en": np.nan,
                    "champion_probability": np.nan,
                    "champion_probability_rank": np.nan,
                    "note": "The supplied final score/winner is missing; no champion probability or rank is reported.",
                }
            ]
        )
    rows: list[dict[str, Any]] = []
    for snapshot_id in _available_snapshot_ids(root):
        stage = _read_snapshot_table(root, snapshot_id, "stage_probabilities")
        title = stage[stage["stage"] == "champion"].sort_values("probability", ascending=False).reset_index(drop=True)
        selected = title[title["team_en"] == winner]
        if selected.empty:
            continue
        row = selected.iloc[0]
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "status": "evaluable",
                "champion_en": winner,
                "champion_probability": float(row.probability),
                "champion_probability_rank": int(selected.index[0] + 1),
                "note": "Rank 1 is the largest title probability in that snapshot.",
            }
        )
    return pd.DataFrame(rows)


def _observed_knockout_winners(project_root: Path) -> pd.DataFrame:
    """Return known knockout winners using scores, shootouts and extra-time file."""
    actual = actual_world_cup_results(project_root)
    actual = actual[actual["calendar_phase"].isin(
        ["Fase de 32", "Oitavas de final", "Quartas de final", "Semifinal", "Final", "Disputa de 3º lugar"]
    )].copy()
    shootouts = pd.read_csv(project_root / "data" / "shootouts.csv")
    shootouts["date"] = pd.to_datetime(shootouts["date"], errors="coerce")
    aet_path = project_root / "data" / "aet_winners.csv"
    aet = pd.read_csv(aet_path) if aet_path.is_file() else pd.DataFrame(columns=["date", "home_team", "away_team", "winner"])
    aet["date"] = pd.to_datetime(aet["date"], errors="coerce")
    winners: list[str | None] = []
    for row in actual.itertuples(index=False):
        if not row.score_observed:
            winners.append(None)
            continue
        if row.home_score > row.away_score:
            winners.append(row.home_team)
            continue
        if row.home_score < row.away_score:
            winners.append(row.away_team)
            continue
        winner: str | None = None
        for decisions in (shootouts, aet):
            found = decisions[
                (decisions["date"] == row.date)
                & (decisions["home_team"] == row.home_team)
                & (decisions["away_team"] == row.away_team)
            ]
            if not found.empty:
                winner = str(found.iloc[0]["winner"])
                break
        winners.append(winner)
    actual["observed_winner_en"] = winners
    return actual


def bracket_hit_rates(project_root: str | Path = ".") -> pd.DataFrame:
    """Compare each deterministic display bracket with realised knockout pairs."""
    root = ensure_project_imports(project_root)
    from src.constants import TEAM_MAP_EN_TO_PT

    phase_by_round = {
        "R32": "Fase de 32",
        "Oitavas": "Oitavas de final",
        "Quartas": "Quartas de final",
        "Semifinal": "Semifinal",
        "Final": "Final",
        "3º Lugar": "Disputa de 3º lugar",
    }
    actual = _observed_knockout_winners(root)
    rows: list[dict[str, Any]] = []
    for snapshot_id in _available_snapshot_ids(root):
        bracket = _read_snapshot_table(root, snapshot_id, "projected_bracket")
        for round_label, displayed in bracket.groupby("round_label", sort=False):
            phase = phase_by_round.get(str(round_label))
            realised = actual[actual["calendar_phase"] == phase] if phase else actual.iloc[0:0]
            actual_pairs = {
                frozenset(
                    (
                        TEAM_MAP_EN_TO_PT.get(row.home_team, row.home_team),
                        TEAM_MAP_EN_TO_PT.get(row.away_team, row.away_team),
                    )
                ): row
                for row in realised.itertuples(index=False)
            }
            pair_hits = 0
            advancement_evaluable = 0
            advancement_hits = 0
            for row in displayed.itertuples(index=False):
                pair = frozenset((str(row.home_team), str(row.away_team)))
                actual_row = actual_pairs.get(pair)
                if actual_row is None:
                    continue
                pair_hits += 1
                if pd.notna(actual_row.observed_winner_en):
                    predicted = row.home_team if row.winner == "home" else row.away_team
                    observed_display = TEAM_MAP_EN_TO_PT.get(actual_row.observed_winner_en, actual_row.observed_winner_en)
                    advancement_evaluable += 1
                    advancement_hits += int(predicted == observed_display)
            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "round_label": round_label,
                    "n_displayed_matches": int(len(displayed)),
                    "n_realized_matches": int(len(realised)),
                    "pair_hits": pair_hits,
                    "pair_hit_rate": pair_hits / len(displayed) if len(displayed) else np.nan,
                    "advancement_evaluable_matches": advancement_evaluable,
                    "advancement_hits": advancement_hits,
                    "advancement_hit_rate": advancement_hits / advancement_evaluable
                    if advancement_evaluable
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def snapshot_information_gain(project_root: str | Path = ".") -> pd.DataFrame:
    """Calculate consecutive Bernoulli KL changes in stage probabilities."""
    root = find_project_root(project_root)
    plan = snapshot_plan(root)
    available = [snapshot_id for snapshot_id in plan["snapshot_id"] if snapshot_id in _available_snapshot_ids(root)]
    rows: list[dict[str, Any]] = []
    previous: pd.DataFrame | None = None
    previous_id: str | None = None
    for snapshot_id in available:
        current = _read_snapshot_table(root, snapshot_id, "stage_probabilities")
        if previous is not None:
            merged = current.merge(
                previous[["team_en", "stage", "probability"]],
                on=["team_en", "stage"],
                suffixes=("_new", "_old"),
            )
            p_new = np.clip(merged["probability_new"].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
            p_old = np.clip(merged["probability_old"].to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
            kl = p_new * np.log(p_new / p_old) + (1.0 - p_new) * np.log((1.0 - p_new) / (1.0 - p_old))
            merged["bernoulli_kl_new_vs_old"] = kl
            for stage, group in merged.groupby("stage", sort=False):
                rows.append(
                    {
                        "previous_snapshot_id": previous_id,
                        "snapshot_id": snapshot_id,
                        "stage": stage,
                        "mean_bernoulli_kl": float(group["bernoulli_kl_new_vs_old"].mean()),
                        "sum_bernoulli_kl": float(group["bernoulli_kl_new_vs_old"].sum()),
                        "note": "A probability-distribution change, not a score or causal measure of update value.",
                    }
                )
        previous = current
        previous_id = snapshot_id
    return pd.DataFrame(rows)


def paired_bootstrap(
    detailed_metrics: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = 20261014,
) -> pd.DataFrame:
    """Paired match bootstrap comparing as-of and pre-tournament model scores."""
    model = detailed_metrics[detailed_metrics["system"] == "model"]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for metric in ("brier", "rps", "log_score"):
        pivot = model.pivot(index="match_id", columns="prediction_set", values=metric)
        if not {"as_of", "pre_tournament"}.issubset(pivot.columns):
            continue
        difference = (pivot["as_of"] - pivot["pre_tournament"]).dropna().to_numpy(dtype=float)
        if len(difference) == 0:
            continue
        samples = np.mean(
            difference[rng.integers(0, len(difference), size=(n_bootstrap, len(difference)))],
            axis=1,
        )
        rows.append(
            {
                "metric": metric,
                "n_matches": int(len(difference)),
                "n_bootstrap": n_bootstrap,
                "mean_difference_asof_minus_pre": float(np.mean(difference)),
                "bootstrap_q2_5": float(np.quantile(samples, 0.025)),
                "bootstrap_q97_5": float(np.quantile(samples, 0.975)),
                "interpretation": "Negative values favour the as-of forecast; the match count is small and the interval is descriptive.",
            }
        )
    return pd.DataFrame(rows)


def backtest_preflight(project_root: str | Path = ".") -> pd.DataFrame:
    """State exactly whether ``src.model_sel.validate`` can run in this checkout."""
    root = find_project_root(project_root)
    raw_results = root / "data" / "raw" / "results.csv"
    rows = [
        {
            "component": "src.model_sel.validate",
            "required_file": "data/raw/results.csv",
            "available": raw_results.is_file(),
            "status": "ready" if raw_results.is_file() else "blocked",
            "detail": (
                "The historical full-results input is present. FULL mode may run the project's validation script in an isolated runtime."
                if raw_results.is_file()
                else "The project script hard-codes data/raw/results.csv, which is absent from this checkout. No backtest is run."
            ),
        },
        {
            "component": "jogos_2014/2018/2022",
            "required_file": "data/raw/jogos_2014.csv; data/raw/jogos_2018.csv; data/raw/jogos_2022.csv",
            "available": all((root / "data" / "raw" / f"jogos_{year}.csv").is_file() for year in (2014, 2018, 2022)),
            "status": "available",
            "detail": "These three test-set files are present, but validate.py also needs the missing full historical training file.",
        },
    ]
    return pd.DataFrame(rows)


def run_project_backtests(project_root: str | Path = ".") -> pd.DataFrame:
    """Optionally invoke the existing validation script in an isolated runtime.

    This is intentionally not called by default. ``validate.py`` fits every
    configured model/cycle pair, so it is a long MCMC workflow rather than a
    notebook-side calculation. It remains disabled when its required historical
    input is absent.
    """
    root = ensure_project_imports(project_root)
    preflight = backtest_preflight(root)
    output = _mkdir(analysis_root(root) / "artifacts" / "evaluation")
    if not bool(preflight.iloc[0]["available"]):
        return preflight
    _require_full_dependencies()
    runtime = analysis_root(root) / ".runtime" / "backtests"
    _mkdir(runtime / "data" / "raw")
    for source in (root / "data" / "raw").glob("*.csv"):
        _copy_if_changed(source, runtime / "data" / "raw" / source.name)
    for source in (root / "stan_models").glob("*.stan"):
        _copy_if_changed(source, runtime / "stan_models" / source.name)
    started = time.perf_counter()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "src.model_sel.validate"],
        cwd=runtime,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output / "validate_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    result = pd.DataFrame(
        [
            {
                "component": "src.model_sel.validate",
                "returncode": completed.returncode,
                "runtime_seconds": time.perf_counter() - started,
                "stdout_path": "artifacts/evaluation/validate_stdout.txt",
            }
        ]
    )
    result.to_parquet(output / "backtest_run_manifest.parquet", index=False)
    return result

def evaluate_from_saved(
    project_root: str | Path = ".",
    *,
    n_bootstrap: int = 2000,
    seed: int = 20261014,
) -> dict[str, pd.DataFrame]:
    """Run all paper evaluation tables from saved snapshot products only."""
    root = find_project_root(project_root)
    if not _available_snapshot_ids(root):
        raise FileNotFoundError(
            "No FULL snapshot outputs are available. Evaluation intentionally does not refit models."
        )
    output = _mkdir(analysis_root(root) / "artifacts" / "evaluation")
    predictions = build_match_predictions_from_saved(root)
    predictions, baseline_parameters = attach_baselines(predictions, root)
    metrics, detailed_metrics = match_level_metrics(predictions)
    calibration, calibration_error = calibration_tables(predictions)
    draws = draw_analysis(predictions, detailed_metrics)
    goal_detailed, score_frequency, dispersion = goal_level_tables(predictions, root, seed=seed)
    score_details, score_metrics, score_by_phase = score_prediction_tables(predictions, root)
    stage = tournament_level_metrics(root)
    title = title_outcome_evaluation(root)
    bracket = bracket_hit_rates(root)
    information = snapshot_information_gain(root)
    bootstrap = paired_bootstrap(detailed_metrics, n_bootstrap=n_bootstrap, seed=seed)
    backtests = backtest_preflight(root)

    products = {
        "match_predictions": predictions,
        "baseline_parameters": baseline_parameters,
        "match_level_metrics": metrics,
        "match_score_contributions": detailed_metrics,
        "calibration_bins": calibration,
        "calibration_error": calibration_error,
        "draw_analysis": draws,
        "goal_level_match_details": goal_detailed,
        "scoreline_frequency_check": score_frequency,
        "goal_dispersion": dispersion,
        "score_prediction_details": score_details,
        "score_prediction_metrics": score_metrics,
        "score_prediction_by_phase": score_by_phase,
        "tournament_level_metrics": stage,
        "title_outcome_evaluation": title,
        "bracket_hit_rates": bracket,
        "snapshot_information_gain": information,
        "paired_bootstrap": bootstrap,
    }
    for name, frame in products.items():
        frame.to_parquet(output / f"{name}.parquet", index=False)
    write_json(
        output / "evaluation_metadata.json",
        {
            "schema_version": SCHEMA_VERSION,
            "mode": "SAVED_ARTIFACT_EVALUATION",
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "available_snapshots": _available_snapshot_ids(root),
            "important_limitations": [
                "As-of snapshots are selected using a strictly earlier calendar date because the supplied results lack kick-off timestamps.",
                "Champion and third-place-specific evaluation is reported only when the current source files verify those outcomes.",
                "Scoreline log scores are censored to the exported 0--4 grid; observations outside the grid use its saved residual mass.",
                "Point-score diagnostics distinguish model expected-goal errors from modal published-grid scoreline errors.",
                "Bootstrap intervals are paired over a small set of matches and are descriptive rather than confirmatory.",
            ],
        },
    )
    return {**products, "backtest_preflight": backtests}
