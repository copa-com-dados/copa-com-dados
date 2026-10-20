"""Exploratory analysis and manuscript-ready figures and tables.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.ticker import PercentFormatter
from scipy.stats import poisson

from .audit_evaluation import (
    build_match_predictions_from_saved,
    score_prediction_tables,
)
from .audit_snapshots import (
    DEFAULT_START_DATE,
    EDA_END_DATE,
    GRID_MAX,
    OKABE_ITO,
    SCORELINE_HEATMAP_EXAMPLES,
    SNAPSHOT_DISPLAY_LABELS,
    STAGE_HEATMAP_N_TEAMS,
    STAGE_ORDER,
    TITLE_TRAJECTORY_COLOURS,
    TITLE_TRAJECTORY_TEAMS_EN,
    _available_snapshot_ids,
    _mkdir,
    _read_results,
    _read_snapshot_table,
    _source_file_inventory,
    analysis_root,
    data_inventory,
    ensure_project_imports,
    snapshot_plan,
)

__all__ = [
    "configure_publication_style",
    "data_inventory",
    "run_exploratory_data_analysis",
]

def configure_publication_style() -> None:
    """Apply a compact, colourblind-safe Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "font.family": "DejaVu Sans",
        }
    )


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    """Save each paper figure in both requested formats."""
    _mkdir(output_dir)
    outputs = [output_dir / f"{stem}.png"]
    for path in outputs:
        figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return outputs


def _write_latex_table(frame: pd.DataFrame, destination: Path, float_format: str = "%.3f") -> None:
    """Export a booktabs-compatible tabular body without hard-coded statistics."""
    _mkdir(destination.parent)
    # pandas>=2.2 delegates to Styler and consequently requires Jinja2. The
    # paper pipeline needs only a simple booktabs tabular, so writing it here
    # keeps the actual-data EDA runnable in the project's lean environment.
    def escape(value: object) -> str:
        text = str(value)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def format_value(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return float_format % float(value)
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return escape(value)

    lines = [
        r"\begin{tabular}{" + "l" * len(frame.columns) + "}",
        r"\toprule",
        " & ".join(escape(column) for column in frame.columns) + r" \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(format_value(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    destination.write_text("\n".join(lines), encoding="utf-8")


def _file_quality_tables(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    overview: list[dict[str, Any]] = []
    missingness: list[dict[str, Any]] = []
    for label, path, _ in _source_file_inventory(project_root):
        if not path.is_file():
            overview.append(
                {
                    "file": label,
                    "exists": False,
                    "rows": np.nan,
                    "columns": np.nan,
                    "duplicate_rows": np.nan,
                    "note": "Missing in this checkout.",
                }
            )
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - saved for audit
            overview.append(
                {
                    "file": label,
                    "exists": True,
                    "rows": np.nan,
                    "columns": np.nan,
                    "duplicate_rows": np.nan,
                    "note": f"Read error: {type(exc).__name__}: {exc}",
                }
            )
            continue
        overview.append(
            {
                "file": label,
                "exists": True,
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "duplicate_rows": int(frame.duplicated().sum()),
                "note": "",
            }
        )
        for column, count in frame.isna().sum().items():
            missingness.append(
                {
                    "file": label,
                    "column": column,
                    "missing_count": int(count),
                    "missing_share": float(count / len(frame)) if len(frame) else np.nan,
                }
            )
    return pd.DataFrame(overview), pd.DataFrame(missingness)


def _training_data_for_eda(project_root: Path) -> tuple[pd.DataFrame, list[str]]:
    ensure_project_imports(project_root)
    from src.data import prepare_cycle_data

    data, teams, _ = prepare_cycle_data(
        str(project_root / "data" / "results.csv"),
        DEFAULT_START_DATE,
        end_date=EDA_END_DATE,
        apply_decay=True,
    )
    return data.copy(), teams


def _generated_results_ready(project_root: Path) -> bool:
    return bool(_available_snapshot_ids(project_root))


def _snapshot_order_map(project_root: Path) -> dict[str, int]:
    plan = snapshot_plan(project_root)
    return dict(zip(plan["snapshot_id"], plan["snapshot_order"], strict=True))


def _plot_time_decay(output: Path) -> list[Path]:
    """Plot the analytical temporal component of the implemented match weight."""
    alpha = 0.001
    half_life = math.log(2.0) / alpha
    ages = np.linspace(0.0, 2_500.0, 600)
    weights = np.exp(-alpha * ages)
    figure, axis = plt.subplots(figsize=(7.6, 4.4))
    axis.plot(ages, weights, color=OKABE_ITO["blue"], linewidth=2.2, label="Exponential time component")
    axis.fill_between(ages, weights, color=OKABE_ITO["sky_blue"], alpha=0.22)
    axis.axvline(half_life, color=OKABE_ITO["vermillion"], linestyle="--", linewidth=1.25)
    axis.axhline(0.5, color=OKABE_ITO["grey"], linestyle=":", linewidth=1.0)
    axis.annotate(
        f"Half-life = {half_life:.1f} days",
        xy=(half_life, 0.5),
        xytext=(half_life + 220, 0.66),
        arrowprops={"arrowstyle": "->", "color": OKABE_ITO["vermillion"]},
        color=OKABE_ITO["vermillion"],
        fontsize=9,
    )
    axis.set_xlim(0, ages.max())
    axis.set_ylim(0, 1.04)
    axis.set_xlabel("Match age at the training cutoff (days)")
    axis.set_ylabel("Exponential time weight")
    axis.set_title("Temporal downweighting in the implemented likelihood")
    axis.legend(frameon=False, loc="upper right")
    figure.subplots_adjust(bottom=0.22)
    figure.text(
        0.5,
        0.055,
        r"For each match, this component is multiplied by the tournament factor $q(c)/10$.",
        ha="center",
        fontsize=8,
        color="#425466",
    )
    return _save_figure(figure, output, "time_decay")

def _plot_manuscript_method_figures(project_root: Path, output: Path) -> list[str]:
    """Generate every non-empirical manuscript figure from repository code."""
    _plot_time_decay(output)
    return ["time_decay"]

def _plot_matches_per_year(
    project_root: Path,
    training: pd.DataFrame,
    output: Path,
) -> list[Path]:
    """Compare complete cycle records with the exact final training filter."""
    raw = _read_results(project_root).copy()
    cutoff = pd.Timestamp(EDA_END_DATE)
    complete_cycle = raw.loc[
        raw["date"].between(pd.Timestamp(DEFAULT_START_DATE), cutoff, inclusive="both")
    ].dropna(subset=["home_score", "away_score"])
    raw_counts = complete_cycle.assign(year=complete_cycle["date"].dt.year).groupby("year").size()
    filtered_counts = training.assign(year=training["date"].dt.year).groupby("year").size()
    years = np.arange(int(min(raw_counts.index.min(), filtered_counts.index.min())), int(max(raw_counts.index.max(), filtered_counts.index.max())) + 1)
    raw_values = raw_counts.reindex(years, fill_value=0).to_numpy()
    filtered_values = filtered_counts.reindex(years, fill_value=0).to_numpy()

    figure, axis = plt.subplots(figsize=(8.3, 4.6))
    width = 0.36
    axis.bar(years - width / 2, raw_values, width, label="Complete records before team filters", color=OKABE_ITO["sky_blue"])
    axis.bar(years + width / 2, filtered_values, width, label="Final training records", color=OKABE_ITO["blue"])
    #cutoff_x = cutoff.year + (cutoff.dayofyear - 1) / 365.25
    #axis.axvline(cutoff_x, color=OKABE_ITO["vermillion"], linestyle="--", linewidth=1.25, label="Forecast cutoff")
    axis.set_xticks(years, years.astype(str))
    axis.set_xlabel("Calendar year")
    axis.set_ylabel("Complete international matches")
    axis.set_title("Match volume before and after the 2026 cycle filters")
    axis.legend(frameon=False, loc="upper left")
    figure.subplots_adjust(bottom=0.22)
    return _save_figure(figure, output, "matches_per_year")


def _plot_tournament_mix(training: pd.DataFrame, output: Path) -> list[Path]:
    counts = training["tournament"].value_counts().head(12).sort_values()
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.barh(counts.index, counts.values, color=OKABE_ITO["bluish_green"])
    axis.set_xlabel("Training matches")
    axis.set_ylabel("Tournament")
    axis.set_title("Most frequent tournament categories in the training window")
    return _save_figure(figure, output, "tournament_mix")


def _plot_team_coverage(training: pd.DataFrame, output: Path) -> tuple[list[Path], pd.DataFrame]:
    counts = (
        pd.concat([training["home_team"], training["away_team"]], ignore_index=True)
        .value_counts()
        .rename_axis("team")
        .reset_index(name="training_matches")
    )
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.hist(counts["training_matches"], bins=30, color=OKABE_ITO["sky_blue"], edgecolor="white")
    axis.set_xlabel("Matches per team")
    axis.set_ylabel("Teams")
    axis.set_title("Training-window team coverage")
    return _save_figure(figure, output, "matches_per_team"), counts


def _plot_goal_distribution(training: pd.DataFrame, output: Path) -> tuple[list[Path], pd.DataFrame]:
    goals = pd.concat([training["home_score"], training["away_score"]], ignore_index=True).astype(int)
    support = np.arange(0, max(8, int(goals.max())) + 1)
    empirical = goals.value_counts(normalize=True).reindex(support, fill_value=0.0).to_numpy()
    fitted = poisson.pmf(support, goals.mean())
    table = pd.DataFrame(
        {
            "goals": support,
            "observed_frequency": empirical,
            "poisson_frequency": fitted,
        }
    )
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.bar(support - 0.18, empirical, width=0.36, label="Observed", color=OKABE_ITO["sky_blue"])
    axis.bar(support + 0.18, fitted, width=0.36, label="Poisson fit", color=OKABE_ITO["vermillion"])
    axis.set_xlabel("Goals by one team in one match")
    axis.set_ylabel("Probability")
    axis.set_title("Goal distribution in the training window")
    axis.legend(frameon=False)
    return _save_figure(figure, output, "goals_poisson_fit"), table


def _plot_draw_rates(training: pd.DataFrame, output: Path) -> list[Path]:
    annual = (
        training.assign(
            year=training["date"].dt.year,
            draw=training["home_score"].eq(training["away_score"]),
        )
        .groupby("year", as_index=False)["draw"]
        .mean()
    )
    figure, axis = plt.subplots(figsize=(7.2, 3.5))
    axis.plot(annual["year"], annual["draw"], marker="o", color=OKABE_ITO["purple"])
    axis.set_xlabel("Match year")
    axis.set_ylabel("Observed draw frequency")
    axis.set_ylim(0, max(0.4, annual["draw"].max() * 1.15))
    axis.set_title("Observed draw frequency by year")
    return _save_figure(figure, output, "draw_rates_by_year")


def _plot_draw_rates_by_tournament(training: pd.DataFrame, output: Path) -> tuple[list[Path], pd.DataFrame]:
    table = (
        training.assign(draw=training["home_score"].eq(training["away_score"]))
        .groupby("tournament", as_index=False)
        .agg(matches=("draw", "size"), draw_frequency=("draw", "mean"))
        .query("matches >= 20")
        .sort_values("draw_frequency", ascending=False)
        .head(12)
        .sort_values("draw_frequency")
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.barh(table["tournament"], table["draw_frequency"], color=OKABE_ITO["purple"])
    axis.set_xlabel("Observed draw frequency")
    axis.set_ylabel("Tournament")
    axis.set_title("Draw frequency by tournament category (at least 20 matches)")
    return _save_figure(figure, output, "draw_rates_by_tournament"), table


def _plot_home_advantage(training: pd.DataFrame, output: Path) -> tuple[list[Path], pd.DataFrame]:
    summaries = (
        training.assign(venue=np.where(training["neutral"], "Neutral", "Non-neutral"))
        .groupby("venue", as_index=False)
        .agg(
            matches=("home_score", "size"),
            home_goals=("home_score", "mean"),
            away_goals=("away_score", "mean"),
            home_win_rate=("home_score", lambda x: np.nan),
        )
    )
    rates = []
    for venue, group in training.assign(venue=np.where(training["neutral"], "Neutral", "Non-neutral")).groupby("venue"):
        rates.append(
            {
                "venue": venue,
                "home_win_rate": float(np.mean(group["home_score"] > group["away_score"])),
                "draw_rate": float(np.mean(group["home_score"] == group["away_score"])),
                "away_win_rate": float(np.mean(group["home_score"] < group["away_score"])),
            }
        )
    summaries = summaries.drop(columns="home_win_rate").merge(pd.DataFrame(rates), on="venue")
    figure, axis = plt.subplots(figsize=(6.6, 3.8))
    positions = np.arange(len(summaries))
    width = 0.26
    axis.bar(positions - width / 2, summaries["home_goals"], width, label="Home goals", color=OKABE_ITO["blue"])
    axis.bar(positions + width / 2, summaries["away_goals"], width, label="Away goals", color=OKABE_ITO["orange"])
    axis.set_xticks(positions, summaries["venue"])
    axis.set_ylabel("Mean goals")
    axis.set_title("Observed home and away goals by venue status")
    axis.legend(frameon=False)
    return _save_figure(figure, output, "home_advantage"), summaries


def _plot_weights(training: pd.DataFrame, output: Path) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))
    ordered = training.sort_values("date")
    axes[0].scatter(ordered["date"], ordered["time_weight"], s=7, alpha=0.35, color=OKABE_ITO["blue"])
    axes[0].set_xlabel("Match date")
    axes[0].set_ylabel("Exponential time weight")
    axes[0].set_title("Time-decay component")
    axes[1].hist(training["game_weight"], bins=35, color=OKABE_ITO["orange"], edgecolor="white")
    axes[1].set_xlabel("Combined game weight")
    axes[1].set_ylabel("Matches")
    axes[1].set_title("Combined time and tournament weights")
    return _save_figure(figure, output, "training_weights")


def _model_selection_results(project_root: Path) -> pd.DataFrame:
    """Read the project's existing 2018/2022 candidate-model Brier artifacts."""
    output = project_root / "data" / "outputs" / "results"
    rows: list[pd.DataFrame] = []
    for cycle in ("2018", "2022"):
        path = output / f"brier_score_{cycle}.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        required = {"Modelo", "Brier Mediana", "IC Inferior", "IC Superior"}
        if not required.issubset(frame.columns):
            continue
        selected = frame.loc[:, ["Modelo", "Brier Mediana", "IC Inferior", "IC Superior"]].copy()
        selected.columns = ["model", "brier_median", "brier_lower", "brier_upper"]
        selected.insert(0, "cycle", cycle)
        rows.append(selected)
    if not rows:
        return pd.DataFrame(
            columns=["cycle", "model", "brier_median", "brier_lower", "brier_upper"]
        )
    return pd.concat(rows, ignore_index=True)


def _plot_model_selection(
    project_root: Path,
    figures: Path,
    tables: Path,
) -> tuple[list[str], list[str]]:
    """Plot archived candidate-model Brier intervals without rerunning validation."""
    results = _model_selection_results(project_root)
    if results.empty:
        return [], []
    cycles = list(results["cycle"].drop_duplicates())
    figure, axes = plt.subplots(1, len(cycles), figsize=(10.8, 4.8), sharex=False)
    if len(cycles) == 1:
        axes = [axes]
    for axis, cycle in zip(axes, cycles, strict=True):
        subset = results[results["cycle"] == cycle].sort_values("brier_median", ascending=True)
        y = np.arange(len(subset))
        selected_mask = subset["model"].str.lower().eq("n poisson ranking")
        colours = np.where(selected_mask, OKABE_ITO["vermillion"], OKABE_ITO["blue"])
        errors = np.vstack(
            [
                subset["brier_median"] - subset["brier_lower"],
                subset["brier_upper"] - subset["brier_median"],
            ]
        )
        for index, row in enumerate(subset.itertuples(index=False)):
            axis.errorbar(
                row.brier_median,
                index,
                xerr=np.array([[errors[0, index]], [errors[1, index]]]),
                fmt="o",
                color=colours[index],
                ecolor=colours[index],
                capsize=2,
                markersize=5,
            )
        axis.set_yticks(y, subset["model"])
        axis.invert_yaxis()
        axis.set_xlabel("Median Brier score (95% interval)")
        axis.set_title(f"{cycle} World Cup holdout")
        axis.grid(axis="x")
    figure.suptitle("Candidate-model comparison from archived validation outputs", y=1.02)
    _save_figure(figure, figures, "model_selection_brier")
    _write_latex_table(
        results.sort_values(["cycle", "brier_median"]),
        tables / "model_selection_brier.tex",
    )
    return ["model_selection_brier"], ["model_selection_brier"]


def _plot_score_prediction_diagnostics(
    details: pd.DataFrame,
    figures: Path,
) -> list[str]:
    """Visualise all as-of modal-score predictions against realised scores."""
    as_of = details[details["prediction_set"] == "as_of"].dropna(
        subset=["modal_home_goals", "modal_away_goals"]
    )
    if as_of.empty:
        return []
    predicted_total = as_of["modal_home_goals"] + as_of["modal_away_goals"]
    observed_total = as_of["home_score"] + as_of["away_score"]
    predicted_difference = as_of["modal_home_goals"] - as_of["modal_away_goals"]
    observed_difference = as_of["home_score"] - as_of["away_score"]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    panels = (
        (axes[0], predicted_total, observed_total, "Total goals"),
        (axes[1], predicted_difference, observed_difference, "Goal difference (home minus away)"),
    )
    for axis, predicted, observed, label in panels:
        lower = min(float(predicted.min()), float(observed.min())) - 0.2
        upper = max(float(predicted.max()), float(observed.max())) + 0.2
        axis.scatter(predicted, observed, color=OKABE_ITO["blue"], alpha=0.55, s=28)
        axis.plot([lower, upper], [lower, upper], linestyle="--", color=OKABE_ITO["grey"], linewidth=1)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_xlabel(f"Predicted modal {label.lower()}")
        axis.set_ylabel(f"Observed {label.lower()}")
        axis.set_title(label)
    figure.suptitle("As-of modal-score forecasts versus realised World Cup scores", y=1.02)
    _save_figure(figure, figures, "score_prediction_diagnostics")
    return ["score_prediction_diagnostics"]


def _plot_parameter_trajectory_if_available(
    project_root: Path,
    figures: Path,
) -> list[str]:
    """Render selected-team parameter paths from refits or saved snapshots.

    The first implementation only looked for the optional year-end refit
    artifact.  A normal FULL notebook run does not create that artifact, even
    though it already saves posterior team summaries for every World Cup
    snapshot.  Falling back to those saved summaries makes the figure part of
    the standard EDA output without retraining any model.
    """
    path = (
        analysis_root(project_root)
        / "artifacts"
        / "parameter_trajectory"
        / "team_parameter_trajectory.parquet"
    )
    title_suffix = "from repeated cutoff fits"
    if path.is_file():
        trajectory = pd.read_parquet(path)
    else:
        available = _available_snapshot_ids(project_root)
        plan = snapshot_plan(project_root)
        plan = plan[plan["snapshot_id"].isin(available)]
        frames: list[pd.DataFrame] = []
        for row in plan.itertuples(index=False):
            parameters = _read_snapshot_table(
                project_root, str(row.snapshot_id), "team_parameter_summaries"
            ).copy()
            parameters["trajectory_id"] = f"snapshot_{row.snapshot_id}"
            parameters["trajectory_order"] = int(row.snapshot_order)
            parameters["trajectory_label"] = SNAPSHOT_DISPLAY_LABELS.get(
                str(row.snapshot_id), str(row.snapshot_id)
            )
            frames.append(parameters)
        if not frames:
            return []
        trajectory = pd.concat(frames, ignore_index=True)
        title_suffix = "across World Cup snapshots"
    selected = trajectory[trajectory["team_en"].isin(TITLE_TRAJECTORY_TEAMS_EN)].copy()
    if selected.empty:
        return []
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharex=True)
    for axis, parameter, title in (
        (axes[0], "attack", "Attack component"),
        (axes[1], "defense", "Defence component (higher = stronger)"),
    ):
        subset = selected[selected["parameter"] == parameter]
        for team in TITLE_TRAJECTORY_TEAMS_EN:
            series = subset[subset["team_en"] == team].sort_values("trajectory_order")
            if series.empty:
                continue
            axis.plot(
                series["trajectory_order"],
                series["mean"],
                marker="o",
                linewidth=1.5,
                markersize=3.5,
                color=TITLE_TRAJECTORY_COLOURS[team],
                label=team,
            )
        labels = (
            subset[["trajectory_order", "trajectory_label"]]
            .drop_duplicates()
            .sort_values("trajectory_order")
        )
        axis.set_xticks(labels["trajectory_order"], labels["trajectory_label"], rotation=30, ha="right")
        axis.axhline(0.0, color=OKABE_ITO["grey"], linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Training cutoff")
        axis.set_ylabel("Posterior mean log-rate component")
    axes[1].legend(ncol=2, frameon=False, fontsize=7, loc="best")
    figure.suptitle(f"Posterior team-strength trajectories {title_suffix}", y=1.03)
    _save_figure(figure, figures, "parameter_trajectories")
    return ["parameter_trajectories"]


def _latest_snapshot_id(project_root: Path) -> str | None:
    available = _available_snapshot_ids(project_root)
    if not available:
        return None
    plan = snapshot_plan(project_root)
    return str(plan[plan["snapshot_id"].isin(available)].iloc[-1]["snapshot_id"])


def _posterior_strength_components(team_parameters: pd.DataFrame) -> pd.DataFrame:
    """Pivot saved attack/defence summaries into one team-level table."""
    attack = team_parameters[team_parameters["parameter"] == "attack"].rename(
        columns={column: f"attack_{column}" for column in ["mean", "q2_5", "q97_5"]}
    )
    defense = team_parameters[team_parameters["parameter"] == "defense"].rename(
        columns={column: f"defense_{column}" for column in ["mean", "q2_5", "q97_5"]}
    )
    return attack[
        ["team_en", "team_display", "attack_mean", "attack_q2_5", "attack_q97_5"]
    ].merge(
        defense[["team_en", "defense_mean", "defense_q2_5", "defense_q97_5"]],
        on="team_en",
        how="inner",
    )


def _plot_world_cup_attack_defense(
    strengths: pd.DataFrame,
    figures: Path,
    stem: str,
    cutoff_label: str,
) -> str | None:
    """Plot one Portuguese-labelled strength map for the qualified teams."""
    selected = strengths[strengths["world_cup_team"]].copy()
    if selected.empty:
        return None
    figure, axis = plt.subplots(figsize=(9.2, 7.4))
    axis.errorbar(
        selected["attack_mean"],
        selected["defense_mean"],
        xerr=np.vstack(
            [
                selected["attack_mean"] - selected["attack_q2_5"],
                selected["attack_q97_5"] - selected["attack_mean"],
            ]
        ),
        yerr=np.vstack(
            [
                selected["defense_mean"] - selected["defense_q2_5"],
                selected["defense_q97_5"] - selected["defense_mean"],
            ]
        ),
        fmt="none",
        ecolor=OKABE_ITO["grey"],
        alpha=0.26,
        linewidth=0.5,
        zorder=1,
    )
    axis.scatter(
        selected["attack_mean"],
        selected["defense_mean"],
        color=OKABE_ITO["blue"],
        s=24,
        alpha=0.85,
        zorder=2,
    )
    for index, row in selected.sort_values("team_display").reset_index(drop=True).iterrows():
        horizontal = 3 if index % 2 == 0 else -3
        vertical = 3 if (index // 2) % 2 == 0 else -5
        axis.annotate(
            str(row.team_display),
            (row.attack_mean, row.defense_mean),
            xytext=(horizontal, vertical),
            textcoords="offset points",
            fontsize=5.8,
            alpha=0.9,
        )
    axis.axhline(0.0, color=OKABE_ITO["grey"], linewidth=0.8)
    axis.axvline(0.0, color=OKABE_ITO["grey"], linewidth=0.8)
    axis.set_xlabel("Posterior attack component (higher = stronger attack)")
    axis.set_ylabel("Posterior defence component (higher = stronger defence)")
    axis.set_title(f"Attack and defence posterior means for World Cup teams ({cutoff_label})")
    _save_figure(figure, figures, stem)
    return stem


def _plot_all_teams_attack_defense(strengths: pd.DataFrame, figures: Path) -> str | None:
    """Plot all fitted teams while labelling only World Cup-qualified teams."""
    if strengths.empty:
        return None
    figure, axis = plt.subplots(figsize=(13.0, 7.8))
    for selected, colour, label in (
        (False, "#9BB3F0", "Other national teams"),
        (True, "#DC143C", "World Cup-qualified teams"),
    ):
        subset = strengths[strengths["world_cup_team"] == selected]
        axis.scatter(
            subset["attack_mean"],
            subset["defense_mean"],
            s=32 if selected else 26,
            alpha=0.88 if selected else 0.76,
            color=colour,
            label=label,
            zorder=3 if selected else 2,
        )
    qualified = strengths[strengths["world_cup_team"]].sort_values("team_en").reset_index(drop=True)
    for index, row in qualified.iterrows():
        horizontal = 4 if index % 2 == 0 else -4
        vertical = 4 if (index // 2) % 2 == 0 else -6
        axis.annotate(
            str(row.team_en),
            (row.attack_mean, row.defense_mean),
            xytext=(horizontal, vertical),
            textcoords="offset points",
            fontsize=5.5,
            color="#31496D",
            alpha=0.95,
            zorder=4,
        )
    axis.axhline(0.0, color=OKABE_ITO["grey"], linewidth=0.8, linestyle="--")
    axis.axvline(0.0, color=OKABE_ITO["grey"], linewidth=0.8, linestyle="--")
    axis.set_xlabel("Attack strength (posterior component)")
    axis.set_ylabel("Defence strength (posterior component; higher = stronger)")
    axis.set_title("Attack and defence across all fitted national teams")
    axis.legend(frameon=False, loc="upper left")
    _save_figure(figure, figures, "all_teams_attack_defense")
    return "all_teams_attack_defense"


def _plot_generated_outputs(project_root: Path, figures: Path, tables: Path) -> tuple[list[str], list[str]]:
    """Generate result-dependent EDA figures only when FULL products exist."""
    snapshot_ids = _available_snapshot_ids(project_root)
    if not snapshot_ids:
        return [], []
    order_map = _snapshot_order_map(project_root)
    stages = pd.concat(
        [_read_snapshot_table(project_root, snapshot_id, "stage_probabilities") for snapshot_id in snapshot_ids],
        ignore_index=True,
    )
    stages["snapshot_order"] = stages["snapshot_id"].map(order_map)
    latest_id = _latest_snapshot_id(project_root)
    assert latest_id is not None
    latest = stages[stages["snapshot_id"] == latest_id]
    generated_figure_names: list[str] = []
    generated_table_names: list[str] = []

    # Posterior team parameters and their relationship to the static FIFA prior.
    team_parameters = _read_snapshot_table(project_root, latest_id, "team_parameter_summaries")
    strengths = _posterior_strength_components(team_parameters)
    ensure_project_imports(project_root)
    from src.constants import GROUPS
    from src.data import load_ranking_priors

    strength_prior = load_ranking_priors(
        project_root / "data" / "raw" / "fifa_ranking_2022.csv",
        strengths["team_en"].tolist(),
    )
    strengths["fifa_prior_strength"] = strength_prior
    wc_set = {team for group in GROUPS.values() for team in group}
    strengths["world_cup_team"] = strengths["team_en"].isin(wc_set)
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    axes[0].errorbar(
        strengths["defense_mean"],
        strengths["attack_mean"],
        xerr=np.vstack(
            [
                strengths["defense_mean"] - strengths["defense_q2_5"],
                strengths["defense_q97_5"] - strengths["defense_mean"],
            ]
        ),
        yerr=np.vstack(
            [
                strengths["attack_mean"] - strengths["attack_q2_5"],
                strengths["attack_q97_5"] - strengths["attack_mean"],
            ]
        ),
        fmt="none",
        ecolor=OKABE_ITO["grey"],
        alpha=0.15,
        linewidth=0.5,
    )
    for selected, colour, label in (
        (False, OKABE_ITO["grey"], "Other fitted teams"),
        (True, OKABE_ITO["blue"], "World Cup teams"),
    ):
        subset = strengths[strengths["world_cup_team"] == selected]
        axes[0].scatter(subset["defense_mean"], subset["attack_mean"], s=18, alpha=0.65, color=colour, label=label)
        axes[1].scatter(subset["fifa_prior_strength"], subset["attack_mean"], s=18, alpha=0.65, color=colour, label=label)
    axes[0].set_xlabel("Posterior defence log-rate component")
    axes[0].set_ylabel("Posterior attack log-rate component")
    axes[0].set_title("Posterior team components")
    axes[1].set_xlabel("Normalised static FIFA-ranking prior")
    axes[1].set_ylabel("Posterior attack log-rate component")
    axes[1].set_title("Ranking prior and attack component")
    axes[1].legend(frameon=False, fontsize=7)
    _save_figure(figure, figures, "team_strengths_and_ranking_prior")
    _write_latex_table(
        strengths[strengths["world_cup_team"]].sort_values("attack_mean", ascending=False),
        tables / "world_cup_team_parameter_summaries.tex",
    )
    generated_figure_names.append("team_strengths_and_ranking_prior")
    generated_table_names.append("world_cup_team_parameter_summaries")
    generated_figure_names.extend(_plot_parameter_trajectory_if_available(project_root, figures))

    # The two World-Cup-only maps deliberately use Portuguese display names.
    # The separate all-team view below labels only qualified teams in English.
    last_map = _plot_world_cup_attack_defense(
        strengths,
        figures,
        "last_world_cup_attack_defense",
        SNAPSHOT_DISPLAY_LABELS.get(latest_id, latest_id),
    )
    if last_map is not None:
        generated_figure_names.append(last_map)

    pre_world_cup_id = (
        "pre_tournament_post_fifa"
        if "pre_tournament_post_fifa" in snapshot_ids
        else snapshot_ids[0]
    )
    pre_world_cup_strengths = _posterior_strength_components(
        _read_snapshot_table(project_root, pre_world_cup_id, "team_parameter_summaries")
    )
    pre_world_cup_strengths["world_cup_team"] = pre_world_cup_strengths["team_en"].isin(wc_set)
    before_map = _plot_world_cup_attack_defense(
        pre_world_cup_strengths,
        figures,
        "before_world_cup_attack_defense",
        SNAPSHOT_DISPLAY_LABELS.get(pre_world_cup_id, pre_world_cup_id),
    )
    if before_map is not None:
        generated_figure_names.append(before_map)

    all_team_map = _plot_all_teams_attack_defense(strengths, figures)
    if all_team_map is not None:
        generated_figure_names.append(all_team_map)

    # Freeze the compact heatmap before the tournament, then show only the
    # leading teams.  This preserves the readable layout of the original
    # post-semi-final plot without conditioning its early stages on results.
    heatmap_id = pre_world_cup_id
    heatmap_source = stages[stages["snapshot_id"] == heatmap_id]
    full_title_order = (
        heatmap_source[heatmap_source["stage"] == "champion"]
        .sort_values("probability", ascending=False)["team_en"]
        .tolist()
    )
    full_heat = (
        heatmap_source[
            (heatmap_source["team_en"].isin(full_title_order))
            & (heatmap_source["stage"].isin(STAGE_ORDER))
        ]
        .pivot(index="team_en", columns="stage", values="probability")
        .reindex(full_title_order)
        .reindex(columns=STAGE_ORDER)
    )
    heat = full_heat.head(STAGE_HEATMAP_N_TEAMS)
    figure, axis = plt.subplots(figsize=(8.5, 4.9))
    image = axis.imshow(heat.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1.0)
    axis.set_xticks(range(len(heat.columns)), [name.replace("_", " ") for name in heat.columns], rotation=30, ha="right")
    axis.set_yticks(range(len(heat.index)), heat.index)
    axis.tick_params(axis="y", labelsize=9)
    axis.set_title(
        f"Stage probabilities for leading teams ({SNAPSHOT_DISPLAY_LABELS.get(heatmap_id, heatmap_id)})"
    )
    figure.colorbar(image, ax=axis, label="Probability")
    _save_figure(figure, figures, "stage_probability_heatmap")
    _write_latex_table(
        heat.reset_index().rename(columns={"team_en": "team"}),
        tables / "stage_probability_heatmap_data.tex",
    )
    # Keep the complete matrix as a readable appendix-ready table.  The plot
    # remains on the unit interval, whereas the table uses explicit percent
    # column names so it cannot be mistaken for probabilities in [0, 1].
    stage_table = (100.0 * full_heat).reset_index().rename(
        columns={
            "team_en": "team",
            **{stage: f"{stage}_pct" for stage in STAGE_ORDER},
        }
    )
    _write_latex_table(
        stage_table,
        tables / "pre_tournament_stage_probabilities.tex",
        float_format="%.1f",
    )
    generated_figure_names.append("stage_probability_heatmap")
    generated_table_names.append("stage_probability_heatmap_data")
    generated_table_names.append("pre_tournament_stage_probabilities")

    trajectories = stages[
        (stages["team_en"].isin(TITLE_TRAJECTORY_TEAMS_EN))
        & (stages["stage"] == "champion")
    ]
    figure, axis = plt.subplots(figsize=(9.2, 4.6))
    for team in TITLE_TRAJECTORY_TEAMS_EN:
        group = trajectories[trajectories["team_en"] == team].sort_values("snapshot_order")
        if group.empty:
            continue
        axis.plot(
            group["snapshot_order"],
            group["probability"],
            marker="o",
            markersize=3.8,
            linewidth=1.7,
            color=TITLE_TRAJECTORY_COLOURS[team],
            label=team,
        )
    ticks = (
        trajectories[["snapshot_id", "snapshot_order"]]
        .drop_duplicates()
        .sort_values("snapshot_order")
    )
    axis.set_xticks(
        ticks["snapshot_order"],
        [SNAPSHOT_DISPLAY_LABELS.get(snapshot_id, snapshot_id) for snapshot_id in ticks["snapshot_id"]],
    )
    axis.set_xlabel("Forecast cutoff")
    axis.set_ylabel("Title probability")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_ylim(bottom=-0.01)
    axis.set_title("Title-probability trajectories for selected World Cup teams")
    axis.legend(ncol=2, frameon=False, fontsize=7, loc="upper left")
    _save_figure(figure, figures, "title_probability_trajectories")
    generated_figure_names.append("title_probability_trajectories")

    concentration = (
        stages[stages["stage"] == "champion"]
        .groupby(["snapshot_id", "snapshot_order"], as_index=False)["probability"]
        .agg(lambda p: float(np.sum(np.square(p))))
        .rename(columns={"probability": "title_probability_hhi"})
        .sort_values("snapshot_order")
    )
    figure, axis = plt.subplots(figsize=(7.2, 3.5))
    axis.plot(concentration["snapshot_order"], concentration["title_probability_hhi"], marker="o", color=OKABE_ITO["vermillion"])
    axis.set_xlabel("Snapshot order")
    axis.set_ylabel("HHI")
    axis.set_title("Concentration of title probability")
    _save_figure(figure, figures, "title_probability_concentration")
    _write_latex_table(concentration, tables / "title_probability_concentration.tex")
    generated_figure_names.append("title_probability_concentration")
    generated_table_names.append("title_probability_concentration")

    matchup = _read_snapshot_table(project_root, latest_id, "matchup_summary")
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    difference = matchup["expected_home_goals"] - matchup["expected_away_goals"]
    scatter = axis.scatter(
        difference,
        matchup["draw_probability"],
        c=matchup["expected_home_goals"] + matchup["expected_away_goals"],
        cmap="viridis",
        alpha=0.6,
        s=16,
    )
    axis.set_xlabel("Expected-goal difference (home minus away)")
    axis.set_ylabel("Predicted draw probability")
    axis.set_title(f"Draw probability and expected-goal mismatch ({latest_id})")
    figure.colorbar(scatter, ax=axis, label="Expected total goals")
    _save_figure(figure, figures, "draw_probability_vs_goal_difference")
    generated_figure_names.append("draw_probability_vs_goal_difference")

    scoreline_panels: list[tuple[str, str, str, str, pd.DataFrame]] = []
    for snapshot_id, home, away, subtitle in SCORELINE_HEATMAP_EXAMPLES:
        if snapshot_id not in snapshot_ids:
            continue
        scorelines = _read_snapshot_table(project_root, snapshot_id, "scoreline_probabilities")
        subset = scorelines[
            (scorelines["home_team_display"] == home)
            & (scorelines["away_team_display"] == away)
        ].copy()
        if subset.empty:
            subset = scorelines[
                (scorelines["home_team_display"] == away)
                & (scorelines["away_team_display"] == home)
            ].copy()
            if not subset.empty:
                original_home = subset["home_goals"].copy()
                subset["home_goals"] = subset["away_goals"]
                subset["away_goals"] = original_home
        if not subset.empty:
            scoreline_panels.append((snapshot_id, home, away, subtitle, subset))
    if scoreline_panels:
        figure, axes = plt.subplots(1, len(scoreline_panels), figsize=(5.0 * len(scoreline_panels), 4.2))
        if len(scoreline_panels) == 1:
            axes = [axes]
        matrices = [
            panel[-1]
            .pivot(index="home_goals", columns="away_goals", values="probability")
            .reindex(index=range(GRID_MAX + 1), columns=range(GRID_MAX + 1), fill_value=0.0)
            for panel in scoreline_panels
        ]
        vmax = max(float(matrix.to_numpy().max()) for matrix in matrices)
        image = None
        for axis, panel, matrix in zip(axes, scoreline_panels, matrices, strict=True):
            snapshot_id, home, away, subtitle, _ = panel
            image = axis.imshow(matrix.to_numpy(), origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
            axis.set_xticks(range(GRID_MAX + 1), range(GRID_MAX + 1))
            axis.set_yticks(range(GRID_MAX + 1), range(GRID_MAX + 1))
            axis.set_xlabel(f"{away} goals")
            axis.set_ylabel(f"{home} goals")
            axis.set_title(f"{home} vs {away}\n{subtitle} ({snapshot_id})", fontsize=9)
        assert image is not None
        figure.colorbar(image, ax=axes, label="Probability of exact 0--4 scoreline", shrink=0.86)
        figure.suptitle("Pre-match scoreline probability examples", y=1.02)
        _save_figure(figure, figures, "scoreline_heatmap_examples")
        generated_figure_names.append("scoreline_heatmap_examples")

    consistency = matchup.assign(
        outcome_probability_error=np.abs(
            matchup["home_win_probability"] + matchup["draw_probability"] + matchup["away_win_probability"] - 1.0
        ),
        score_grid_mass_error=np.abs(
            matchup["grid_probability_mass"] + matchup["residual_probability_mass"] - 1.0
        ),
    )
    consistency_table = consistency[["outcome_probability_error", "score_grid_mass_error"]].describe().T.reset_index().rename(columns={"index": "check"})
    _write_latex_table(consistency_table, tables / "matchup_probability_consistency.tex")
    generated_table_names.append("matchup_probability_consistency")

    evaluation_dir = analysis_root(project_root) / "artifacts" / "evaluation"
    calibration_path = evaluation_dir / "calibration_bins.parquet"
    if calibration_path.is_file():
        calibration = pd.read_parquet(calibration_path)
        calibration = calibration[calibration["prediction_set"] == "as_of"]
        figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharex=True, sharey=True)
        for axis, outcome in zip(axes, ("home_win", "draw", "away_win"), strict=True):
            subset = calibration[calibration["outcome"] == outcome]
            axis.plot([0, 1], [0, 1], linestyle="--", color=OKABE_ITO["grey"], linewidth=1)
            axis.scatter(subset["mean_predicted_probability"], subset["observed_frequency"], s=20, color=OKABE_ITO["blue"])
            axis.set_title(outcome.replace("_", " "))
            axis.set_xlabel("Mean predicted probability")
        axes[0].set_ylabel("Observed frequency")
        figure.suptitle("As-of reliability diagrams", y=1.02)
        _save_figure(figure, figures, "calibration_reliability")
        generated_figure_names.append("calibration_reliability")

    score_details_path = evaluation_dir / "score_prediction_details.parquet"
    score_metrics_path = evaluation_dir / "score_prediction_metrics.parquet"
    score_details: pd.DataFrame | None = None
    score_metrics: pd.DataFrame | None = None
    required_score_columns = {
        "prediction_set",
        "modal_home_goals",
        "modal_away_goals",
        "home_score",
        "away_score",
    }
    if score_details_path.is_file() and score_metrics_path.is_file():
        loaded_details = pd.read_parquet(score_details_path)
        if required_score_columns.issubset(loaded_details.columns):
            score_details = loaded_details
            score_metrics = pd.read_parquet(score_metrics_path)

    # Older evaluation folders pre-date the modal-score products.  Rebuild
    # them from saved match predictions (or, if necessary, only the saved
    # snapshot outputs) so EDA always emits its requested diagnostic figure.
    if score_details is None or score_metrics is None:
        prediction_path = evaluation_dir / "match_predictions.parquet"
        predictions = (
            pd.read_parquet(prediction_path)
            if prediction_path.is_file()
            else build_match_predictions_from_saved(project_root)
        )
        score_details, score_metrics, score_by_phase = score_prediction_tables(
            predictions, project_root
        )
        _mkdir(evaluation_dir)
        score_details.to_parquet(score_details_path, index=False)
        score_metrics.to_parquet(score_metrics_path, index=False)
        score_by_phase.to_parquet(
            evaluation_dir / "score_prediction_by_phase.parquet", index=False
        )

    generated_figure_names.extend(_plot_score_prediction_diagnostics(score_details, figures))
    if score_metrics is not None:
        display = score_metrics[
            [
                "prediction_set",
                "n_matches",
                "modal_team_goal_mae",
                "modal_total_goals_mae",
                "modal_goal_difference_mae",
                "modal_exact_scoreline_rate",
                "modal_both_teams_within_one_goal_rate",
            ]
        ].copy()
        display["prediction_set"] = display["prediction_set"].replace(
            {"as_of": "As-of", "pre_tournament": "Pre-tournament"}
        )
        for column in (
            "modal_exact_scoreline_rate",
            "modal_both_teams_within_one_goal_rate",
        ):
            display[column] = 100.0 * display[column]
        display = display.rename(
            columns={
                "prediction_set": "forecast",
                "n_matches": "matches",
                "modal_team_goal_mae": "team_goal_mae",
                "modal_total_goals_mae": "total_goals_mae",
                "modal_goal_difference_mae": "goal_difference_mae",
                "modal_exact_scoreline_rate": "exact_modal_scoreline_pct",
                "modal_both_teams_within_one_goal_rate": "within_one_goal_each_pct",
            }
        )
        _write_latex_table(display, tables / "modal_score_prediction_metrics.tex")
        generated_table_names.append("modal_score_prediction_metrics")

    scoreline_frequency_path = evaluation_dir / "scoreline_frequency_check.parquet"
    if scoreline_frequency_path.is_file():
        frequency = pd.read_parquet(scoreline_frequency_path)
        frequency = frequency[frequency["prediction_set"] == "as_of"].copy()
        frequency = frequency.nlargest(12, "observed_count")
        positions = np.arange(len(frequency))
        figure, axis = plt.subplots(figsize=(8.3, 3.8))
        axis.bar(positions - 0.18, frequency["observed_frequency"], width=0.36, label="Observed", color=OKABE_ITO["sky_blue"])
        axis.bar(positions + 0.18, frequency["expected_frequency"], width=0.36, label="Predicted", color=OKABE_ITO["orange"])
        axis.set_xticks(positions, frequency["scoreline_category"], rotation=35, ha="right")
        axis.set_ylabel("Frequency")
        axis.set_title("Observed and predicted scoreline frequencies")
        axis.legend(frameon=False)
        _save_figure(figure, figures, "scoreline_frequency_check")
        generated_figure_names.append("scoreline_frequency_check")

    return generated_figure_names, generated_table_names


def run_exploratory_data_analysis(project_root: str | Path = ".") -> dict[str, Any]:
    """Run the EDA on actual files; optional forecast panels need FULL artifacts."""
    root = ensure_project_imports(project_root)
    analysis = analysis_root(root)
    figures = _mkdir(analysis / "figures")
    tables = _mkdir(analysis / "tables")
    configure_publication_style()
    quality, missingness = _file_quality_tables(root)
    training, teams = _training_data_for_eda(root)
    training_summary = pd.DataFrame(
        [
            {
                "training_start": str(pd.Timestamp(training["date"].min()).date()),
                "training_end": str(pd.Timestamp(training["date"].max()).date()),
                "matches": int(len(training)),
                "teams": int(len(teams)),
                "neutral_share": float(training["neutral"].mean()),
                "draw_frequency": float(np.mean(training["home_score"] == training["away_score"])),
                "mean_total_goals": float((training["home_score"] + training["away_score"]).mean()),
            }
        ]
    )
    all_goals = pd.concat([training["home_score"], training["away_score"]], ignore_index=True).astype(float)
    goal_summary = pd.DataFrame(
        [
            {
                "n_team_match_goal_counts": int(len(all_goals)),
                "mean_goals": float(all_goals.mean()),
                "variance_goals": float(all_goals.var(ddof=1)),
                "dispersion_index": float(all_goals.var(ddof=1) / all_goals.mean()),
                "observed_zero_share": float(np.mean(all_goals == 0)),
                "poisson_zero_share_at_mean": float(poisson.pmf(0, all_goals.mean())),
            }
        ]
    )
    generated_figures: list[str] = []
    generated_tables: list[str] = []
    generated_figures.extend(_plot_manuscript_method_figures(root, figures))
    model_selection_figures, model_selection_tables = _plot_model_selection(root, figures, tables)
    generated_figures.extend(model_selection_figures)
    generated_tables.extend(model_selection_tables)
    _plot_matches_per_year(root, training, figures)
    generated_figures.append("matches_per_year")
    _plot_tournament_mix(training, figures)
    generated_figures.append("tournament_mix")
    _, team_coverage = _plot_team_coverage(training, figures)
    generated_figures.append("matches_per_team")
    _, distribution = _plot_goal_distribution(training, figures)
    generated_figures.append("goals_poisson_fit")
    _plot_draw_rates(training, figures)
    generated_figures.append("draw_rates_by_year")
    _, draw_by_tournament = _plot_draw_rates_by_tournament(training, figures)
    generated_figures.append("draw_rates_by_tournament")
    _, home_summary = _plot_home_advantage(training, figures)
    generated_figures.append("home_advantage")
    _plot_weights(training, figures)
    generated_figures.append("training_weights")
    #quality.to_csv(tables / "data_quality.csv", index=False)
    #missingness.to_csv(tables / "missingness.csv", index=False)
    #training_summary.to_csv(tables / "training_window_summary.csv", index=False)
    #goal_summary.to_csv(tables / "goal_distribution_summary.csv", index=False)
    #ome_summary.to_csv(tables / "home_advantage_summary.csv", index=False)
    #team_coverage.to_csv(tables / "matches_per_team.csv", index=False)
    #draw_by_tournament.to_csv(tables / "draw_rates_by_tournament.csv", index=False)
    _write_latex_table(missingness, tables / "missingness.tex")
    _write_latex_table(quality, tables / "data_quality.tex")
    _write_latex_table(training_summary, tables / "training_window_summary.tex")
    _write_latex_table(goal_summary, tables / "goal_distribution.tex")
    _write_latex_table(home_summary, tables / "home_advantage.tex")
    _write_latex_table(team_coverage, tables / "matches_per_team.tex")
    _write_latex_table(draw_by_tournament, tables / "draw_rates_by_tournament.tex")
    generated_tables.extend(
        [
            "data_quality",
            "training_window_summary",
            "goal_distribution",
            "home_advantage",
            "matches_per_team",
            "draw_rates_by_tournament",
        ]
    )

    result_figures, result_tables = _plot_generated_outputs(root, figures, tables)
    generated_figures.extend(result_figures)
    generated_tables.extend(result_tables)
    result = {
        "mode": "EDA_FROM_ACTUAL_INPUTS" if not result_figures else "EDA_WITH_FULL_SNAPSHOT_OUTPUTS",
        "generated_figures": generated_figures,
        "generated_tables": generated_tables,
        "notes": [
            "Input-data figures are computed directly from the current repository data.",
            "Forecast-dependent figures are omitted until Notebook 1 has produced FULL snapshot CSV files.",
            "No figure or table contains hand-entered numerical results.",
        ],
    }
    return result
