"""Reproducible data products for the Copa com Dados paper notebooks.

This module deliberately delegates model fitting, posterior sampling, score
simulation and tournament simulation to the project's existing ``src`` code.
It only creates an isolated working directory and converts the resulting
native objects into documented, tidy research tables.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import types
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.ticker import PercentFormatter
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import poisson


GRID_MAX = 4
GOAL_MARGINAL_MAX = 10
SCHEMA_VERSION = "1.3"
DEFAULT_START_DATE = "2022-11-19"
DEFAULT_MODEL_NAME = "n_poisson_ranking"
DEFAULT_SEED = 42
DEFAULT_N_SIM = 100_000
# The frozen exploratory extract in the manuscript ends after the last
# complete pre-final World Cup result.  Keeping this date explicit makes the
# annual-volume graphic reproduce the same training window as the tables.
EDA_END_DATE = "2026-07-15"

STAGE_ORDER = [
    "round_of_32",
    "round_of_16",
    "quarterfinals",
    "semifinals",
    "final",
    "champion",
]
STAGE_TO_TOURNAMENT_ATTRIBUTE = {
    "round_of_32": "round_of_32",
    "round_of_16": "round_of_16",
    "quarterfinals": "quarterfinals",
    "semifinals": "semifinals",
    "final": "final",
    "champion": "champion",
}
SCORE_WORDS = ("zero", "one", "two", "three", "four")
OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "grey": "#6E6E6E",
}

# These teams are deliberately fixed rather than selected ex post by their
# final title probabilities.  They retain the leading finalists, the two
# semi-finalists, and four geographically diverse World Cup participants.
TITLE_TRAJECTORY_TEAMS_EN = (
    "Spain",
    "Argentina",
    "France",
    "England",
    "Morocco",
    "Brazil",
    "Mexico",
    "South Africa",
)
TITLE_TRAJECTORY_COLOURS = {
    "Spain": "#0072B2",
    "Argentina": "#E69F00",
    "France": "#009E73",
    "England": "#CC79A7",
    "Morocco": "#D55E00",
    "Brazil": "#56B4E9",
    "Mexico": "#000000",
    "South Africa": "#7A3E00",
}

# The trajectory is intentionally separate from the nine World Cup snapshots:
# it refits the selected model at year-end cutoffs, then adds the two most
# useful 2026 pre-tournament/post-semi-final anchors.  Running it is expensive
# and therefore opt-in.
DEFAULT_PARAMETER_TRAJECTORY_CUTOFFS = (
    ("2023 year-end", "2023-12-31"),
    ("2024 year-end", "2024-12-31"),
    ("2025 year-end", "2025-12-31"),
    ("Pre-FIFA 2026", "2026-05-28"),
    ("Pre-World-Cup 2026", "2026-06-10"),
    ("Post-semi-finals 2026", "2026-07-15"),
)

# The two panels use forecasts made immediately before their named fixtures.
# They are illustrative score matrices, not an aggregate accuracy assessment.
SCORELINE_HEATMAP_EXAMPLES = (
    ("after_round_of_16", "França", "Marrocos", "Quarter-final; actual 2--0"),
    ("after_semifinals", "Espanha", "Argentina", "Final; actual 1--0"),
)

# Keep the compact heatmap readable in the manuscript.  The complete 48-team
# probability matrix remains available as a generated LaTeX table.
STAGE_HEATMAP_N_TEAMS = 12

SNAPSHOT_DISPLAY_LABELS = {
    "pre_fifa": "Pre-FIFA",
    "pre_tournament_post_fifa": "Pre-WC",
    "after_group_round_1": "G1",
    "after_group_round_2": "G2",
    "after_group_stage": "Groups",
    "after_round_of_32": "R32",
    "after_round_of_16": "R16",
    "after_quarterfinals": "QF",
    "after_semifinals": "SF",
}


@dataclass(frozen=True)
class SnapshotSpec:
    """A reproducible as-of state used in the paper analyses."""

    snapshot_id: str
    public_version_label: str
    cutoff_date: str
    phase: str
    cutoff_basis: str
    note: str


def find_project_root(start: str | Path = ".") -> Path:
    """Find the repository root without embedding a machine-specific path."""
    candidate = Path(start).resolve()
    for path in (candidate, *candidate.parents):
        if (path / "src").is_dir() and (path / "data" / "results.csv").is_file():
            return path
    raise FileNotFoundError(
        "Could not locate the project root. Run the notebook from the repository "
        "root or from analysis/."
    )


def analysis_root(project_root: str | Path) -> Path:
    """Return the single directory owned by the paper-analysis workflow."""
    return find_project_root(project_root) / "analysis"


def ensure_project_imports(project_root: str | Path) -> Path:
    """Make the repository importable while leaving its source files unchanged."""
    root = find_project_root(project_root)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    # src.data.loader imports kagglehub at module import time although the
    # paper workflow reads already-present local CSVs and never downloads data.
    # A narrowly scoped placeholder lets us call the project's own preparation
    # function in a minimal environment; any attempt to download still fails
    # loudly with an installation instruction.
    try:
        import kagglehub  # noqa: F401
    except ModuleNotFoundError:
        placeholder = types.ModuleType("kagglehub")

        def _missing_kagglehub(*_: Any, **__: Any) -> None:
            raise RuntimeError(
                "kagglehub is not installed. Install the project's dependencies before "
                "running data_pipeline() to download data."
            )

        placeholder.dataset_download = _missing_kagglehub  # type: ignore[attr-defined]
        sys.modules.setdefault("kagglehub", placeholder)
    return root


@contextlib.contextmanager
def working_directory(path: str | Path) -> Iterator[None]:
    """Temporarily use an isolated directory for code with relative paths."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def _shared_training_inputs(project_root: Path) -> Iterator[None]:
    """Read immutable model inputs from the project root during a runtime fit.

    The trainer uses relative paths for its input files and for posterior
    outputs. This narrow redirection changes only its input reads: the shared
    results and ranking files remain in ``data/`` at the project root, while
    relative posterior paths continue to resolve inside ``analysis/.runtime``.
    """
    from src.simulations import analyze_matchup_events

    results_path = project_root / "data" / "results.csv"
    canonical_path = project_root / "data" / "raw" / "fifa_ranking_2022.csv"
    for path in (results_path, canonical_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required model input is missing: {path}")
    original_prepare_cycle_data = analyze_matchup_events.prepare_cycle_data
    original_loader = analyze_matchup_events.load_ranking_priors

    def prepare_with_shared_results(_: str | Path, *args: Any, **kwargs: Any) -> Any:
        return original_prepare_cycle_data(str(results_path), *args, **kwargs)

    def load_shared_ranking_priors(_: str | Path, teams: Sequence[str]) -> np.ndarray:
        return original_loader(canonical_path, list(teams))

    analyze_matchup_events.prepare_cycle_data = prepare_with_shared_results
    analyze_matchup_events.load_ranking_priors = load_shared_ranking_priors
    try:
        yield
    finally:
        analyze_matchup_events.prepare_cycle_data = original_prepare_cycle_data
        analyze_matchup_events.load_ranking_priors = original_loader


def _mkdir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: str | Path) -> str:
    """Return a content hash suitable for run metadata."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Mapping[str, Any] | list[Any]) -> None:
    """Write deterministic UTF-8 JSON."""
    destination = Path(path)
    _mkdir(destination.parent)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def read_json(path: str | Path) -> dict[str, Any]:
    """Read an analysis metadata file."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _git_revision(project_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _package_versions() -> dict[str, str | None]:
    packages = ("numpy", "pandas", "scipy", "matplotlib", "cmdstanpy", "pyarrow")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _read_results(project_root: str | Path) -> pd.DataFrame:
    results = pd.read_csv(find_project_root(project_root) / "data" / "results.csv")
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    return results


def _calendar_name_to_english(name: object) -> str:
    """Resolve the two display names that differ from the project mapping."""
    ensure_project_imports(find_project_root())
    from src.constants import TEAM_MAP_PT_TO_EN

    aliases = {
        "Estados Unidos": "United States",
        "Países Baixos (Holanda)": "Netherlands",
    }
    text = str(name)
    return aliases.get(text, TEAM_MAP_PT_TO_EN.get(text, text))


def read_calendar(project_root: str | Path) -> pd.DataFrame:
    """Read the committed public calendar and add code-facing team names."""
    root = find_project_root(project_root)
    calendar_path = root / "docs" / "csv" / "vis" / "calendario.csv"
    calendar = pd.read_csv(calendar_path)
    calendar["date"] = pd.to_datetime(calendar["data"], errors="coerce")
    calendar["home_team_en"] = calendar["mandante"].map(_calendar_name_to_english)
    calendar["away_team_en"] = calendar["visitante"].map(_calendar_name_to_english)
    return calendar


def _website_versions(project_root: Path) -> list[str]:
    """Extract the planned publication labels from the checked-in site source."""
    source = (project_root / "docs" / "js" / "chances.js").read_text(encoding="utf-8")
    match = re.search(
        r"PLANNED_VERSIONS\s*=\s*\[(?P<body>.*?)\];", source, flags=re.DOTALL
    )
    if match is None:
        raise RuntimeError("Could not find PLANNED_VERSIONS in docs/js/chances.js.")
    return re.findall(r"'([^']+)'", match.group("body"))


def snapshot_plan(project_root: str | Path = ".") -> pd.DataFrame:
    """Build the phase plan from the website labels and public calendar.

    The website's date windows are status labels. The cutoffs below instead use
    the final scheduled date in each completed round, which is the conservative
    choice when match kick-off times are unavailable in ``results.csv``.
    """
    root = find_project_root(project_root)
    labels = _website_versions(root)
    expected = [
        "Antes da data FIFA",
        "Antes da Copa - pós data FIFA",
        "Após 1ª rodada da fase de grupos",
        "Após 2ª rodada da fase de grupos",
        "Após a Fase de Grupos",
        "Após os 16-Avos",
        "Após as Oitavas",
        "Após as Quartas",
        "Após as Semifinais",
    ]
    if labels != expected:
        raise RuntimeError(
            "The site publication labels changed. Review the snapshot plan before "
            "running the paper pipeline."
        )

    calendar = read_calendar(root)
    last_dates = (
        calendar.groupby(["fase", "rodada"], dropna=False)["date"].max().to_dict()
    )
    group_1 = last_dates[("Fase de grupos", "1ª rodada")].date().isoformat()
    group_2 = last_dates[("Fase de grupos", "2ª rodada")].date().isoformat()
    group_3 = last_dates[("Fase de grupos", "3ª rodada")].date().isoformat()
    r32 = last_dates[("Fase de 32", "Mata-mata")].date().isoformat()
    r16 = last_dates[("Oitavas de final", "Mata-mata")].date().isoformat()
    qf = last_dates[("Quartas de final", "Mata-mata")].date().isoformat()
    sf = last_dates[("Semifinal", "Mata-mata")].date().isoformat()

    specs = [
        SnapshotSpec(
            "pre_fifa",
            labels[0],
            "2026-05-28",
            "Pre-tournament",
            "Day immediately before the FIFA-window boundary in src.constants.",
            "The public site distinguishes this earlier pre-tournament version.",
        ),
        SnapshotSpec(
            "pre_tournament_post_fifa",
            labels[1],
            "2026-06-10",
            "Pre-tournament",
            "Day before the opening match in the committed calendar.",
            "Primary pre-tournament comparator in the evaluation section.",
        ),
        SnapshotSpec(
            "after_group_round_1",
            labels[2],
            group_1,
            "Group stage, round 1 complete",
            "Last scheduled date of group-stage round 1 in calendario.csv.",
            "The code's display boundary is one day later; this analysis uses the "
            "strict completed-round cutoff to avoid ambiguity.",
        ),
        SnapshotSpec(
            "after_group_round_2",
            labels[3],
            group_2,
            "Group stage, round 2 complete",
            "Last scheduled date of group-stage round 2 in calendario.csv.",
            "The code's display boundary is one day later; this analysis uses the "
            "strict completed-round cutoff to avoid ambiguity.",
        ),
        SnapshotSpec(
            "after_group_stage",
            labels[4],
            group_3,
            "Group stage complete",
            "Last scheduled group-stage date in calendario.csv.",
            "All group-stage results on or before this date are fixed in simulation.",
        ),
        SnapshotSpec(
            "after_round_of_32",
            labels[5],
            r32,
            "Round of 32 complete",
            "Last scheduled round-of-32 date in calendario.csv.",
            "The next knockout phase is simulated conditionally on results known by this date.",
        ),
        SnapshotSpec(
            "after_round_of_16",
            labels[6],
            r16,
            "Round of 16 complete",
            "Last scheduled round-of-16 date in calendario.csv.",
            "The next knockout phase is simulated conditionally on results known by this date.",
        ),
        SnapshotSpec(
            "after_quarterfinals",
            labels[7],
            qf,
            "Quarter-finals complete",
            "Last scheduled quarter-final date in calendario.csv.",
            "The next knockout phase is simulated conditionally on results known by this date.",
        ),
        SnapshotSpec(
            "after_semifinals",
            labels[8],
            sf,
            "Semi-finals complete",
            "Last scheduled semi-final date in calendario.csv.",
            "This is the final pre-final snapshot and the reference-comparison target.",
        ),
    ]
    frame = pd.DataFrame(asdict(spec) for spec in specs)
    frame.insert(0, "snapshot_order", range(1, len(frame) + 1))
    frame["source_site_file"] = "docs/js/chances.js"
    frame["source_schedule_file"] = "docs/csv/vis/calendario.csv"
    return frame


def snapshot_specs(project_root: str | Path = ".") -> list[SnapshotSpec]:
    """Return the typed plan in publication order."""
    plan = snapshot_plan(project_root)
    return [
        SnapshotSpec(
            snapshot_id=row.snapshot_id,
            public_version_label=row.public_version_label,
            cutoff_date=row.cutoff_date,
            phase=row.phase,
            cutoff_basis=row.cutoff_basis,
            note=row.note,
        )
        for row in plan.itertuples(index=False)
    ]


def _source_file_inventory(project_root: Path) -> list[tuple[str, Path, Sequence[str]]]:
    return [
        (
            "results.csv",
            project_root / "data" / "results.csv",
            (
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "tournament",
                "city",
                "country",
                "neutral",
            ),
        ),
        (
            "shootouts.csv",
            project_root / "data" / "shootouts.csv",
            ("date", "home_team", "away_team", "winner", "first_shooter"),
        ),
        (
            "goalscorers.csv",
            project_root / "data" / "goalscorers.csv",
            (
                "date",
                "home_team",
                "away_team",
                "team",
                "scorer",
                "minute",
                "own_goal",
                "penalty",
            ),
        ),
        (
            "world_cup_results.csv (documented input)",
            project_root / "data" / "world_cup_results.csv",
            ("stage", "group", "home_team", "away_team", "home_real", "away_real", "date"),
        ),
        (
            "public_calendar.csv",
            project_root / "docs" / "csv" / "vis" / "calendario.csv",
            ("fase", "rodada", "grupo", "data", "mandante", "visitante"),
        ),
        (
            "fifa_ranking_2010.csv",
            project_root / "data" / "raw" / "fifa_ranking_2010.csv",
            ("date", "semester", "rank", "team", "acronym"),
        ),
        (
            "fifa_ranking_2014.csv",
            project_root / "data" / "raw" / "fifa_ranking_2014.csv",
            ("date", "semester", "rank", "team", "acronym"),
        ),
        (
            "fifa_ranking_2018.csv",
            project_root / "data" / "raw" / "fifa_ranking_2018.csv",
            ("date", "semester", "rank", "team", "acronym"),
        ),
        (
            "fifa_ranking_2022.csv",
            project_root / "data" / "raw" / "fifa_ranking_2022.csv",
            ("date", "semester", "rank", "team", "acronym"),
        ),
        (
            "jogos_2014.csv",
            project_root / "data" / "raw" / "jogos_2014.csv",
            ("date", "home_team", "away_team", "home_score", "away_score", "tournament"),
        ),
        (
            "jogos_2018.csv",
            project_root / "data" / "raw" / "jogos_2018.csv",
            ("date", "home_team", "away_team", "home_score", "away_score", "tournament"),
        ),
        (
            "jogos_2022.csv",
            project_root / "data" / "raw" / "jogos_2022.csv",
            ("date", "home_team", "away_team", "home_score", "away_score", "tournament"),
        ),
        (
            "raw/results.csv (validate.py requirement)",
            project_root / "data" / "raw" / "results.csv",
            ("date", "home_team", "away_team", "home_score", "away_score", "tournament"),
        ),
    ]


def data_inventory(project_root: str | Path = ".") -> pd.DataFrame:
    """Audit documented files and schemas without modifying source data."""
    root = find_project_root(project_root)
    rows: list[dict[str, Any]] = []
    for label, path, expected_columns in _source_file_inventory(root):
        exists = path.is_file()
        row: dict[str, Any] = {
            "file": label,
            "path": str(path.relative_to(root)),
            "exists": exists,
            "expected_columns": ", ".join(expected_columns),
            "n_rows": np.nan,
            "missing_expected_columns": "",
            "read_error": "",
        }
        if exists:
            try:
                frame = pd.read_csv(path, nrows=100)
                row["n_rows"] = sum(1 for _ in path.open(encoding="utf-8")) - 1
                absent = sorted(set(expected_columns) - set(frame.columns))
                row["missing_expected_columns"] = ", ".join(absent)
            except Exception as exc:  # pragma: no cover - surfaced in notebook
                row["read_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def actual_world_cup_results(project_root: str | Path = ".") -> pd.DataFrame:
    """Return 2026 World Cup records with schedule phase and score completeness."""
    root = find_project_root(project_root)
    results = _read_results(root)
    wc = results[
        (results["tournament"] == "FIFA World Cup")
        & (results["date"].dt.year == 2026)
    ].copy()
    wc["home_score"] = pd.to_numeric(wc["home_score"], errors="coerce")
    wc["away_score"] = pd.to_numeric(wc["away_score"], errors="coerce")
    wc["score_observed"] = wc["home_score"].notna() & wc["away_score"].notna()

    calendar = read_calendar(root)
    schedule: dict[tuple[pd.Timestamp, frozenset[str]], tuple[str, str]] = {}
    for row in calendar.itertuples(index=False):
        key = (pd.Timestamp(row.date).normalize(), frozenset((row.home_team_en, row.away_team_en)))
        schedule[key] = (str(row.fase), str(row.rodada))

    phase: list[str | None] = []
    round_name: list[str | None] = []
    for row in wc.itertuples(index=False):
        key = (pd.Timestamp(row.date).normalize(), frozenset((row.home_team, row.away_team)))
        assigned = schedule.get(key, (None, None))
        phase.append(assigned[0])
        round_name.append(assigned[1])
    wc["calendar_phase"] = phase
    wc["calendar_round"] = round_name
    wc["result_status"] = np.where(wc["score_observed"], "observed", "score_missing")
    return wc.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def _cmdstan_available() -> tuple[bool, str]:
    try:
        import cmdstanpy

        path = cmdstanpy.cmdstan_path()
        if Path(path).is_dir():
            return True, str(path)
        return False, "CmdStan path is not a directory."
    except Exception as exc:  # CmdStanPy intentionally optional in FAST mode
        return False, f"{type(exc).__name__}: {exc}"


def run_fast_preflight(project_root: str | Path = ".") -> dict[str, Any]:
    """Run an explicit no-MCMC readiness audit for FAST mode.

    No forecast or posterior output is created here. The report is a preflight
    record only and must not be interpreted as a sampled result.
    """
    root = ensure_project_imports(project_root)
    inventory = data_inventory(root)
    plan = snapshot_plan(root)

    prepared: dict[str, Any] = {}
    preparation_error: str | None = None
    try:
        from src.data import prepare_cycle_data

        data, teams, _ = prepare_cycle_data(
            str(root / "data" / "results.csv"),
            DEFAULT_START_DATE,
            end_date="2026-07-15",
            apply_decay=True,
        )
        prepared = {
            "n_training_matches_at_final_cutoff": int(len(data)),
            "n_teams_at_final_cutoff": int(len(teams)),
            "max_training_date": str(pd.Timestamp(data["date"].max()).date()),
        }
    except Exception as exc:  # pragma: no cover - belongs in the saved report
        preparation_error = f"{type(exc).__name__}: {exc}"

    actual = actual_world_cup_results(root)
    cmdstan_ok, cmdstan_detail = _cmdstan_available()
    pyarrow_ok = False
    try:
        import pyarrow  # noqa: F401

        pyarrow_ok = True
    except ImportError:
        pyarrow_ok = False

    report = {
        "mode": "FAST_PREFLIGHT",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "git_commit": _git_revision(root),
        "dependencies": {
            "package_versions": _package_versions(),
            "cmdstan_ready": cmdstan_ok,
            "cmdstan_detail": cmdstan_detail,
            "parquet_engine_ready": pyarrow_ok,
        },
        # These in-memory tables are displayed by Notebook 1.  Preflight is
        # intentionally transient: it must not create an artifacts/preflight
        # directory or write audit files.
        "input_inventory": inventory,
        "snapshot_plan": plan,
        "actual_results_completeness": actual,
        "results_completeness": {
            "world_cup_rows": int(len(actual)),
            "observed_scores": int(actual["score_observed"].sum()),
            "missing_scores": int((~actual["score_observed"]).sum()),
            "missing_score_matches": actual.loc[
                ~actual["score_observed"], ["date", "home_team", "away_team"]
            ].assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
        },
        "prepared_data_check": prepared,
        "prepared_data_error": preparation_error,
        "blocking_conditions_for_full_mode": [
            message
            for message, condition in (
                ("CmdStanPy and a CmdStan installation are required for MCMC.", not cmdstan_ok),
                ("pyarrow is required for the requested Parquet outputs.", not pyarrow_ok),
            )
            if condition
        ],
        "important_data_notes": [
            "data/world_cup_results.csv is documented in data/README.md but is absent "
            "from this checkout; the active _load_schedule_orientations() implementation "
            "reads data/results.csv.",
            "data/raw/results.csv is absent, so src.model_sel.validate cannot be run on "
            "this checkout without an additional historical results file.",
            "FAST_PREFLIGHT validates source data and orchestration only. It creates no model "
            "probabilities, posterior summaries, or evaluation scores.",
        ],
    }
    return report


def _require_full_dependencies() -> None:
    """Fail before any fit when required FULL-mode dependencies are unavailable."""
    cmdstan_ok, cmdstan_detail = _cmdstan_available()
    missing: list[str] = []
    if not cmdstan_ok:
        missing.append(f"CmdStanPy/CmdStan ({cmdstan_detail})")
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        missing.append("pyarrow")
    if missing:
        joined = "; ".join(missing)
        raise RuntimeError(
            "FULL mode cannot start because the following requirement(s) are missing: "
            f"{joined}. Run `uv sync --extra jupyter`, install pyarrow in the "
            "project environment, and configure CmdStan for cmdstanpy before rerunning."
        )


def _copy_if_changed(source: Path, destination: Path) -> None:
    """Copy an input to an analysis-owned runtime directory when it changed."""
    _mkdir(destination.parent)
    if destination.exists() and sha256_file(source) == sha256_file(destination):
        return
    shutil.copy2(source, destination)


def prepare_isolated_runtime(project_root: str | Path, snapshot_id: str) -> Path:
    """Create the per-snapshot directory used by the unmodified project code.

    The Stan source and generated posterior archives remain isolated in the
    runtime. Immutable match inputs are read from the project-root ``data/``
    directory, so no copies of them are created for individual snapshots.
    """
    root = find_project_root(project_root)
    runtime = analysis_root(root) / ".runtime" / snapshot_id
    copies = {
        root / "stan_models" / "n_poisson_ranking.stan": runtime
        / "stan_models"
        / "n_poisson_ranking.stan",
    }
    for source, destination in copies.items():
        if not source.is_file():
            raise FileNotFoundError(f"Required project input is missing: {source}")
        _copy_if_changed(source, destination)
    _mkdir(runtime / "models")
    return runtime


def _summary(values: np.ndarray) -> dict[str, float]:
    numeric = np.asarray(values, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(numeric)),
        "sd": float(np.std(numeric, ddof=1)),
        "q2_5": float(np.quantile(numeric, 0.025)),
        "q97_5": float(np.quantile(numeric, 0.975)),
        "n_draws": int(len(numeric)),
    }


def posterior_parameter_summaries(
    draws_path: str | Path,
    teams: Sequence[str],
    project_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise native Stan draws without changing their parameterisation."""
    root = ensure_project_imports(project_root)
    from src.constants import TEAM_MAP_EN_TO_PT

    loaded = np.load(draws_path)
    team_rows: list[dict[str, Any]] = []
    for parameter in ("attack", "defense"):
        if parameter not in loaded.files:
            continue
        values = np.asarray(loaded[parameter])
        if values.ndim != 2:
            continue
        for index, team in enumerate(teams):
            stats = _summary(values[:, index])
            team_rows.append(
                {
                    "team_en": team,
                    "team_display": TEAM_MAP_EN_TO_PT.get(team, team),
                    "parameter": parameter,
                    "parameter_scale": "log_rate_component",
                    **stats,
                }
            )

    global_rows: list[dict[str, Any]] = []
    for parameter in loaded.files:
        values = np.asarray(loaded[parameter])
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
            continue
        global_rows.append({"parameter": parameter, **_summary(values)})
    return pd.DataFrame(team_rows), pd.DataFrame(global_rows)


def _stage_probability_table(
    tournament_result: Any,
    project_root: str | Path,
) -> pd.DataFrame:
    """Convert the project's TournamentResult counts into long probabilities."""
    root = ensure_project_imports(project_root)
    from src.constants import GROUPS, TEAM_MAP_EN_TO_PT

    n_sim = int(tournament_result.counts)
    teams = [team for group in GROUPS.values() for team in group]
    rows: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        counts = getattr(tournament_result, STAGE_TO_TOURNAMENT_ATTRIBUTE[stage])
        for team in teams:
            count = int(counts.get(team, 0))
            probability = count / n_sim
            rows.append(
                {
                    "team_en": team,
                    "team_display": TEAM_MAP_EN_TO_PT.get(team, team),
                    "stage": stage,
                    "count": count,
                    "n_simulations": n_sim,
                    "probability": probability,
                    "mcse": math.sqrt(probability * (1.0 - probability) / n_sim),
                }
            )
    return pd.DataFrame(rows)


def _goal_marginal_rows(
    goals: np.ndarray,
    side: str,
    common: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in range(GOAL_MARGINAL_MAX + 1):
        rows.append(
            {
                **common,
                "side": side,
                "goals": value,
                "category": str(value),
                "probability": float(np.mean(goals == value)),
            }
        )
    rows.append(
        {
            **common,
            "side": side,
            "goals": np.nan,
            "category": f">{GOAL_MARGINAL_MAX}",
            "probability": float(np.mean(goals > GOAL_MARGINAL_MAX)),
        }
    )
    return rows


def matchup_tables_from_project_exporter(
    teams_list: Sequence[str],
    posterior_draws: Mapping[str, np.ndarray],
    n_sim: int,
    project_root: str | Path,
    results_path: str | Path,
    batch_size: int = 128,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create paper tables from the project's native all-matchup exporter.

    The source project's ``build_all_matchups_dataframe_mc`` is the sole source
    of the published 1X2 and 0--4 score-grid values. This helper reshapes that
    native percentage output into documented unit-scale tables and uses the
    source goal simulator only for the additional goal-marginal diagnostics.
    """
    ensure_project_imports(project_root)
    from src.constants import GROUPS, TEAM_MAP_EN_TO_PT, TEAM_MAP_PT_TO_EN
    from src.simulations.utils import (
        _simulate_match_goals,
        build_all_matchups_dataframe_mc,
    )
    from src.tournament.bayesian import _match_lambdas, _sample_posterior

    sampled_attack, sampled_defense, sampled_rho, sampled_eta = _sample_posterior(
        dict(posterior_draws), n_sim=n_sim, seed=seed
    )
    wc_teams = [team for group in GROUPS.values() for team in group]
    team_index = {team: index for index, team in enumerate(teams_list)}
    missing = sorted(set(wc_teams) - set(team_index))
    if missing:
        raise RuntimeError(
            "The fitted model is missing World Cup teams needed for all-matchup "
            f"generation: {missing}"
        )

    project_export = build_all_matchups_dataframe_mc(
        list(teams_list),
        wc_teams,
        sampled_attack,
        sampled_defense,
        sampled_rho,
        sampled_eta,
        n_sim=n_sim,
        results_path=results_path,
        batch_size=batch_size,
    ).copy()
    expected_matchups = len(wc_teams) * (len(wc_teams) - 1) // 2
    if len(project_export) != expected_matchups:
        raise RuntimeError(
            "The project all-matchup exporter returned an unexpected number of "
            f"rows ({len(project_export)} instead of {expected_matchups})."
        )

    score_columns = [
        f"{SCORE_WORDS[home_goals]}_{SCORE_WORDS[away_goals]}"
        for home_goals in range(GRID_MAX + 1)
        for away_goals in range(GRID_MAX + 1)
    ]
    required_columns = {"home_team", "away_team", "home_win", "draw", "away_win", *score_columns}
    missing_columns = sorted(required_columns - set(project_export.columns))
    if missing_columns:
        raise RuntimeError(
            "The project all-matchup exporter is missing expected columns: "
            f"{missing_columns}"
        )

    display_to_en = {str(display): str(team) for display, team in TEAM_MAP_PT_TO_EN.items()}
    for team in wc_teams:
        display_to_en.setdefault(TEAM_MAP_EN_TO_PT.get(team, team), team)
    home_en = project_export["home_team"].map(display_to_en)
    away_en = project_export["away_team"].map(display_to_en)
    if home_en.isna().any() or away_en.isna().any():
        unresolved = sorted(
            set(project_export.loc[home_en.isna(), "home_team"])
            | set(project_export.loc[away_en.isna(), "away_team"])
        )
        raise RuntimeError(f"Could not map exporter team labels to source names: {unresolved}")

    grid_percent = project_export[score_columns].apply(pd.to_numeric, errors="coerce")
    summary = pd.DataFrame(
        {
            "home_team_en": home_en.to_numpy(),
            "away_team_en": away_en.to_numpy(),
            "home_team_display": project_export["home_team"].to_numpy(),
            "away_team_display": project_export["away_team"].to_numpy(),
            "n_simulations": int(n_sim),
            "score_grid_max": GRID_MAX,
            "home_win_probability": pd.to_numeric(project_export["home_win"], errors="coerce") / 100.0,
            "draw_probability": pd.to_numeric(project_export["draw"], errors="coerce") / 100.0,
            "away_win_probability": pd.to_numeric(project_export["away_win"], errors="coerce") / 100.0,
            "grid_probability_mass": grid_percent.sum(axis=1) / 100.0,
            "expected_home_goals": np.nan,
            "expected_away_goals": np.nan,
            "orientation_source": "project_all_matchups_exporter",
        }
    )
    summary["residual_probability_mass"] = (
        1.0 - summary["grid_probability_mass"]
    ).clip(lower=0.0)

    score_frames: list[pd.DataFrame] = []
    common_columns = [
        "home_team_en",
        "away_team_en",
        "home_team_display",
        "away_team_display",
        "n_simulations",
        "score_grid_max",
    ]
    for home_goals in range(GRID_MAX + 1):
        for away_goals in range(GRID_MAX + 1):
            column = f"{SCORE_WORDS[home_goals]}_{SCORE_WORDS[away_goals]}"
            score_frames.append(
                summary[common_columns].assign(
                    home_goals=home_goals,
                    away_goals=away_goals,
                    probability=grid_percent[column].to_numpy(dtype=float) / 100.0,
                )
            )
    scorelines = pd.concat(score_frames, ignore_index=True)

    marginal_rows: list[dict[str, Any]] = []
    for start in range(0, len(summary), batch_size):
        batch = summary.iloc[start : start + batch_size]
        home_indices = np.array(
            [team_index[str(team)] for team in batch["home_team_en"]], dtype=int
        )
        away_indices = np.array(
            [team_index[str(team)] for team in batch["away_team_en"]], dtype=int
        )
        goals_home, goals_away = _simulate_match_goals(
            sampled_attack,
            sampled_defense,
            sampled_rho,
            sampled_eta,
            home_indices,
            away_indices,
            n_sim=n_sim,
        )
        if goals_home.ndim == 1:
            goals_home = goals_home[:, None]
            goals_away = goals_away[:, None]
        lambdas_home, lambdas_away = _match_lambdas(
            sampled_attack,
            sampled_defense,
            home_indices,
            away_indices,
            sampled_eta,
        )
        if lambdas_home.ndim == 1:
            lambdas_home = lambdas_home[:, None]
            lambdas_away = lambdas_away[:, None]
        summary.loc[batch.index, "expected_home_goals"] = np.mean(lambdas_home, axis=0)
        summary.loc[batch.index, "expected_away_goals"] = np.mean(lambdas_away, axis=0)
        for column, row in enumerate(batch.itertuples(index=False)):
            common = {
                "home_team_en": str(row.home_team_en),
                "away_team_en": str(row.away_team_en),
                "home_team_display": str(row.home_team_display),
                "away_team_display": str(row.away_team_display),
                "n_simulations": int(row.n_simulations),
                "score_grid_max": int(row.score_grid_max),
            }
            marginal_rows.extend(_goal_marginal_rows(goals_home[:, column], "home", common))
            marginal_rows.extend(_goal_marginal_rows(goals_away[:, column], "away", common))

    return scorelines, summary, pd.DataFrame(marginal_rows), project_export


def _snapshot_directory(project_root: Path, snapshot_id: str) -> Path:
    return analysis_root(project_root) / "artifacts" / "snapshots" / snapshot_id


def snapshot_is_complete(project_root: str | Path, snapshot_id: str) -> bool:
    """Return True only when all required FULL-mode products are present."""
    root = find_project_root(project_root)
    folder = _snapshot_directory(root, snapshot_id)
    required = (
        "scoreline_probabilities.csv",
        "matchup_summary.csv",
        "goal_marginals.csv",
        "stage_probabilities.csv",
        "projected_bracket.csv",
        "team_parameter_summaries.csv",
        "global_parameter_summaries.csv",
        "realized_world_cup_results.csv",
        "project_all_matchups_export.csv",
        "run_metadata.csv",
    )
    return all((folder / name).is_file() for name in required)


def _legacy_snapshot_is_readable(project_root: Path, snapshot_id: str) -> bool:
    """Recognise snapshots produced before the CSV-only output policy.

    This preserves read-only access for existing figures and tables. It is not
    used as a generation checkpoint, so a new FULL run still regenerates the
    CSV products required by :func:`snapshot_is_complete`.
    """
    folder = _snapshot_directory(project_root, snapshot_id)
    required = (
        "scoreline_probabilities.parquet",
        "matchup_summary.parquet",
        "goal_marginals.parquet",
        "stage_probabilities.parquet",
        "projected_bracket.parquet",
        "team_parameter_summaries.parquet",
        "global_parameter_summaries.parquet",
        "realized_world_cup_results.parquet",
        "project_all_matchups_export.csv",
    )
    return all((folder / name).is_file() for name in required)


def _write_snapshot_table(
    frame: pd.DataFrame,
    folder: Path,
    name: str,
) -> None:
    """Persist a per-snapshot product in its local CSV-only format."""
    frame.to_csv(folder / f"{name}.csv", index=False)


def _reference_comparison(
    project_export: pd.DataFrame,
    reference_path: Path,
    tolerance_pct_points: float,
) -> pd.DataFrame:
    """Compare native exporter percentages to the committed reference CSV."""
    if not reference_path.is_file():
        return pd.DataFrame(
            [
                {
                    "status": "reference_missing",
                    "metric": "all_matchups",
                    "max_abs_difference_pct_points": np.nan,
                    "tolerance_pct_points": tolerance_pct_points,
                    "within_tolerance": False,
                    "note": f"Reference file not found: {reference_path}",
                }
            ]
        )

    reference = pd.read_csv(reference_path)
    generated = project_export.copy()
    generated_columns = [
        column
        for column in generated.columns
        if column in reference.columns and column not in {"home_team", "away_team"}
    ]
    merged = reference.merge(
        generated,
        on=["home_team", "away_team"],
        how="outer",
        suffixes=("_reference", "_generated"),
        indicator=True,
    )
    rows: list[dict[str, Any]] = []
    for column in generated_columns:
        reference_column = f"{column}_reference"
        generated_column = f"{column}_generated"
        if reference_column not in merged or generated_column not in merged:
            continue
        comparable = merged[merged["_merge"] == "both"].copy()
        difference = (
            pd.to_numeric(comparable[reference_column], errors="coerce")
            - pd.to_numeric(comparable[generated_column], errors="coerce")
        ).abs()
        maximum = float(difference.max()) if difference.notna().any() else np.nan
        rows.append(
            {
                "status": "compared",
                "metric": column,
                "n_reference_rows": int(len(reference)),
                "n_generated_rows": int(len(generated)),
                "n_matched_rows": int((merged["_merge"] == "both").sum()),
                "max_abs_difference_pct_points": maximum,
                "mean_abs_difference_pct_points": float(difference.mean()),
                "tolerance_pct_points": tolerance_pct_points,
                "within_tolerance": bool(maximum <= tolerance_pct_points),
                "note": "Native project percentages were compared directly without rescaling.",
            }
        )
    return pd.DataFrame(rows)


def _runtime_metadata(
    project_root: Path,
    snapshot: SnapshotSpec,
    n_sim: int,
    seed: int,
    runtime_seconds: float,
    training_rows: int,
    training_teams: int,
    known_results: Mapping[Any, Any] | None,
    known_knockout_winners: Mapping[Any, Any] | None,
) -> dict[str, Any]:
    """Build a self-contained record of a finished snapshot run."""
    source_paths = {
        "results_csv": project_root / "data" / "results.csv",
        "shootouts_csv": project_root / "data" / "shootouts.csv",
        "aet_winners_csv": project_root / "data" / "aet_winners.csv",
        "fifa_ranking_2022_csv": project_root / "data" / "raw" / "fifa_ranking_2022.csv",
        "stan_model": project_root / "stan_models" / "n_poisson_ranking.stan",
        "cutoff_trainer": project_root / "src" / "simulations" / "analyze_matchup_events.py",
        "tournament_simulator": project_root / "src" / "tournament" / "bayesian.py",
        "all_matchup_exporter": project_root / "src" / "simulations" / "utils.py",
        "public_version_labels": project_root / "docs" / "js" / "chances.js",
        "public_calendar": project_root / "docs" / "csv" / "vis" / "calendario.csv",
        "reference_all_matchups_csv": project_root
        / "docs"
        / "csv"
        / "previsoes"
        / "all_matchups.csv",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "FULL",
        "snapshot": asdict(snapshot),
        "git_commit": _git_revision(project_root),
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "runtime_seconds": runtime_seconds,
        "simulation": {
            "n_simulations": int(n_sim),
            "seed": int(seed),
            "posterior_sample_seed": int(seed),
            "score_grid_max": GRID_MAX,
            "goal_marginal_max": GOAL_MARGINAL_MAX,
        },
        "sampler_settings_from_src_model_sel_validate": {
            "chains": 4,
            "iter_sampling": 5000,
            "seed": 123,
            "iter_warmup": "CmdStanPy default (not explicitly set in project code)",
        },
        "model": {
            "name": DEFAULT_MODEL_NAME,
            "stan_file": "stan_models/n_poisson_ranking.stan",
            "training_start_date": DEFAULT_START_DATE,
            "training_cutoff_date": snapshot.cutoff_date,
            "training_rows": int(training_rows),
            "training_teams": int(training_teams),
            "known_world_cup_scores": 0 if known_results is None else len(known_results),
            "known_knockout_winners": 0
            if known_knockout_winners is None
            else len(known_knockout_winners),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "package_versions": _package_versions(),
            "cmdstan": _cmdstan_available()[1],
        },
        "input_sha256": {
            label: sha256_file(path) if path.is_file() else None
            for label, path in source_paths.items()
        },
    }


def generate_snapshot(
    project_root: str | Path,
    snapshot: SnapshotSpec,
    *,
    n_sim: int = DEFAULT_N_SIM,
    seed: int = DEFAULT_SEED,
    force_retrain: bool = False,
    tolerance_pct_points: float = 0.05,
) -> dict[str, Any]:
    """Fit and simulate one as-of snapshot using only project implementation code."""
    root = ensure_project_imports(project_root)
    _require_full_dependencies()
    destination = _snapshot_directory(root, snapshot.snapshot_id)
    if snapshot_is_complete(root, snapshot.snapshot_id) and not force_retrain:
        return {
            "snapshot_id": snapshot.snapshot_id,
            "status": "skipped_complete_checkpoint",
            "directory": str(destination),
        }

    _mkdir(destination)
    runtime = prepare_isolated_runtime(root, snapshot.snapshot_id)
    start = time.perf_counter()
    from src.simulations.analyze_matchup_events import load_known_state, train_model_up_to

    # This routine reads only fixed inputs, so it can run directly against the
    # canonical data directory without creating any runtime files.
    with working_directory(root):
        known_results, known_knockout_winners = load_known_state(snapshot.cutoff_date)

    with working_directory(runtime), _shared_training_inputs(root):
        # The trainer keeps its original relative output path, which resolves
        # under analysis/.runtime, while its input reads are redirected above.
        from src.data import prepare_cycle_data
        from src.model.bayesian import BayesianDixonColesModel
        from src.tournament.bayesian import BayesianWorldCup2026

        model_path_relative = train_model_up_to(
            snapshot.cutoff_date,
            force_retrain=force_retrain,
        )
        model_path = (runtime / model_path_relative).resolve()
        model = BayesianDixonColesModel(model_path)
        simulator = BayesianWorldCup2026(
            model,
            seed=seed,
            known_results=known_results,
            known_ko_winners=known_knockout_winners,
            schedule_path=root / "data" / "results.csv",
            export_all_matchups=False,
        )
        tournament_result = simulator.simulate(n=n_sim)
        training_data, training_teams, _ = prepare_cycle_data(
            root / "data" / "results.csv",
            DEFAULT_START_DATE,
            end_date=snapshot.cutoff_date,
            apply_decay=True,
        )
        scorelines, matchups, goal_marginals, project_export = matchup_tables_from_project_exporter(
            model.teams,
            model.draws,
            n_sim=n_sim,
            project_root=root,
            results_path=root / "data" / "results.csv",
            seed=seed,
        )
        stage_probabilities = _stage_probability_table(tournament_result, root)
        bracket = simulator.last_bracket
        if bracket is None:
            raise RuntimeError("The project simulator did not populate last_bracket.")
        team_parameters, global_parameters = posterior_parameter_summaries(
            model_path, model.teams, root
        )

    runtime_seconds = time.perf_counter() - start
    actual = actual_world_cup_results(root)
    for frame in (
        scorelines,
        matchups,
        goal_marginals,
        stage_probabilities,
        bracket,
        team_parameters,
        global_parameters,
        actual,
    ):
        frame.insert(0, "snapshot_id", snapshot.snapshot_id)
        frame.insert(1, "cutoff_date", snapshot.cutoff_date)

    _write_snapshot_table(scorelines, destination, "scoreline_probabilities")
    _write_snapshot_table(matchups, destination, "matchup_summary")
    _write_snapshot_table(goal_marginals, destination, "goal_marginals")
    project_export.to_csv(destination / "project_all_matchups_export.csv", index=False)
    _write_snapshot_table(stage_probabilities, destination, "stage_probabilities")
    _write_snapshot_table(bracket, destination, "projected_bracket")
    _write_snapshot_table(team_parameters, destination, "team_parameter_summaries")
    _write_snapshot_table(global_parameters, destination, "global_parameter_summaries")
    _write_snapshot_table(actual, destination, "realized_world_cup_results")

    metadata = _runtime_metadata(
        root,
        snapshot,
        n_sim=n_sim,
        seed=seed,
        runtime_seconds=runtime_seconds,
        training_rows=len(training_data),
        training_teams=len(training_teams),
        known_results=known_results,
        known_knockout_winners=known_knockout_winners,
    )
    pd.DataFrame(
        [
            {
                "snapshot_id": snapshot.snapshot_id,
                "cutoff_date": snapshot.cutoff_date,
                "n_simulations": n_sim,
                "seed": seed,
                "runtime_seconds": runtime_seconds,
                "training_rows": len(training_data),
                "training_teams": len(training_teams),
                "git_commit": metadata["git_commit"],
            }
        ]
    ).to_csv(destination / "run_metadata.csv", index=False)

    if snapshot.snapshot_id == "after_semifinals":
        comparison = _reference_comparison(
            project_export,
            root / "docs" / "csv" / "previsoes" / "all_matchups.csv",
            tolerance_pct_points=tolerance_pct_points,
        )
        _write_snapshot_table(comparison, destination, "reference_comparison")

    return {
        "snapshot_id": snapshot.snapshot_id,
        "status": "generated",
        "directory": str(destination),
        "runtime_seconds": runtime_seconds,
    }


def run_all_snapshots(
    project_root: str | Path = ".",
    *,
    n_sim: int = DEFAULT_N_SIM,
    seed: int = DEFAULT_SEED,
    force_retrain: bool = False,
    tolerance_pct_points: float = 0.05,
) -> pd.DataFrame:
    """Generate the full publication sequence, skipping completed checkpoints."""
    root = find_project_root(project_root)
    _require_full_dependencies()
    outputs = [
        generate_snapshot(
            root,
            spec,
            n_sim=n_sim,
            seed=seed,
            force_retrain=force_retrain,
            tolerance_pct_points=tolerance_pct_points,
        )
        for spec in snapshot_specs(root)
    ]
    manifest = pd.DataFrame(outputs)
    return manifest


def parameter_trajectory_plan() -> pd.DataFrame:
    """Return the costly, opt-in posterior-refit schedule for strength paths."""
    return pd.DataFrame(
        [
            {
                "trajectory_order": order,
                "trajectory_label": label,
                "cutoff_date": cutoff,
                "note": "Posterior parameters are re-estimated using only data on or before this cutoff.",
            }
            for order, (label, cutoff) in enumerate(DEFAULT_PARAMETER_TRAJECTORY_CUTOFFS, start=1)
        ]
    )


def run_parameter_trajectory(
    project_root: str | Path = ".",
    *,
    force_retrain: bool = False,
) -> dict[str, pd.DataFrame]:
    """Refit the selected model at year-end/tournament cutoffs for Appendix paths.

    This workflow intentionally uses the project's existing ``train_model_up_to``
    routine and the selected Stan file.  It does not substitute a rolling
    estimate or an independently implemented strength model.  Because every
    cutoff triggers a full Stan fit when no checkpoint is present, notebooks
    should call it only in FULL mode and only when the appendix figure is
    required.
    """
    root = ensure_project_imports(project_root)
    _require_full_dependencies()
    output = _mkdir(analysis_root(root) / "artifacts" / "parameter_trajectory")
    plan = parameter_trajectory_plan()
    plan_path = output / "parameter_trajectory_plan.json"
    write_json(plan_path, plan.to_dict(orient="records"))
    team_frames: list[pd.DataFrame] = []
    global_frames: list[pd.DataFrame] = []

    for row in plan.itertuples(index=False):
        trajectory_id = f"cutoff_{pd.Timestamp(row.cutoff_date).strftime('%Y%m%d')}"
        team_path = output / f"{trajectory_id}_team_parameters.parquet"
        global_path = output / f"{trajectory_id}_global_parameters.parquet"
        if team_path.is_file() and global_path.is_file() and not force_retrain:
            team = pd.read_parquet(team_path)
            global_parameters = pd.read_parquet(global_path)
        else:
            runtime = prepare_isolated_runtime(root, f"parameter_trajectory_{trajectory_id}")
            started = time.perf_counter()
            with working_directory(runtime), _shared_training_inputs(root):
                from src.model.bayesian import BayesianDixonColesModel
                from src.simulations.analyze_matchup_events import train_model_up_to

                model_path_relative = train_model_up_to(
                    str(row.cutoff_date), force_retrain=force_retrain
                )
                model_path = (runtime / model_path_relative).resolve()
                model = BayesianDixonColesModel(model_path)
                team, global_parameters = posterior_parameter_summaries(
                    model_path, model.teams, root
                )
            team.insert(0, "trajectory_id", trajectory_id)
            team.insert(1, "trajectory_order", int(row.trajectory_order))
            team.insert(2, "trajectory_label", str(row.trajectory_label))
            team.insert(3, "cutoff_date", str(row.cutoff_date))
            team["runtime_seconds"] = time.perf_counter() - started
            global_parameters.insert(0, "trajectory_id", trajectory_id)
            global_parameters.insert(1, "trajectory_order", int(row.trajectory_order))
            global_parameters.insert(2, "trajectory_label", str(row.trajectory_label))
            global_parameters.insert(3, "cutoff_date", str(row.cutoff_date))
            global_parameters.to_parquet(global_path, index=False)
            team.to_parquet(team_path, index=False)
        team_frames.append(team)
        global_frames.append(global_parameters)

    all_team = pd.concat(team_frames, ignore_index=True)
    all_global = pd.concat(global_frames, ignore_index=True)
    all_team.to_parquet(output / "team_parameter_trajectory.parquet", index=False)
    all_global.to_parquet(output / "global_parameter_trajectory.parquet", index=False)
    return {
        "team_parameter_trajectory": all_team,
        "global_parameter_trajectory": all_global,
    }


def _available_snapshot_ids(project_root: Path) -> list[str]:
    return [
        spec.snapshot_id
        for spec in snapshot_specs(project_root)
        if snapshot_is_complete(project_root, spec.snapshot_id)
        or _legacy_snapshot_is_readable(project_root, spec.snapshot_id)
    ]


def _read_snapshot_table(project_root: Path, snapshot_id: str, name: str) -> pd.DataFrame:
    folder = _snapshot_directory(project_root, snapshot_id)
    csv_path = folder / f"{name}.csv"
    if csv_path.is_file():
        return pd.read_csv(csv_path)

    # Read-only compatibility with snapshots generated before the local
    # CSV-only policy. New snapshots are always written as CSV files.
    parquet_path = folder / f"{name}.parquet"
    if parquet_path.is_file():
        return pd.read_parquet(parquet_path)
    raise FileNotFoundError(f"Snapshot table not found: {csv_path}")


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

    The function reads saved Parquet tables only; it performs no refitting or
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


def _diagram_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = "#31496D",
    fontsize: float = 8.0,
) -> None:
    """Add one rounded, labelled box to a manuscript schematic axis."""
    patch = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
        zorder=2,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#17283F",
        linespacing=1.22,
        zorder=3,
    )


def _diagram_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str | None = None,
    label_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Draw a readable directed connection between two manuscript boxes."""
    arrow = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.05,
        color="#31496D",
        connectionstyle="arc3,rad=0",
        zorder=1,
    )
    axis.add_patch(arrow)
    if label:
        midpoint = ((start[0] + end[0]) / 2 + label_offset[0], (start[1] + end[1]) / 2 + label_offset[1])
        axis.text(
            *midpoint,
            label,
            ha="center",
            va="center",
            fontsize=6.7,
            color="#31496D",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.94},
            zorder=4,
        )


def _plot_predictive_pipeline(output: Path) -> list[Path]:
    """Render the code-derived path from inputs to public forecast products."""
    figure, axis = plt.subplots(figsize=(12.2, 5.6))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")

    _diagram_box(
        axis, 0.03, 0.67, 0.18, 0.17,
        "Historical international\nmatch results\nresults.csv",
        facecolor="#E8F1FA",
    )
    _diagram_box(
        axis, 0.03, 0.34, 0.18, 0.17,
        "FIFA ranking points\nfifa_ranking_2022.csv",
        facecolor="#FFF2CC",
    )
    _diagram_box(
        axis, 0.03, 0.06, 0.18, 0.17,
        "Calendar and known\nWorld Cup results\nat a cutoff",
        facecolor="#E5F5EA",
    )
    _diagram_box(
        axis, 0.29, 0.66, 0.20, 0.20,
        "prepare_cycle_data\ncycle and team filters\ntournament and time weights",
        facecolor="#DDEBF7",
    )
    _diagram_box(
        axis, 0.56, 0.66, 0.19, 0.20,
        "Stan fit: N Poisson Ranking\nattack and defence effects\nposterior parameter draws",
        facecolor="#FCE4D6",
    )
    _diagram_box(
        axis, 0.81, 0.66, 0.16, 0.20,
        "Posterior predictive\nscoreline grid and\nwin-draw-loss probabilities",
        facecolor="#EDE7F6",
        fontsize=7.4,
    )
    _diagram_box(
        axis, 0.49, 0.20, 0.26, 0.23,
        "Tournament simulator\n12 groups; best eight third places;\nfixed knockout tree\ncompleted scores fixed at cutoff",
        facecolor="#E5F5EA",
        fontsize=7.5,
    )
    _diagram_box(
        axis, 0.81, 0.20, 0.16, 0.23,
        "Public products\nstage and title probabilities\nprojected bracket\nmatch-up probabilities",
        facecolor="#D9EAD3",
        fontsize=7.3,
    )

    _diagram_arrow(axis, (0.21, 0.755), (0.29, 0.755))
    _diagram_arrow(axis, (0.21, 0.425), (0.56, 0.70), label="ranking-informed prior", label_offset=(0.0, 0.02))
    _diagram_arrow(axis, (0.49, 0.755), (0.56, 0.755))
    _diagram_arrow(axis, (0.75, 0.755), (0.81, 0.755))
    _diagram_arrow(axis, (0.655, 0.66), (0.62, 0.43), label="sample posterior draws", label_offset=(-0.08, 0.0))
    _diagram_arrow(axis, (0.89, 0.66), (0.70, 0.43), label="match distributions", label_offset=(0.08, 0.0))
    _diagram_arrow(axis, (0.21, 0.145), (0.49, 0.275), label="cutoff-aware known results", label_offset=(0.0, -0.035))
    _diagram_arrow(axis, (0.75, 0.315), (0.81, 0.315))
    axis.set_title("Predictive pipeline implemented for the 2026 World Cup", pad=13, fontweight="bold")
    figure.text(
        0.5,
        0.015,
        "The snapshot workflow retrains to each cutoff, fixes only results known then, and simulates all later fixtures.",
        ha="center",
        fontsize=8,
        color="#425466",
    )
    return _save_figure(figure, output, "predictive_pipeline")


def _plot_hierarchical_prior(output: Path) -> list[Path]:
    """Render the selected Stan model hierarchy directly from its declarations."""
    figure, axis = plt.subplots(figsize=(12.0, 5.6))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    source_colour = "#FFF2CC"
    raw_colour = "#E8F1FA"
    centred_colour = "#E5F5EA"

    _diagram_box(axis, 0.02, 0.73, 0.13, 0.14, "r_t\nstandardised FIFA\nranking points", facecolor=source_colour, fontsize=7.4)
    _diagram_box(axis, 0.18, 0.73, 0.12, 0.14, "beta_A ~\nNormal(0, 1)", facecolor=source_colour, fontsize=7.8)
    _diagram_box(axis, 0.33, 0.73, 0.12, 0.14, "z_A,t ~\nNormal(0, 1)", facecolor=source_colour, fontsize=7.8)
    _diagram_box(axis, 0.18, 0.53, 0.12, 0.14, "sigma_A ~\nhalf-Cauchy(0, 2.5)", facecolor=source_colour, fontsize=7.1)
    _diagram_box(axis, 0.51, 0.68, 0.18, 0.19, "attack raw\nA*_t = beta_A r_t + sigma_A z_A,t", facecolor=raw_colour, fontsize=7.4)
    _diagram_box(axis, 0.75, 0.68, 0.16, 0.19, "centred attack\nA_t = A*_t - mean(A*)\nsum A_t = 0", facecolor=centred_colour, fontsize=7.0)

    _diagram_box(axis, 0.02, 0.22, 0.13, 0.14, "r_t\nstandardised FIFA\nranking points", facecolor=source_colour, fontsize=7.4)
    _diagram_box(axis, 0.18, 0.22, 0.12, 0.14, "alpha_D ~\nNormal(0, 1)", facecolor=source_colour, fontsize=7.8)
    _diagram_box(axis, 0.33, 0.22, 0.12, 0.14, "z_D,t ~\nNormal(0, 1)", facecolor=source_colour, fontsize=7.8)
    _diagram_box(axis, 0.18, 0.02, 0.12, 0.14, "sigma_D ~\nhalf-Cauchy(0, 2.5)", facecolor=source_colour, fontsize=7.1)
    _diagram_box(axis, 0.51, 0.17, 0.18, 0.19, "defence raw\nD*_t = alpha_D r_t + sigma_D z_D,t", facecolor=raw_colour, fontsize=7.4)
    _diagram_box(axis, 0.75, 0.17, 0.16, 0.19, "centred defence\nD_t = D*_t - mean(D*)\nsum D_t = 0", facecolor=centred_colour, fontsize=7.0)

    _diagram_box(axis, 0.51, 0.43, 0.18, 0.12, "eta ~ Normal(0, 1)\ncommon scoring level", facecolor="#FCE4D6", fontsize=7.5)
    _diagram_box(axis, 0.75, 0.43, 0.16, 0.12, "Poisson log rates\nfor both teams", facecolor="#EDE7F6", fontsize=7.5)

    for source in ((0.15, 0.80), (0.30, 0.80), (0.45, 0.80), (0.30, 0.60)):
        _diagram_arrow(axis, source, (0.51, 0.775))
    _diagram_arrow(axis, (0.69, 0.775), (0.75, 0.775))
    for source in ((0.15, 0.29), (0.30, 0.29), (0.45, 0.29), (0.30, 0.09)):
        _diagram_arrow(axis, source, (0.51, 0.265))
    _diagram_arrow(axis, (0.69, 0.265), (0.75, 0.265))
    _diagram_arrow(axis, (0.83, 0.68), (0.83, 0.55))
    _diagram_arrow(axis, (0.83, 0.36), (0.83, 0.43))
    _diagram_arrow(axis, (0.69, 0.49), (0.75, 0.49))
    axis.set_title("Ranking-informed hierarchical attack-defence model", pad=13, fontweight="bold")
    figure.text(
        0.5,
        0.015,
        "Raw effects use distinct attack and defence scales; centring enforces identifiability before the weighted Poisson likelihood.",
        ha="center",
        fontsize=8,
        color="#425466",
    )
    return _save_figure(figure, output, "hierarchical_prior")


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


def _plot_bracket_structure(project_root: Path, output: Path) -> list[Path]:
    """Render the tournament topology from the repository constants."""
    ensure_project_imports(project_root)
    from src.constants import (
        FIFA_THIRD_PLACE_TABLE,
        GROUPS,
        GROUP_STAGE_MATCHES,
        QUARTERFINAL_PAIRS,
        ROUND_OF_16_PAIRS,
        ROUND_OF_32_FIXED,
        SEMIFINAL_PAIRS,
    )

    group_count = len(GROUPS)
    teams_per_group = len(next(iter(GROUPS.values())))
    possible_third_sets = math.comb(group_count, 8)
    figure, axis = plt.subplots(figsize=(13.4, 5.2))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")

    _diagram_box(axis, 0.02, 0.65, 0.16, 0.19, f"{group_count} groups of {teams_per_group}\n{group_count * teams_per_group} teams\n{GROUP_STAGE_MATCHES} group matches", facecolor="#E8F1FA")
    _diagram_box(axis, 0.24, 0.70, 0.16, 0.15, "Group ranking\nfirst, second and\nthird place", facecolor="#DDEBF7")
    _diagram_box(axis, 0.46, 0.76, 0.16, 0.12, "Top two in each group\n24 teams", facecolor="#E5F5EA")
    _diagram_box(axis, 0.46, 0.49, 0.16, 0.16, "Third-placed teams\nrank best eight\nfrom 12", facecolor="#FFF2CC")
    _diagram_box(axis, 0.68, 0.49, 0.16, 0.16, f"Third-place mapping\n{possible_third_sets} qualifying group sets\n{len(FIFA_THIRD_PLACE_TABLE)} lookup rows", facecolor="#FCE4D6", fontsize=7.3)
    _diagram_box(axis, 0.68, 0.76, 0.16, 0.12, f"Round of 32\n{len(ROUND_OF_32_FIXED)} ties", facecolor="#EDE7F6")
    _diagram_box(axis, 0.87, 0.76, 0.10, 0.12, f"Round of 16\n{len(ROUND_OF_16_PAIRS)} ties", facecolor="#EDE7F6", fontsize=7.3)
    _diagram_box(axis, 0.87, 0.57, 0.10, 0.12, f"Quarter-finals\n{len(QUARTERFINAL_PAIRS)} ties", facecolor="#EDE7F6", fontsize=7.3)
    _diagram_box(axis, 0.87, 0.38, 0.10, 0.12, f"Semi-finals\n{len(SEMIFINAL_PAIRS)} ties", facecolor="#EDE7F6", fontsize=7.3)
    _diagram_box(axis, 0.87, 0.19, 0.10, 0.12, "Final\n1 tie", facecolor="#EDE7F6", fontsize=7.3)
    _diagram_box(axis, 0.68, 0.19, 0.16, 0.12, "Champion\nstage probability", facecolor="#D9EAD3")

    _diagram_arrow(axis, (0.18, 0.745), (0.24, 0.775))
    _diagram_arrow(axis, (0.40, 0.79), (0.46, 0.82))
    _diagram_arrow(axis, (0.40, 0.75), (0.46, 0.57))
    _diagram_arrow(axis, (0.62, 0.57), (0.68, 0.57))
    _diagram_arrow(axis, (0.62, 0.82), (0.68, 0.82))
    _diagram_arrow(axis, (0.84, 0.57), (0.76, 0.76))
    _diagram_arrow(axis, (0.84, 0.82), (0.87, 0.82))
    _diagram_arrow(axis, (0.92, 0.76), (0.92, 0.69))
    _diagram_arrow(axis, (0.92, 0.57), (0.92, 0.50))
    _diagram_arrow(axis, (0.92, 0.38), (0.92, 0.31))
    _diagram_arrow(axis, (0.87, 0.25), (0.84, 0.25))
    axis.set_title("Structure of the implemented 48-team tournament simulator", pad=13, fontweight="bold")
    figure.text(
        0.5,
        0.02,
        "After the group stage, the simulator applies the coded third-place lookup and a fixed knockout tree; tied simulated knockout matches advance by a fair coin flip.",
        ha="center",
        fontsize=8,
        color="#425466",
    )
    return _save_figure(figure, output, "bracket_structure")


def _plot_manuscript_method_figures(project_root: Path, output: Path) -> list[str]:
    """Generate every non-empirical manuscript figure from repository code."""
    _plot_predictive_pipeline(output)
    _plot_hierarchical_prior(output)
    _plot_time_decay(output)
    _plot_bracket_structure(project_root, output)
    return ["predictive_pipeline", "hierarchical_prior", "time_decay", "bracket_structure"]


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
    #figure.text(
    #    0.5,
    #    0.055,
    #    "Input: data/results.csv. Final series applies >20 appearances in the full input, then >=20 appearances within the 2026 cycle; cutoff: 15 Jul 2026.",
    #    ha="center",
    #    fontsize=7.4,
    #    color="#425466",
    #)
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
