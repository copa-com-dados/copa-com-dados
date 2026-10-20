"""Input audit, cutoff-aware fitting and snapshot generation.

This module is the execution layer for the first notebook. It delegates model
fitting, posterior sampling, score simulation and tournament simulation to the
project's existing ``src`` code. Its public functions validate inputs, create
one reproducible forecast snapshot per cutoff, and persist tidy snapshot
products for the separate evaluation and reporting modules.
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
