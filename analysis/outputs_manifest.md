# Figure and Table Manifest

All paths are relative to `analysis/`. The manuscript labels identify the intended Results or Appendix placement.

## Figures

| Artifact | Files | LaTeX label | Status / use |
|---|---|---|---|
| `predictive_pipeline` | `figures/predictive_pipeline.png` | `\label{fig:pipeline}` | Methods: code-derived flow from inputs through posterior simulation and public outputs. |
| `hierarchical_prior` | `figures/hierarchical_prior.png` | `\label{fig:hierarchical-prior}` | Methods: direct rendering of the selected ranking-informed Stan hierarchy. |
| `time_decay` | `figures/time_decay.png` | `\label{fig:time-decay}` | Methods: exact exponential temporal factor in the implemented match weight. |
| `bracket_structure` | `figures/bracket_structure.png` | `\label{fig:bracket-structure}` | Methods: 48-team simulator topology derived from repository constants. |
| `model_selection_brier` | `figures/model_selection_brier.png` | `\label{fig:model-selection-brier}` | Main Results: archived 2018/2022 candidate-model comparison. |
| `matches_per_year` | `figures/matches_per_year.png` | `\label{fig:matches-per-year}` | Data: complete cycle records versus the final two-stage team-filtered training set. |
| `tournament_mix` | `figures/tournament_mix.png` | `\label{fig:eda-tournament-mix}` | Proposed label; add it when populating the Results section. |
| `matches_per_team` | `figures/matches_per_team.png` | `\label{fig:eda-matches-per-team}` | Proposed label; add it when populating the Results section. |
| `goals_poisson_fit` | `figures/goals_poisson_fit.png` | `\label{fig:eda-goal-distribution}` | Proposed label; add it when populating the Results section. |
| `draw_rates_by_year` | `figures/draw_rates_by_year.png` | `\label{fig:eda-draw-rates}` | Proposed label; add it when populating the Results section. |
| `draw_rates_by_tournament` | `figures/draw_rates_by_tournament.png` | `\label{fig:eda-draw-rates-tournament}` | Proposed label; add it when populating the Results section. |
| `home_advantage` | `figures/home_advantage.png` | `\label{fig:eda-home-advantage}` | Proposed label; add it when populating the Results section. |
| `training_weights` | `figures/training_weights.png` | `\label{fig:eda-training-weights}` | Exploratory diagnostic distinct from the analytic time-decay figure. |
| `team_strengths_and_ranking_prior` | `figures/team_strengths_and_ranking_prior.png` | `\label{fig:eda-team-strengths-prior}` | Proposed label; add it when populating the Results section. |
| `parameter_trajectories` | `figures/parameter_trajectories.png` | `\label{fig:appendix-parameter-trajectories}` | Appendix: uses repeated cutoff fits when available, otherwise saved World Cup snapshots. |
| `last_world_cup_attack_defense` | `figures/last_world_cup_attack_defense.png` | `\label{fig:appendix-last-world-cup-attack-defense}` | Appendix: Portuguese-labelled posterior strengths after the final available snapshot. |
| `before_world_cup_attack_defense` | `figures/before_world_cup_attack_defense.png` | `\label{fig:appendix-before-world-cup-attack-defense}` | Appendix: Portuguese-labelled posterior strengths before the World Cup. |
| `all_teams_attack_defense` | `figures/all_teams_attack_defense.png` | `\label{fig:appendix-all-teams-attack-defense}` | Appendix: all fitted teams with English labels only for World Cup-qualified teams. |
| `stage_probability_heatmap` | `figures/stage_probability_heatmap.png` | `\label{fig:stage-probability-heatmap}` | Main Results: pre-tournament stage probabilities for the 12 leading teams. |
| `title_probability_trajectories` | `figures/title_probability_trajectories.png` | `\label{fig:title-probability-trajectories}` | Main Results: fixed set of eight World Cup teams with distinct colours. |
| `title_probability_concentration` | `figures/title_probability_concentration.png` | `\label{fig:eda-title-concentration}` | Proposed label; add it when populating the Results section. |
| `draw_probability_vs_goal_difference` | `figures/draw_probability_vs_goal_difference.png` | `\label{fig:draw-probability}` | Existing conceptual placeholder in main.tex; replace with empirical figure if desired. |
| `scoreline_heatmap_examples` | `figures/scoreline_heatmap_examples.png` | `\label{fig:scoreline-heatmap-examples}` | Main Results: France--Morocco pre-quarter-final and Spain--Argentina pre-final examples. |
| `calibration_reliability` | `figures/calibration_reliability.png` | `\label{fig:eda-calibration-reliability}` | Proposed label; add it when populating the Results section. |
| `score_prediction_diagnostics` | `figures/score_prediction_diagnostics.png` | `\label{fig:score-prediction-diagnostics}` | Main Results: as-of modal score forecasts versus realised totals and goal differences. |
| `scoreline_frequency_check` | `figures/scoreline_frequency_check.png` | `\label{fig:eda-scoreline-frequency-check}` | Proposed label; add it when populating the Results section. |

## Tables

| Artifact | File | LaTeX label | Status / use |
|---|---|---|---|
| `model_selection_brier` | `tables/model_selection_brier.tex` | `\label{tab:appendix-model-selection-brier}` | Appendix: numeric source for candidate-model comparison. |
| `data_quality` | `tables/data_quality.tex` | `\label{tab:eda-data-quality}` | Proposed Results/Data label. |
| `training_window_summary` | `tables/training_window_summary.tex` | `\label{tab:eda-training-window}` | Proposed Results/Data label. |
| `goal_distribution` | `tables/goal_distribution.tex` | `\label{tab:eda-goal-distribution}` | Proposed Results/Data label. |
| `home_advantage` | `tables/home_advantage.tex` | `\label{tab:eda-home-advantage}` | Proposed Results/Data label. |
| `matches_per_team` | `tables/matches_per_team.tex` | `\label{tab:eda-matches-per-team}` | Proposed Results/Data label. |
| `draw_rates_by_tournament` | `tables/draw_rates_by_tournament.tex` | `\label{tab:eda-draw-rates-tournament}` | Proposed Results/Data label. |
| `world_cup_team_parameter_summaries` | `tables/world_cup_team_parameter_summaries.tex` | `\label{tab:eda-team-parameters}` | Proposed Results label. |
| `stage_probability_heatmap_data` | `tables/stage_probability_heatmap_data.tex` | `\label{tab:eda-stage-probabilities}` | Proposed Results label. |
| `pre_tournament_stage_probabilities` | `tables/pre_tournament_stage_probabilities.tex` | `\label{tab:appendix-pre-tournament-stage-probabilities}` | Appendix: complete pre-tournament stage-probability matrix. |
| `title_probability_concentration` | `tables/title_probability_concentration.tex` | `\label{tab:eda-title-concentration}` | Proposed Results label. |
| `matchup_probability_consistency` | `tables/matchup_probability_consistency.tex` | `\label{tab:eda-matchup-consistency}` | Proposed Results/diagnostics label. |
| `modal_score_prediction_metrics` | `tables/modal_score_prediction_metrics.tex` | `\label{tab:modal-score-prediction-metrics}` | Main Results: modal 0--4 scoreline point-forecast diagnostics. |
