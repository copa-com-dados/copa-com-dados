# Output Manifest

All paths are relative to `analysis/`. This inventory covers the figures and LaTeX tables emitted by the current exploratory-reporting workflow; legacy PNGs that are no longer generated are not listed.

## Figures

| Artifact | Files | Description | Source |
|---|---|---|---|
| `time_decay` | `figures/time_decay.png` | Exponential time-decay component of the match weight, including its half-life. | The implemented temporal-weight formula with decay rate 0.001; no match-level data. |
| `model_selection_brier` | `figures/model_selection_brier.png` | Candidate-model median Brier scores and 95% intervals for the 2018 and 2022 World Cup holdouts. | `data/outputs/results/brier_score_2018.csv` and `data/outputs/results/brier_score_2022.csv`. |
| `matches_per_year` | `figures/matches_per_year.png` | Annual count of complete international matches before and after the 2026 training filters. | `data/results.csv`, restricted to the exploratory cutoff and compared with the filtered training data. |
| `tournament_mix` | `figures/tournament_mix.png` | Most frequent tournament categories in the training window. | Filtered 2026 training data derived from `data/results.csv`. |
| `matches_per_team` | `figures/matches_per_team.png` | Distribution of the number of training matches available for each national team. | Filtered 2026 training data derived from `data/results.csv`. |
| `goals_poisson_fit` | `figures/goals_poisson_fit.png` | Observed team-level goal distribution compared with a Poisson distribution fitted at the sample mean. | Home and away scores in the filtered 2026 training data from `data/results.csv`. |
| `draw_rates_by_year` | `figures/draw_rates_by_year.png` | Observed draw frequency by calendar year in the training window. | Home and away scores and dates in the filtered 2026 training data from `data/results.csv`. |
| `draw_rates_by_tournament` | `figures/draw_rates_by_tournament.png` | Draw frequencies for tournament categories with at least 20 training matches. | Scores and tournament labels in the filtered 2026 training data from `data/results.csv`. |
| `home_advantage` | `figures/home_advantage.png` | Mean home and away goals by neutral versus non-neutral venue status. | Scores and neutral-venue indicator in the filtered 2026 training data from `data/results.csv`. |
| `training_weights` | `figures/training_weights.png` | Distribution of the time-decay and combined tournament weights assigned to training matches. | Match dates, tournament categories and derived weights in the filtered 2026 training data from `data/results.csv`. |
| `team_strengths_and_ranking_prior` | `figures/team_strengths_and_ranking_prior.png` | Posterior attack and defence components and their relation to the FIFA-ranking prior. | Latest snapshot `team_parameter_summaries.csv`; `data/raw/fifa_ranking_2022.csv`; World Cup team allocation. |
| `parameter_trajectories` | `figures/parameter_trajectories.png` | Attack and defence posterior means for selected teams across fitted cutoffs. | Snapshot `team_parameter_summaries.csv` files; uses the optional parameter-trajectory archive when available. |
| `last_world_cup_attack_defense` | `figures/last_world_cup_attack_defense.png` | Posterior attack and defence estimates, with uncertainty intervals, for World Cup-qualified teams at the latest cutoff. | Latest snapshot `team_parameter_summaries.csv` and the World Cup team allocation. |
| `before_world_cup_attack_defense` | `figures/before_world_cup_attack_defense.png` | Posterior attack and defence estimates, with uncertainty intervals, for World Cup-qualified teams before the tournament. | `pre_tournament_post_fifa/team_parameter_summaries.csv` and the World Cup team allocation. |
| `all_teams_attack_defense` | `figures/all_teams_attack_defense.png` | Posterior attack and defence estimates for all fitted national teams, labelling qualified teams only. | Latest snapshot `team_parameter_summaries.csv` and the World Cup team allocation. |
| `stage_probability_heatmap` | `figures/stage_probability_heatmap.png` | Pre-tournament probabilities of reaching each stage for the 12 teams with the highest title probabilities. | `pre_tournament_post_fifa/stage_probabilities.csv`. |
| `title_probability_trajectories` | `figures/title_probability_trajectories.png` | Evolution of title probabilities across the cutoff-specific World Cup snapshots for eight selected teams. | `stage_probabilities.csv` from all available World Cup snapshots. |
| `title_probability_concentration` | `figures/title_probability_concentration.png` | Herfindahl--Hirschman index of the simulated title-probability distribution by snapshot. | Champion probabilities in `stage_probabilities.csv` from all available World Cup snapshots. |
| `draw_probability_vs_goal_difference` | `figures/draw_probability_vs_goal_difference.png` | Relationship between predicted draw probability, expected-goal difference and expected total goals for simulated matchups. | Latest snapshot `matchup_summary.csv`. |
| `scoreline_heatmap_examples` | `figures/scoreline_heatmap_examples.png` | Exact-scoreline probability grids for the France--Morocco quarter-final and Spain--Argentina final examples. | `after_round_of_16/scoreline_probabilities.csv` and `after_semifinals/scoreline_probabilities.csv`. |
| `calibration_reliability` | `figures/calibration_reliability.png` | Reliability diagrams comparing as-of predicted outcome probabilities with observed frequencies. | `artifacts/evaluation/calibration_bins.parquet`, derived from saved snapshots and observed World Cup results. |
| `score_prediction_diagnostics` | `figures/score_prediction_diagnostics.png` | As-of modal-score forecasts compared with realised total goals and goal differences. | `artifacts/evaluation/score_prediction_details.parquet`, derived from saved snapshots and observed World Cup results. |
| `scoreline_frequency_check` | `figures/scoreline_frequency_check.png` | Observed and predicted frequencies of the most common scoreline categories. | `artifacts/evaluation/scoreline_frequency_check.parquet`, derived from saved snapshots and observed World Cup results. |

## Tables

| Artifact | Files | Description | Source |
|---|---|---|---|
| `model_selection_brier` | `tables/model_selection_brier.tex` | Numerical candidate-model Brier-score comparison for the 2018 and 2022 World Cup holdouts. | `data/outputs/results/brier_score_2018.csv` and `data/outputs/results/brier_score_2022.csv`. |
| `missingness` | `tables/missingness.tex` | Missing-value counts and proportions by field in each audited input file. | Historical results, shootouts, goalscorers, World Cup, calendar and FIFA-ranking input files enumerated by the data inventory. |
| `data_quality` | `tables/data_quality.tex` | File availability, dimensions, duplicate-row counts and read-status audit. | Historical results, shootouts, goalscorers, World Cup, calendar and FIFA-ranking input files enumerated by the data inventory. |
| `training_window_summary` | `tables/training_window_summary.tex` | Size, period, team coverage, venue mix, draw rate and mean goals of the training window. | Filtered 2026 training data derived from `data/results.csv`. |
| `goal_distribution` | `tables/goal_distribution.tex` | Mean, variance, dispersion and zero-goal frequencies used to assess the Poisson goal distribution. | Home and away scores in the filtered 2026 training data from `data/results.csv`. |
| `home_advantage` | `tables/home_advantage.tex` | Goal and outcome summaries for neutral and non-neutral matches. | Scores and neutral-venue indicator in the filtered 2026 training data from `data/results.csv`. |
| `matches_per_team` | `tables/matches_per_team.tex` | Training-match count for every retained national team. | Filtered 2026 training data derived from `data/results.csv`. |
| `draw_rates_by_tournament` | `tables/draw_rates_by_tournament.tex` | Match counts and draw frequencies for the leading eligible tournament categories. | Scores and tournament labels in the filtered 2026 training data from `data/results.csv`. |
| `world_cup_team_parameter_summaries` | `tables/world_cup_team_parameter_summaries.tex` | Posterior attack and defence summaries and FIFA-prior values for World Cup-qualified teams. | Latest snapshot `team_parameter_summaries.csv`; `data/raw/fifa_ranking_2022.csv`; World Cup team allocation. |
| `stage_probability_heatmap_data` | `tables/stage_probability_heatmap_data.tex` | Stage-probability matrix underlying the compact heatmap of leading teams. | `pre_tournament_post_fifa/stage_probabilities.csv`. |
| `pre_tournament_stage_probabilities` | `tables/pre_tournament_stage_probabilities.tex` | Complete 48-team pre-tournament stage-probability matrix, expressed in percentages. | `pre_tournament_post_fifa/stage_probabilities.csv`. |
| `title_probability_concentration` | `tables/title_probability_concentration.tex` | Title-probability concentration index by forecast snapshot. | Champion probabilities in `stage_probabilities.csv` from all available World Cup snapshots. |
| `matchup_probability_consistency` | `tables/matchup_probability_consistency.tex` | Descriptive checks that outcome probabilities and score-grid plus residual mass sum to one. | Latest snapshot `matchup_summary.csv`. |
| `modal_score_prediction_metrics` | `tables/modal_score_prediction_metrics.tex` | Error and hit-rate measures for as-of and pre-tournament modal scoreline forecasts. | `artifacts/evaluation/score_prediction_metrics.parquet`, derived from saved snapshots and observed World Cup results. |
