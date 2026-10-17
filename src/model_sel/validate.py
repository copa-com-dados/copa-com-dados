import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from cmdstanpy import CmdStanModel

from src.data import load_ranking_priors, prepare_cycle_data


def train_and_save(
    cycle_name, model_name, stan_file, exe_file, df, teams, team_map, ranking_priors
):
    """
    Run one Stan model for one cycle and save its posterior draws.

    Prepared data is passed in so parallel jobs do not repeat preprocessing.
    """
    print(f"\n[{cycle_name}] Treinando: {model_name} ...")

    # Keep this schema aligned with the data block in every Stan model.
    stan_data = {
        "N": len(df),
        "T": len(teams),
        "team_i": df["home_team"].map(team_map).values,
        "team_j": df["away_team"].map(team_map).values,
        "y_i": df["home_score"].values.astype(int),
        "y_j": df["away_score"].values.astype(int),
        "game_weight": df["game_weight"].values,
        "prior_strength": ranking_priors,
        "is_home": (~df["neutral"].values).astype(int),
    }

    stan_model = CmdStanModel(stan_file=stan_file, exe_file=exe_file)

    # Recreate the model from source so CmdStan can rebuild stale binaries.
    stan_model = CmdStanModel(stan_file=stan_file)
    fit = stan_model.sample(data=stan_data, chains=4, iter_sampling=5000, seed=123)

    post_draws = fit.stan_variables()
    output_path = f"data/outputs/models/draws_{cycle_name}_{model_name}.npz"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez_compressed(output_path, teams=np.array(teams), **post_draws)
    print(f"Salvo em: {output_path}")


if __name__ == "__main__":
    # Friendly model names map directly to Stan files and output filenames.
    stan_model_files = {
        "n_poisson_noranking": "stan_models/n_poisson_noranking.stan",
        "st_dc_ranking": "stan_models/st_dc_ranking.stan",
        "st_dc_noranking": "stan_models/st_dc_noranking.stan",
        "n_dc_ranking": "stan_models/n_dc_ranking.stan",
        "n_dc_noranking": "stan_models/n_dc_noranking.stan",
        "st_poisson_ranking": "stan_models/st_poisson_ranking.stan",
        "st_poisson_noranking": "stan_models/st_poisson_noranking.stan",
        "n_poisson_ranking": "stan_models/n_poisson_ranking.stan",
    }

    print("=" * 50)
    print("PRECOMPILANDO MODELOS STAN")
    print("=" * 50)

    compiled_models = {}

    for model_key, stan_file in stan_model_files.items():
        print(f"Compilando {model_key} ...")

        model = CmdStanModel(stan_file=stan_file)

        compiled_models[model_key] = {
            "stan_file": stan_file,
            "exe_file": model.exe_file,
        }

    print("\nTodos os modelos compilados!")

    cycles = []

    print("=" * 50)
    print("PREPARANDO DADOS DO CICLO 2018")
    print("=" * 50)

    df_18, teams_18, team_map_18 = prepare_cycle_data(
        "data/raw/results.csv", "2014-06-11", "2018-06-13", apply_decay=True
    )

    ranking_priors_18 = load_ranking_priors("data/raw/fifa_ranking_2014.csv", teams_18)

    cycles.append(("2018", df_18, teams_18, team_map_18, ranking_priors_18))

    print("=" * 50)
    print("PREPARANDO DADOS DO CICLO 2022")
    print("=" * 50)

    df_22, teams_22, team_map_22 = prepare_cycle_data(
        "data/raw/results.csv", "2018-06-13", "2022-11-20", apply_decay=True
    )

    ranking_priors_22 = load_ranking_priors("data/raw/fifa_ranking_2018.csv", teams_22)

    cycles.append(("2022", df_22, teams_22, team_map_22, ranking_priors_22))

    # One parallel job per cycle/model pair.
    jobs = []

    for cycle_name, df, teams, team_map, ranking_priors in cycles:
        for model_key in stan_model_files:
            jobs.append(
                (
                    cycle_name,
                    model_key,
                    compiled_models[model_key]["stan_file"],
                    compiled_models[model_key]["exe_file"],
                    df,
                    teams,
                    team_map,
                    ranking_priors,
                )
            )

    print("\n" + "=" * 50)
    print("INICIANDO TREINAMENTO PARALELO")
    print("=" * 50)

    # Keep worker count conservative because Stan chains are CPU-heavy.
    MAX_WORKERS = 3

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(train_and_save, *job) for job in jobs]

        for future in as_completed(futures):
            future.result()

    print("\nTodos os modelos foram treinados e salvos!")
