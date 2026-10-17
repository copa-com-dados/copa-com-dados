data {
  int N;
  int T;
  array[N] int team_i;
  array[N] int team_j;
  array[N] int y_i;
  array[N] int y_j;
  vector[N] game_weight;
  vector[T] prior_strength;
  array[N] int is_home;
}

parameters {
  vector[T] attack_std;
  vector[T] defense_std;

  real eta;
  real<lower=0.01> sigma_att;
  real<lower=0.01> sigma_def;
  real beta_prior;
  real alpha_prior;
  real beta_home;
}

transformed parameters {
  vector[T] attack_raw = (prior_strength * beta_prior) + attack_std * sigma_att;
  vector[T] defense_raw = (prior_strength * alpha_prior) + defense_std * sigma_def;

  vector[T] attack = attack_raw - mean(attack_raw);
  vector[T] defense = defense_raw - mean(defense_raw);
}

model {
  attack_std ~ student_t(3, 0, 1);
  defense_std ~ student_t(3, 0, 1);

  eta ~ normal(0, 1);
  beta_prior ~ normal(0, 1);
  alpha_prior ~ normal(0, 1);
  sigma_att ~ cauchy(0, 2.5);
  sigma_def ~ cauchy(0, 2.5);
  beta_home ~ normal(0, 0.5);

  for (n in 1:N) {
    target += game_weight[n] * (
      poisson_log_lpmf(
        y_i[n] | attack[team_i[n]] - defense[team_j[n]] + eta + beta_home * is_home[n]
      )
      + poisson_log_lpmf(
        y_j[n] | attack[team_j[n]] - defense[team_i[n]] + eta
      )
    );
  }
}
