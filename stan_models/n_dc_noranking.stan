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
  vector[T] attack_raw;
  vector[T] defense_raw;
  real eta;
  real<lower=-0.5, upper=0.5> rho;
  real<lower=0.01> sigma_att;
  real<lower=0.01> sigma_def;
  real beta_prior;
  real beta_home;
}

transformed parameters {
  vector[T] attack = attack_raw - mean(attack_raw);
  vector[T] defense = defense_raw - mean(defense_raw);
}

model {
  attack_raw ~ normal(0, sigma_att);
  defense_raw ~ normal(0, sigma_def);
  rho ~ normal(0, 0.1);
  eta ~ normal(0, 1);
  beta_prior ~ normal(0, 1);
  sigma_att ~ cauchy(0, 2.5);
  sigma_def ~ cauchy(0, 2.5);
  beta_home ~ normal(0, 0.5);

  for (n in 1:N) {
    real mu = exp(attack[team_i[n]] - defense[team_j[n]] + eta + beta_home * is_home[n]);
    real lambda = exp(attack[team_j[n]] - defense[team_i[n]] + eta);
    real log_prob = poisson_lpmf(y_i[n] | mu)
      + poisson_lpmf(y_j[n] | lambda);

    if (y_i[n] == 0 && y_j[n] == 0) {
      log_prob += log(fmax(0.001, 1 - mu * lambda * rho));
    } else if (y_i[n] == 0 && y_j[n] == 1) {
      log_prob += log(fmax(0.001, 1 + mu * rho));
    } else if (y_i[n] == 1 && y_j[n] == 0) {
      log_prob += log(fmax(0.001, 1 + lambda * rho));
    } else if (y_i[n] == 1 && y_j[n] == 1) {
      log_prob += log(fmax(0.001, 1 - rho));
    }

    target += game_weight[n] * log_prob;
  }
}
