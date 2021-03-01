

// This Stan program implements sampling of a GLM using the
// conjugate prior of Chen & Ibrahim when the dispersion
// parameter is gamma distributed

functions {
#include /functions/functions.stan
}


// posterior quantities computed in R--just need Stan for sampling
data {
  int<lower = 0> N;                      // number of observations
  int<lower = 0> K;                      // number of predictors
  matrix[N, K] X;                        // design matrix
  vector[N] m_post;                      // posterior location parameter
  real<lower = 0> lambda_post;           // posterior precision parameter
  int<lower = 1, upper = 9> mu_link;     // indicator for link function to use
  int<lower = 1, upper = 4> dist;        // indicator for which distribtuion we're using
  real<lower = 0> dispersion_shape;      // shape parameter for gamma prior on 1 / dispersion (shape param for inv gamma on dispersion)
  real<lower = 0> dispersion_rate;       // RATE parameter for gamma prior on 1 / dispersion  (SCALE param for inv gamma on dispersion)
  real sum_fy;                           // sum(f(y)) for dispersion; f(y) = y^2 for gaussian and f(y) = log(y) for Gamma
}

// The parameters accepted by the model.
parameters {
  vector[K] beta;             // coefficients for predictors
  real<lower=0> dispersion;   // scalar dispersion parameter
}

// The model is the conjugate posterior.
model {
  dispersion ~ inv_gamma(dispersion_shape, dispersion_rate);                         // prior for dispersion
  target += glm_logpost(beta, dispersion, m_post, lambda_post, X, dist, mu_link, N); // conjugate posterior portion
  target += sum_cfun(sum_fy, dispersion, N, dist);                                   // sum of cfunction in likelhood.
}




