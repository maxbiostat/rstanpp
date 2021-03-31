

// This Stan program implements sampling of a GLM using the
// conjugate prior of Chen & Ibrahim when the dispersion
// parameter is known and fixed.

functions {
#include /functions/functions.stan
#include /functions/priors.stan
}


// posterior quantities computed in R--just need Stan for sampling
data {
  int<lower = 0> N;                      // number of observations
  int<lower = 0> K;                      // number of predictors
  vector<lower = 0>[N] y;                // response vector
  matrix[N, K] X;                        // design matrix
  real<lower = 0> lambda;                // prior precision parameter for beta
  real<lower = 0> lambda0;               // prior precision parameter for m
  vector<lower = 0>[N] m0;               // prior location parameter for m
  int<lower = 1, upper = 9> mu_link;     // indicator for link function to use
  vector[K] start;                       // starting value for IRLS (typically MLE using prior mean of m as data)
  int<lower = 1> maxit;                  // max num iterations in IRLS (typically <= 100)
  real<lower = 0> tol;                   // threshold for convergence in IRLS (typically <= 1.0e-6)
}

transformed data {
  matrix[N, K] Q    = qr_thin_Q(X);      // Q of QR decomposiotion of X
  matrix[K, K] R    = qr_thin_R(X);      // R of QR decomposition of X
  matrix[K, K] Rinv = inverse(R);        // inverse of R
  real lambdapost   = 1.0 + lambda;      // posterior precision parameter
  vector[N] shape   = lambda0 * m0;      // prior shape1 param for m
}

// The parameters accepted by the model.
parameters {
  vector[K] beta;                     // coefficients for predictors
  vector<lower = 0>[N] m;             // location parameter for conjugate prior (prior response rates)
}



// The model is the conjugate posterior + log nc of prior + log prior of m
model {
  vector[N] mpost = (y + lambda * m) / lambdapost;

  target += binomial_conjugate_lpdf(beta | mpost, X, lambdapost, mu_link);                            // log posterior density excl. normalizing constant
  target += -1.0 * lognc_laplace(m, X, Q, R, Rinv, 2, mu_link, start, lambda, 1.0, 100, tol, N, K);   // estimated log nc of conjugate prior using laplace
  target += gamma_lpdf(m | shape, lambda0);                                                           // log prior density of m
}
