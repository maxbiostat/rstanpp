
functions {
#include /functions/link.stan
#include /functions/glm_pp.stan
#include /functions/finda0closest.stan
}

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0>          N;             // number of observations in current data
  int<lower=0>          N0;            // number of observations in historical data
  int<lower=0>          K;             // number of values normalizing constant is evaluated at
  int<lower=0>          p;             // number of predictors
  real                  y[N];          // current response vector
  real                  y0[N0];        // historical response vector
  matrix[N,p]           X;             // design matrix for current data
  matrix[N0,p]          X0;            // design matrix for historical data
  vector[p]             beta0;         // initial prior mean of beta
  matrix[p,p]           Sigma0;        // initial prior covariance matrix of beta
  int<lower=1,upper=9>  link;          // index of link function
  int<lower=0,upper=1>  incl_offset;   // whether an offset is included
  vector[N]             offset;        // offset for current data (defaults to vector of 0s in R)
  vector[N0]            offset0;       // offset for historical data (defaults to vector of 0s in R)
  real<lower=0,upper=1> a0vec[K];      // array of pp params for which lognc was evaluated
  real                  lognca0[K];    // array of lognc evaluated at a0vec
  real<lower=0>         a0_shape1;     // shape 1 parameter for beta prior on a0
  real<lower=0>         a0_shape2;     // shape 2 parameter for beta prior on a0
  real<lower=0>         invdisp_shape; // shape parameter for gamma prior on inverse dispersion
  real<lower=0>         invdisp_rate;  // rate parameter on gamma prior for inverse dispersion
}

// Two parameters: regression coefficient and power prior param
parameters {
  vector[p] beta;
  real<lower=0> invdisp;
  real<lower=0,upper=1> a0;
}

model {
  // get phi = 1 / invdisp
  real phi = inv(invdisp);
  
  // obtain linear predictor (adding offset if applicable)
  vector[N0] eta0 = X0 * beta;
  vector[N]  eta  = X  * beta;
  if (incl_offset == 1) {
    eta0 = eta0 + offset0;
    eta  = eta  + offset;
  }
  
  // add on beta prior for a0 if shape parameters are not 1; otherwise U(0,1) assumed
  if ( a0_shape1 != 1 || a0_shape2 != 1 ) {
    a0 ~ beta(a0_shape1, a0_shape2);
  }
  target  += -pp_lognc(a0, a0vec, lognca0);
  invdisp ~  gamma(invdisp_shape, invdisp_rate);
  beta    ~  multi_normal(beta0, Sigma0);
  target  += normal_glm_pp_lp(y0, a0, eta0, phi, link);
  target  += normal_glm_pp_lp(y, 1.0, eta, phi, link);
}
