
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
  real<lower=0>         disp_shape; // shape parameter for gamma prior on inverse dispersion
  real<lower=0>         disp_scale;  // rate parameter on gamma prior for inverse dispersion
}

transformed data {
  vector[N0] y0vec = to_vector(y0);
  vector[N]  yvec  = to_vector(y);
}

// Two parameters: regression coefficient and power prior param
parameters {
  vector[p] beta;
  real<lower=0> dispersion;
  real<lower=0,upper=1> a0;
}

model {
  vector[( (link != 1) || (incl_offset == 1) ) ? N :  0] eta;
  vector[( (link != 1) || (incl_offset == 1) ) ? N0 :  0] eta0;
  vector[(link != 1) ? N :  0] mu;
  vector[(link != 1) ? N0 :  0] mu0;
  real sigma = sqrt(dispersion);
  
  
  // add to target prior distributions and log nc
  target     += -pp_lognc(a0, a0vec, lognca0);
  dispersion ~  inv_gamma(disp_shape, disp_scale);
  beta       ~  multi_normal(beta0, Sigma0);
  if ( a0_shape1 != 1 || a0_shape2 != 1 )
    a0 ~ beta(a0_shape1, a0_shape2);
  
  // if no offset and identity link, use Stan GLM function
  if ( link == 1 && incl_offset == 0 ) {
    target += a0 * normal_id_glm_lpdf(y0vec | X0, 0, beta, sigma);
    target += normal_id_glm_lpdf(yvec | X, 0, beta, sigma);
  }
  // otherwise, compute linear predictor and add offset if necessary
  else {
    eta  = X  * beta;
    eta0 = X0 * beta0;
    if ( incl_offset == 1 ) {
      eta  += offset;
      eta0 += offset0;
    }
    if ( link == 1 ) {
      target += a0 * normal_lpdf(y0 | eta0, sigma);
      target += normal_lpdf(y | eta, sigma);
    }
    else {
      mu  = linkinv(eta, link);
      mu0 = linkinv(eta0, link);
      target += a0 * normal_lpdf(y0 | mu0, sigma);
      target += normal_lpdf(y | mu, sigma);
    }
  }
}
