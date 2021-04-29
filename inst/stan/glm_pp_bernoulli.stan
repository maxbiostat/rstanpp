
functions {
#include /functions/link.stan
#include /functions/glm_pp.stan
}

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0>          nobs;          // number of observations
  int<lower=0>          p;             // number of predictors
  int<lower=0,upper=1>  y0[nobs];      // historical data
  matrix[nobs,p]        X;             // design matrix
  real<lower=0,upper=1> a0;            // power prior parameter
  vector[p]             beta0;         // initial prior mean of beta
  matrix[p,p]           Sigma0;        // initial prior covariance matrix of beta
  int<lower=1,upper=9>  link;          // index of link function
  int<lower=0,upper=1>  incl_offset;   // whether an offset is included
  vector[nobs]          offset;        // offset (defaults to vector of 0s in R)
}

// Only one parameter: the regression coefficient
parameters {
  vector[p] beta;
}

// Assume beta is a priori MVN; obtain posterior
// based on power prior
model {
  vector[( (link != 3) || (incl_offset == 1) ) ? nobs :  0] eta;
  vector[(link != 3) ? nobs :  0] mu;
  
  // initial prior is MVN for beta
  beta ~ multi_normal(beta0, Sigma0);
  
  // add log power prior--using the best stan function available per scenario
  if ( a0 > 0 ) {
    if ( link == 3 && incl_offset == 0 )
      target += a0 * bernoulli_logit_glm_lpmf(y0 | X, 0, beta);
    else {
      eta = X * beta;
      if ( incl_offset == 1 )
        eta += offset;
      if ( link == 3 )
        target += a0 * bernoulli_logit_lpmf(y0 | eta);
      else {
        mu = linkinv(eta, link);
        target += a0 * bernoulli_lpmf(y0 | mu);
      }
    }
  }
}
