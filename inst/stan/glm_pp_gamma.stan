
functions {
#include /functions/link.stan
#include /functions/glm_pp.stan
}


data {
  int<lower=0>          nobs;          // number of observations
  int<lower=0>          p;             // number of predictors
  real<lower=0>         y0[nobs];      // historical data
  matrix[nobs,p]        X;             // design matrix
  real<lower=0,upper=1> a0;            // power prior parameter
  vector[p]             beta0;         // initial prior mean of beta
  matrix[p,p]           Sigma0;        // initial prior covariance matrix of beta
  real<lower=0>         disp_shape;    // shape parameter for inverse-gamma prior on dispersion
  real<lower=0>         disp_scale;    // scale parameter for inverse-gamma prior on dispersion
  int<lower=1,upper=9>  link;          // index of link function
  int<lower=0,upper=1>  incl_offset;   // whether an offset is included
  vector[nobs]          offset;        // offset (defaults to vector of 0s in R)
}

// p+1 params: p-dim vector of regression coefficients and scalar inverse dispersion
parameters {
  vector[p]     beta;
  real<lower=0> dispersion;
}

// Assume beta is a priori MVN; obtain posterior
// based on power prior
model {
  vector[nobs] eta;
  vector[nobs] mu;
  real alpha = inv(dispersion);                    // shape param for gamma likelihood
  beta       ~ multi_normal(beta0, Sigma0);        // MVN initial prior on beta
  dispersion ~ inv_gamma(disp_shape, disp_scale);  // inverse-gamma initial prior on dispersion
  
  // add down-weighted likelihood if a0 > 0
  if ( a0 > 0 ) {
    // add offset to eta if incl_offset == 1
    eta = X * beta;
    if ( incl_offset == 1 )
      eta += offset;
    
    // get mean via inverse link and increase target by log likelihood
    mu      = linkinv(eta, link);
    target += a0 * gamma_lpdf(y0 | alpha, alpha * inv(mu));
  }
}
