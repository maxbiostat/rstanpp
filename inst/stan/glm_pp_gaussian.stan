
functions {
#include /functions/link.stan
#include /functions/PC_prior.stan
#include /functions/glm_pp.stan
}


data {
  int<lower=0>          nobs;          // number of observations
  int<lower=0>          p;             // number of predictors
  real                  y0[nobs];      // historical data
  matrix[nobs,p]        X;             // design matrix
  real<lower=0,upper=1> a0;            // power prior parameter
  vector[p]             beta0;         // initial prior mean of beta
  matrix[p,p]           Sigma0;        // initial prior covariance matrix of beta
  real<lower=0>         disp_threshold;    // shape parameter for inverse-gamma prior on dispersion
  real<lower=0>         disp_probability;    // scale parameter for inverse-gamma prior on dispersion
  int<lower=1,upper=9>  link;          // index of link function
  int<lower=0,upper=1>  incl_offset;   // whether an offset is included
  vector[nobs]          offset;        // offset (defaults to vector of 0s in R)
}
transformed data {
  vector[nobs] y0vec = to_vector(y0);
}

// p+1 params: p-dim vector of regression coefficients and scalar inverse dispersion
parameters {
  vector[p] beta;
  real<lower=0> dispersion;
}

// Assume beta is a priori MVN; obtain posterior
// based on power prior
model {
  vector[( (link != 1) || (incl_offset == 1) ) ? nobs :  0] eta;
  vector[(link != 1) ? nobs :  0] mu;
  real sigma = sqrt(dispersion);
  
  // initial priors
  beta       ~ multi_normal(beta0, Sigma0);        // MVN initial prior on beta
  dispersion ~ PC(disp_threshold, disp_probability, 0.5);  // inverse-gamma initial prior on dispersion
  
  if ( a0 > 0 ) {
    if ( link == 1 && incl_offset == 0 )
      target += a0 * normal_id_glm_lpdf(y0vec | X, 0.0, beta, sigma);
    else {
      eta = X * beta;
      if ( incl_offset == 1 )
        eta += offset;
      if ( link == 1 )
        target += a0 * normal_lpdf(y0 | eta, sigma);
      else {
        mu      = linkinv(eta, link);
        target += a0 * normal_lpdf(y0 | mu, sigma);
      }
    }
  }
}
