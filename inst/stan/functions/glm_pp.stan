
/**
  * Bernoulli GLM likelihood w/ power prior
  *
  * @param y    integer array of responses
  * @param a0   scalar between 0 and 1 giving pp parameter
  * @param eta  linear predictor
  * @param link index giving the link function
  *
  * @return unnormalized log prior
**/
real bernoulli_glm_pp_lp(int[] y, real a0, vector eta, int link) {
  int nobs         = num_elements(y);
  if ( link == 3 ) { // logit
    target += a0 * bernoulli_logit_lpmf(y | eta );
  }
  else { 
    vector[nobs] mu = linkinv(eta, link);
    target += a0 * bernoulli_lpmf(y | mu);
  }
  return target();
}


/**
  * Binomial GLM likelihood w/ power prior
  *
  * @param y       integer array of responses
  * @param ntrials number of trials
  * @param a0      scalar between 0 and 1 giving pp parameter
  * @param eta     linear predictor
  * @param link    index giving the link function
  *
  * @return unnormalized log prior
**/
real binomial_glm_pp_lp(int[] y, int[] ntrials, vector eta, real a0, int link) {
  int nobs         = num_elements(y);
  if ( link == 3 ) {  // logit
    target += a0 * binomial_logit_lpmf(y | ntrials, eta);
  }
  else {
    vector[nobs] mu = linkinv(eta, link);
    target += a0 * binomial_lpmf(y | ntrials, mu);
  }
  return target();
}



/**
  * Poisson GLM likelihood w/ power prior
  *
  * @param y    integer array of responses
  * @param a0   scalar between 0 and 1 giving pp parameter
  * @param eta  linear predictor
  * @param link index giving the link function
  *
  * @return unnormalized log prior
**/
real poisson_glm_pp_lp(int[] y, real a0, vector eta, int link) {
  int nobs         = num_elements(y);
  if ( link == 2 ) {   // log link
    target += a0 * poisson_log_lpmf(y | eta);
  }
  else {
    vector[nobs] mu = linkinv(eta, link);
    target += a0 * poisson_lpmf(y | mu);
  }
  return target();
}




/**
  * Gaussian GLM likelihood w/ power prior
  *
  * @param y    vector of responses
  * @param a0   scalar between 0 and 1 giving pp parameter
  * @param eta  linear predictor
  * @param phi  dispersion parameter (sigma^2)
  * @param link index giving the link function
  *
  * @return unnormalized log prior
**/
real normal_glm_pp_lp(real[] y, real a0, vector eta, real phi, int link) {
  int nobs         = num_elements(y);
  if ( link == 1 ) {  // identity link
    target += a0 * normal_lpdf(y | eta, sqrt(phi));
  }
  else {
    vector[nobs] mu = linkinv(eta, link);
    target += a0 * normal_lpdf(y | mu, sqrt(phi));
  }
  return target();
}





/**
  * Gamma GLM likelihood w/ power prior
  *
  * @param y    vector of responses
  * @param a0   scalar between 0 and 1 giving pp parameter
  * @param eta  linear predictor
  * @param phi  dispersion parameter (1 / alpha)
  * @param X    design matrix (incl intercept)
  * @param link index giving the link function
  *
  * @return unnormalized log prior
**/
real gamma_glm_pp_lp(real[] y, real a0, vector eta, real phi, int link) {
  int nobs         = num_elements(y);
  vector[nobs] mu  = linkinv(eta, link);
  
  // shape parameter for gamma distribution
  real alpha = inv(phi);
  
  target += a0 * gamma_lpdf(y | alpha, alpha * inv(mu) );
  return target();
}




