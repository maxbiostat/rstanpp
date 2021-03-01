/**
 * Inverse link function on mean scale
 *
 * @param eta  linear predictor vector (X * beta)
 * @param mu_link an integer indicating the link function to the mean
 * @return A vector, i.e. inverse-link(eta)
 */
vector mu_linkinv(vector eta, int mu_link) {
  if      (mu_link == 1) return eta;                      // identity link
  else if (mu_link == 2) return exp(eta);                 // log link
  else if (mu_link == 3) return inv_logit(eta);           // logit link
  else if (mu_link == 4) return inv(eta);                 // inverse link
  else if (mu_link == 5) return Phi(eta);                 // probit link
  else if (mu_link == 6) return atan(eta) / pi() + 0.5;   // cauchit link
  else if (mu_link == 7) return inv_cloglog(eta);         // complementary log-log link
  else if (mu_link == 8) return square(eta);              // sqrt link
  else if (mu_link == 9) return inv(sqrt(eta));           // 1/mu^2 link
  else reject("Invalid link");
  return eta;                                             // never reached
}

/**
 * Link function on theta scale
 *
 * @param eta     linear predictor vector (X * beta)
 * @param mu_link an integer indicating the link function to the mean
 * @param dist    an integer indicating the distribution of the dependent variable
 * @return A vector, theta(X * beta); composition of canonical link and mu_link
 */
vector theta_link(vector eta, int mu_link, int dist) {
  if (dist == 1) return mu_linkinv(eta, mu_link);             // normal
  else if (dist == 2) return logit(mu_linkinv(eta, mu_link)); // bernoulli
  else if (dist == 3) return log(mu_linkinv(eta, mu_link));   // poisson
  else if (dist == 4) return inv(mu_linkinv(eta, mu_link));   // gamma
  else reject("Invalid density");
  return eta;                                           // never reached
}

real sum_bfun(vector theta, int dist) {
  if (dist == 1) return 0.5 * sum(square(theta));        // normal
  else if (dist == 2) return sum(log(1.0 + exp(theta))); // bernoulli
  else if (dist == 3) return sum(exp(theta));            // poisson
  else if (dist == 4) return -1.0 * sum(log(theta));     // gamma
  else reject("Invalid density");
  return sum(theta);                                // never reached
}

/** Conjugate log posterior density
  @param beta        regression coefficient for GLM
  @param dispersion  dispersion parameter for GLM
  @param m_post      posterior location parameter
  @param lambda_post posterior scale parameter
  @param X           design matrix
  @param dist        indicator for distribution (1 = normal; 2 = bernoulli; 3 = poisson; 4 = gamma)
*/
real glm_logpost(vector beta, real dispersion, vector m_post, real lambda_post, matrix X, int dist, int mu_link, int N) {
  vector[N] theta = X * beta;
  theta = theta_link(theta, mu_link, dist);

  return lambda_post / dispersion * ( dot_product(m_post, theta) - sum_bfun(theta, dist) );
}


/**  sum of c(y_i, phi) function in exponential family distribution
  @param sum_fy      sum of a function of y (y^2 for normal, log(y) for gamma)
  @param dispersion  dispersion parameter
*/
real sum_cfun(real sum_fy, real dispersion, int N, int dist) {
  real disp_inv = inv(dispersion);
  if (dist == 1)      return -0.5 * ( sum_fy * disp_inv + N * log(dispersion) );
  else if (dist == 4) return (disp_inv - 1) * sum_fy - N * ( disp_inv * log(dispersion) + lgamma(disp_inv) );
  else reject("Density must be gaussian or Gamma");
  return 0; // never reached
}


