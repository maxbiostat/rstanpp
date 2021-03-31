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
  return eta;                                             // never reached
}
vector mu_eta(vector eta, int mu_link, int N) {
  vector[N] temp;
  if      (mu_link == 1) return rep_vector(1.0, N);               // identity link
  else if (mu_link == 2) return exp(eta);                         // log link
  else if (mu_link == 3) {                                        // logit link
    temp = exp(eta);
    return temp ./ square(1.0 + temp);
  }
  else if (mu_link == 4) return -1.0 * inv(square(eta));                 // inverse link
  else if (mu_link == 5) return sqrt(2 * pi()) * exp(-0.5 * square(eta));  // probit link
  else if (mu_link == 6) return inv(1 + square(eta)) / pi();      // cauchit link
  else if (mu_link == 7) {                                        // complementary log-log link
    temp = exp(eta);
    return temp .* exp(-temp);
  }
  else if (mu_link == 8) return 2.0 * eta;                        // sqrt link
  else if (mu_link == 9){                                          // 1/mu^2 link
    for ( i in 1:N ) {
      temp[i] = -1.0 * inv(2.0 * pow(eta[i], 1.5));
    }
    return temp;
  }
  return eta;                                                     // never reached
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
  return eta;                                           // never reached
}

real sum_bfun(vector theta, int dist) {
  if (dist == 1) return 0.5 * sum(square(theta));        // normal
  else if (dist == 2) return sum(log(1.0 + exp(theta))); // bernoulli
  else if (dist == 3) return sum(exp(theta));            // poisson
  else if (dist == 4) return -1.0 * sum(log(theta));     // gamma
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
  return 0; // never reached
}


vector vfun(vector mu, int dist) {
  if (dist == 1) return rep_vector(1.0, num_elements(mu));  // normal
  else if (dist == 2) return mu .* (1 - mu);                // bernoulli
  else if (dist == 3) return mu;                            // poisson
  else if (dist == 4) return square(mu);                    // gamma
  return mu;                                               // never reached
}

real lognc_laplace(vector m, matrix X, matrix Q, matrix R, matrix Rinv, int dist, int mu_link, vector start, real lambda, real tau, int maxit, real tol, int n, int p) {
  //int p;
  //int n;
  real log2pi;
  real lambda_tau = lambda * tau;
  vector[p] s;
  vector[p] s_old;
  vector[p] s1;
  vector[n] eta;
  vector[n] W;
  vector[n] z;
  vector[n] mu;
  vector[n] mu_p;
  matrix[p, p] C;
  real logdetI;
  int is_converged;
  vector[p] betahat;
  // vector[p+1] res;
  matrix[p,p] Iinv;
  real logp;
  real laplace;

  log2pi     = 1.83787706641;
  lambda_tau = lambda * tau;
  s          = start;
  eta        = rep_vector(1.0, n);
  for (i in 1:maxit) {
    s_old = s;
    mu   = mu_linkinv(eta, mu_link);
    mu_p = mu_eta(eta, mu_link, n);
    z    = eta + (m - mu) ./ mu_p;
    W    = square(mu_p) ./ vfun(mu, dist);
    C    = Q' * diag_pre_multiply(W, Q);
    s    = mdivide_left_spd(C, Q' * (W .* z));
    eta  = Q * s;

    is_converged = sqrt(sum(square(s - s_old))) < tol;
    if (is_converged == 1) break;
  }
  // get MLE
  betahat = Rinv * Q' * eta;
  // res[1:p] = betahat;

  // get expected fisher info
  Iinv = 1.0 / lambda_tau * X' * diag_pre_multiply(W, X);

  // compute log determinant of expected fisher info
  logdetI = -1.0 * log_determinant(Iinv);

  // compute log of laplace approximation of NC
  laplace = 0.5 * ( p * log2pi + logdetI ) + glm_logpost(betahat, 1.0/tau, m, lambda, X, dist, mu_link, n);;
  // res[p+1] = laplace;
  return laplace;
}


real locprior_lpdf(vector nu, real lambda0, real tau, vector m0, int dist) {
  return lambda0 * tau * ( sum(m0 .* nu) - sum_bfun(nu, dist) );
}



