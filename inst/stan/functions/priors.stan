

/****************************
Binomial functions
****************************/

real binomial_conjugate_lpdf(vector beta, vector m, matrix X, real lambda, int mu_link) {
  real res;
  vector[num_elements(m)] eta = X * beta;
  if (mu_link != 3) {
      eta = mu_linkinv(eta, mu_link);   // eta <- mu
      eta = logit(eta);                 // eta <- theta
  }
  return lambda * sum( m .* eta - log(1.0 + exp(eta)) );
}

/****************************
Poisson functions
****************************/

real poisson_conjugate_lpdf(vector beta, vector m, matrix X, real lambda, int mu_link) {
  vector[rows(m)] eta = X * beta;
  if (mu_link != 2) {
      eta = mu_linkinv(eta, mu_link);   // eta <- mu
      eta = log(eta);                   // eta <- theta
  }
  return lambda * sum( m .* eta - exp(eta) );
}
//
//
// /****************************
// Gaussian functions
// ****************************/
// real gaussian_location_lpdf(vector m, real lambda0, vector m0, real tau) {
//   return tau * lambda0 * sum( m0 .* m - 0.5 * square(m) );
// }
//
// real gaussian_conjugate_lpdf(vector beta, vector m, matrix X, real lambda, real tau, int mu_link) {
//   vector[rows(m)] eta = X * beta;
//   if (mu_link != 1) {
//       eta = mu_linkinv(eta, mu_link);   // eta <- mu = theta
//   }
//   return tau * lambda * sum( m .* eta - 0.5 * square(eta) );
// }
//
//
// /****************************
// Gamma functions
// ****************************/
// real Gamma_location_lpdf(vector m, real lambda0, vector m0, real tau) {
//   return tau * lambda0 * sum( m0 .* nu + log(nu) );
// }
//
// real Gamma_conjugate_lpdf(vector beta, vector m, matrix X, real lambda, real tau, int mu_link) {
//   vector[rows(m)] eta = X * beta;
//   if (mu_link != 4) {
//       eta = mu_linkinv(eta, mu_link);   // eta <- mu
//       eta = inv(eta);                   // eta <- theta
//   }
//   return tau * lambda * sum( m .* eta + log(eta) );
// }


