/**
 * Inverse link function on mean scale
 *
 * @param eta  linear predictor vector (X * beta)
 * @param mu_link an integer indicating the link function to the mean
 * @return A vector, i.e. inverse-link(eta)
 */
vector linkinv(vector eta, int mu_link) {
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




