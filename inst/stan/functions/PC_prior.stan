real PC_lpdf(real x, real S, real p, real  a){
  /* Implements the Gumbel type II prior on the precision */
  real b = -log(p)/S;
  real ans = log(a) + log(b) + -(a  + 1)*log(x) - b * x^(-a); 
  return(ans);
}
