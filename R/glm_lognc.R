


#' Estimate logarithm of normalizing constant using bridge sampling
#' 
#' Estimate the logarithm of the normalizing constant via importance sampling. Uses the \code{optimizing} function in Stan to compute
#' the posterior mode and Hessian, then uses a normal approximation
#' 
#' @include glm_npp_prior.R
#' 
#' @param formula         an object of class \code{\link[stats]{formula}}.
#' @param family          an object of type \code{\link[stats]{family}} giving distribution and link function
#' @param histdata        a \code{\link[base]{data.frame}} of historical data giving all variables in \code{formula}
#' @param a0              either a positive integer giving the number of (equally spaced) values to compute the normalizing constant of the power prior, or a vector of values between 0 and 1 giving values of the normalizing constant to compute
#' @param beta0           mean for initial prior on regression coefficients. Defaults to vector of 0s
#' @param Sigma0          covariance matrix for initial prior on regression coefficients. Defaults to \code{diag(100, ncol(X))}
#' @param offset          offset in GLM. If \code{NULL}, no offset is utilized
#' @param invdisp.shape   shape parameter for inverse dispersion (for Gaussian and gamma models). Ignored for binomial and Poisson models
#' @param invdisp.rate    rate parameter for inverse dispersion (for Gaussian and gamma models). Ignored for binomial and Poisson models
#' @param nsmpl           (optional) number of importance samples to take (ignored if \code{method == 'bridge'})
#' @param bridge.args     (optional) parameters to pass onto `bridgesampling::bridge_sampler` (otherwise, default is performed) (ignored if \code{method == 'importance'})
#' @param ...             (optional) parameters to pass onto `rstan::sampling` (ignored if \code{method == 'importance'})
#' 
#' @export
logncpp_glm = function(formula, family, histdata, a0, beta0 = NULL, Sigma0 = NULL, offset = NULL, invdisp.shape = 1.5, invdisp.rate = .25, method = 'bridge', nsmpl = 1000, bridge.args = NULL, ...) {
  ## if input is a positive integer > 1, create equally-spaced grid of values between 0-1 of length a0
  if (length(a0) == 1 & a0%%1 == 0 & a0 > 1) {
    a0 = seq(0, 1, length.out = a0)
  } else if ( any(a0 < 0) | any(a0 > 1) ) {
    stop("a0 must be a positive integer or a vector of values between 0 and 1")
  }
  lognc = numeric(length(a0))
  if ( method == 'bridge' ) {
    for ( i in seq_along(a0) ) {
      lognc[i] = logncpp_glm_bridge(formula, family, histdata, a0[i], beta0, Sigma0, offset, invdisp.shape, invdisp, bridge.args, ...)
    }
  }
  if ( method == 'importance' ){
    for ( i in seq_along(a0) ) {
      lognc[i] = logncpp_glm_importance(formula, family, histdata, a0[i], beta0, Sigma0, offset, invdisp.shape, invdisp.rate, nsmpl = nsmpl)
    }
  }
  ## result is a data.frame giving (a0, lognc)
  a0lognc                 = data.frame('a0' = a0, 'lognc' = lognc)
  a0lognc                 = a0lognc[order(a0lognc$a0), ]
  attr(a0lognc, 'method') = method
  return(a0lognc)
}




#' Estimate logarithm of normalizing constant using bridge sampling
#' 
#' Estimate the logarithm of the normalizing constant via importance sampling. Uses the \code{optimizing} function in Stan to compute
#' the posterior mode and Hessian, then uses a normal approximation
#' 
#' @param formula         an object of class \code{\link[stats]{formula}}.
#' @param family          an object of type \code{\link[stats]{family}} giving distribution and link function
#' @param histdata        a \code{\link[base]{data.frame}} of historical data giving all variables in \code{formula}
#' @param a0              a positive scalar no larger than 1 giving the power prior parameter
#' @param beta0           mean for initial prior on regression coefficients. Defaults to vector of 0s
#' @param Sigma0          covariance matrix for initial prior on regression coefficients. Defaults to \code{diag(100, ncol(X))}
#' @param offset          offset in GLM. If \code{NULL}, no offset is utilized
#' @param invdisp.shape   shape parameter for inverse dispersion (for Gaussian and gamma models). Ignored for binomial and Poisson models
#' @param invdisp.rate    rate parameter for inverse dispersion (for Gaussian and gamma models). Ignored for binomial and Poisson models
#' @param bridge.args     (optional) parameters to pass onto `bridgesampling::bridge_sampler` (otherwise, default is performed)
#' @param ...             (optional) parameters to pass onto `rstan::sampling`
logncpp_glm_bridge = function(formula, family, histdata, a0, beta0 = NULL, Sigma0 = NULL, offset = NULL, invdisp.shape = 1.5, invdisp.rate = .25, bridge.args = NULL, ...) {
  fit = glm_npp_prior(
    formula, family, histdata = histdata, a0 = a0, ...
  )
  if ( is.null(bridge.args) ) {
    res = bridgesampling::bridge_sampler(fit)
  } else {
    args = c('samples' = fit, bridge.args)
    res = do.call(bridgesampling::bridge_sampler, args)
  }
  return(res$logml)
}






#' Estimate logarithm of normalizing constant using importance sampling
#' 
#' Estimate the logarithm of the normalizing constant via importance sampling. Uses the \code{optimizing} function in Stan to compute
#' the posterior mode and Hessian, then uses a normal approximation
#' 
#' @param formula         an object of class \code{\link[stats]{formula}}.
#' @param family          an object of type \code{\link[stats]{family}} giving distribution and link function
#' @param histdata        a \code{\link[base]{data.frame}} of historical data giving all variables in \code{formula}
#' @param a0              a positive scalar no larger than 1 giving the power prior parameter
#' @param beta0           mean for initial prior on regression coefficients. Defaults to vector of 0s
#' @param Sigma0          covariance matrix for initial prior on regression coefficients. Defaults to \code{diag(100, ncol(X))}
#' @param offset          offset in GLM. If \code{NULL}, no offset is utilized
#' @param invdisp.shape   shape parameter for inverse dispersion (for Gaussian and gamma models). Ignored for binomial and Poisson models
#' @param invdisp.rate    rate parameter for inverse dispersion (for Gaussian and gamma models). Ignored for binomial and Poisson models
#' @param nsmpl           number of importance samples to take
#' 
#' @return scalar giving log normalizing constant
#' @export
logncpp_glm_importance = function(formula, family, histdata, a0, beta0 = NULL, Sigma0 = NULL, offset = NULL, invdisp.shape = 1.5, invdisp.rate = .25, nsmpl = 1000) {
  ## get design matrix
  X = model.matrix(formula, histdata)
  
  ## get response
  y0 = histdata[, all.vars(formula)[1]]
  
  
  ## get mu link function as integer
  links = c('identity', 'log', 'logit', 'inverse', 'probit', 'cauchit', 'cloglog', 'sqrt', '1/mu^2')
  mu_link = which(links == family$link)[1]
  if ( length(mu_link) == 0 ) { stop(paste('Link must be one of', paste(links, collapse = ', '))) }
  
  ## get offset if applicable
  incl_offset = 1
  if ( is.null(offset) ) {
    incl_offset = 0
    offset = rep(0, length(y0))
  }
  
  ## default mean for beta is 0
  if(is.null(beta0)) {
    beta0 = rep(0, ncol(X))
  }
  
  ## default covariance for beta is diag(100)
  if(is.null(Sigma0)) {
    Sigma0 = diag(100, ncol(X))
  }
  
  
  ## assemble stan data
  standat = list(
    'nobs'        = nrow(X),
    'p'           = ncol(X),
    'y0'          = y0,
    'X'           = X,
    'a0'          = a0,
    'beta0'       = beta0,
    'Sigma0'      = Sigma0,
    'link'        = mu_link,
    'incl_offset' = incl_offset,
    'offset'      = offset
  )
  
  if ( family$family %in% c("gaussian", "Gamma") ) {
    standat = c(standat, 'invdisp_shape' = invdisp.shape, 'invdisp_rate' = invdisp.rate)
  }
  
  ## call stan and return stanobject
  if (family$family == 'binomial') {
    opt = rstan::optimizing(
      object  = stanmodels$glm_pp_bernoulli,
      data    = standat,
      hessian = T
    )
    stanobj = suppressMessages(
      rstan::sampling(
        object = stanmodels$glm_pp_bernoulli,
        data   = standat,
        chains = 0
      )
    )
  }
  
  if ( family$family == "poisson" ) {
    opt = rstan::optimizing(
      object = stanmodels$glm_pp_poisson,
      data    = standat,
      hessian = T
    )
    stanobj = suppressMessages(
      rstan::sampling(
        object = stanmodels$glm_pp_poisson,
        data   = standat,
        chains = 0
      )
    )
  }
  
  if ( family$family == "gaussian" ) {
    opt = rstan::optimizing(
      object  = stanmodels$glm_pp_gaussian,
      data    = standat,
      hessian = T
    )
    stanobj = suppressMessages(
      rstan::sampling(
        object  = stanmodels$glm_pp_gaussian,
        data   = standat,
        chains = 0
      )
    )
  }
  
  if ( family$family == "Gamma" ) {
    opt = rstan::optimizing(
      object  = stanmodels$glm_pp_gamma,
      data    = standat,
      hessian = T
    )
    stanobj = suppressMessages(
      rstan::sampling(
        object  = stanmodels$glm_pp_gamma,
        data   = standat,
        chains = 0
      )
    )
  }
  
  ## obtain importance sample from MVN
  invneghess = chol2inv(chol(-opt$hessian) )
  smpl = mvtnorm::rmvnorm(n = nsmpl, mean = opt$par, sigma = invneghess )
  
  ## compute log weights
  logW = apply(smpl, 1, function(x) log_prob(stanobj, x) )
  logW = logW - mvtnorm::dmvnorm(smpl, opt$par, invneghess, log = T)
  
  ## log-sum-exp trick to compute log NC
  M     = max(logW)
  lognc = -log(nsmpl) + M + log( sum( exp( logW - M ) ) )
  
  return(lognc)
}