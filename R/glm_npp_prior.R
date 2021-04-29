


#' Sample from prior of power prior for a GLM
#' 
#' Uses `rstan` to sample from the prior of a power prior for a generalized linear model
#'
#' @param formula         an object of class \code{\link[stats]{formula}}.
#' @param family          an object of type \code{\link[stats]{family}} giving distribution and link function
#' @param histdata        a \code{\link[base]{data.frame}} of historical data giving all variables in \code{formula}
#' @param a0              a positive scalar no larger than 1 giving the power prior parameter
#' @param beta0           mean for initial prior on regression coefficients. Defaults to vector of 0s
#' @param Sigma0          covariance matrix for initial prior on regression coefficients. Defaults to \code{diag(100, ncol(X))}
#' @param offset          offset in GLM. If \code{NULL}, no offset is utilized
#' @param disp.shape      shape parameter for inverse-gamma prior on dispersion (for Gaussian and gamma models). Ignored for binomial and Poisson models
#' @param disp.scale      scale parameter for inverse-gamma prior on dispersion (for Gaussian and gamma models). Ignored for binomial and Poisson models
#' @param ...             optional parameters to pass onto `rstan::sampling`
#'
#' @return an object of class [rstan::stanfit] returned by `rstan::sampling`
#' @examples
#' N = 50
#' @export
glm_npp_prior = function(
  formula, family, histdata, a0, beta0 = NULL, Sigma0 = NULL, offset = NULL, disp.shape = 1e-4, disp.scale = 1e-4, ...
) {
  ## get design matrix
  X = model.matrix(formula, histdata)
  
  ## get response
  y0 = histdata[, all.vars(formula)[1]]
  

  ## get mu link function as integer
  links = c('identity', 'log', 'logit', 'inverse', 'probit', 'cauchit', 'cloglog', 'sqrt', '1/mu^2')
  mu_link = which(links == family$link)[1]
  if ( length(mu_link) == 0 ) { stop(paste('Link must be one of', paste(links, collapse = ', '))) }
  
  incl_offset = 1
  if ( is.null(offset) ) {
    incl_offset = 0
    offset = rep(0, length(y0))
  }
  
  if(is.null(beta0)) {
    beta0 = rep(0, ncol(X))
  }
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
    standat = c(standat, 'disp_shape' = disp.shape, 'disp_scale' = disp.scale)
  }

  ## call stan and return stanobject
  if (family$family == 'binomial') {
    return(
      rstan::sampling(
        object = stanmodels$glm_pp_bernoulli,
        data   = standat,
        ...
      )
    )
  }
  
  if ( family$family == "poisson" ) {
    return(
      rstan::sampling(
        object = stanmodels$glm_pp_poisson,
        data   = standat,
        ...
      )
    )
  }
  
  if ( family$family == "gaussian" ) {
    return(
      rstan::sampling(
        object = stanmodels$glm_pp_gaussian,
        data   = standat,
        ...
      )
    )
  }
  
  if ( family$family == "Gamma" ) {
    return(
      rstan::sampling(
        object = stanmodels$glm_pp_gamma,
        data   = standat,
        ...
      )
    )
  }
  stop("Invalid family")
  return(NA);   ## never reached
}
