


#' Bayesian linear regression with Stan with a fixed dispersion parameter
#'
#' @param formula         an object of class \code{\link[stats]{formula}}.
#'                        Two-sided formulas (e.g., \code{y ~ x}) samples from the posterior density while
#'                        one-sided formulas (e.g., \code{~ x}) samples from the prior density.
#' @param family          an object of type \code{\link[stats]{family}} giving distribution and link function
#' @param data            a \code{\link[base]{data.frame}} giving all variables in \code{formula}
#' @param prior.precision a scalar giving the prior precision parameter. This is denoted by \eqn{a_0} in Chen and Ibrahim (2003)
#' @param prior.y0        a vector of length \code{nrow(data)} giving the pseudo-data.
#'                        The range of each element of \code{prior.y0} should be same as the mean (e.g., for Bernoulli models, each element should
#'                        be between 0 and 1). This is denoted by \eqn{y_0} in Chen and Ibrahim (2003)
#' @param dispersion      a string or scalar giving the dispersion parameter. If \code{dispersion=='mle'}, uses maximum likelihood estimation
#'                        by calling \code{\link[stats]{glm}}. If a positive scalar, passed on as a hyperparameter. For Bernoulli and Poisson
#'                        models, dispersion is fixed at 1 regardless of user input.
#' @param ...             arguments passed to \code{\link[rstan]{sampling}} (e.g. iter, chains).
#'
#' @return an object of class [rstan::stanfit] returned by `rstan::sampling`
#' @noRd
#' @examples
#' N = 50
#' x = rnorm(N, sd = 5)
#' y = rnorm(0.5 + 0.25 * x, sd = 1)
#' data = cbind(y = y, x = x)
#' ## NOT RUN--SAMPLE FROM PRIOR
#' # prior_sample = rstanglm_fixedDispersion(~ x, gaussian(), data = data, prior.y0 = rep(0, N), dispersion = 1)        ## sample from prior--must specify a constant for dispersion
#'
#' ## SAMPLE FROM POSTERIOR
#' options(mc.cores = parallel::detectCores())
#' rstan_options(auto_write = TRUE)
#' post_sample = rstanglm_fixedDispersion(y ~ x, gaussian(), data = data, prior.y0 = rep(0, N), dispersion = 'mle')  ## sample from posterior with mle as dispersion parameter
rstanglm_fixedDispersion = function(
  formula, family, data, prior.precision = 1, prior.y0, dispersion = 1, ...
) {
  ## get design matrix
  X = model.matrix(formula, data)

  ## get mu link function as integer
  links = c('identity', 'log', 'logit', 'inverse', 'probit', 'cauchit', 'cloglog', 'sqrt', '1/mu^2')
  mu_link = which(links == family$link)[1]
  if ( length(mu_link) == 0 ) { stop(paste('Link must be one of', paste(links, collapse = ', '))) }

  ## get distribution as integer
  dists = c('gaussian', 'binomial', 'poisson', 'Gamma')
  dist  = which(dists == family$family)[1]
  if ( length(dist) == 0 ) { stop(paste('Family must be one of', paste(dists, collapse = ', ')) ) }

  ## get dispersion parameter
  if ( family$family %in% c('poisson', 'binomial') ) {
    dispersion = 1    ## set dispersion = 1 if poisson or bernoulli
  } else if ( dispersion == 'mle' ) {
    if ( formula.tools::is.two.sided(formula) ) {
      dispersion = summary( stats::glm(formula, family, data) )$dispersion
    } else {
      stop('dispersion = "mle" may only be used with two-sided formulae')
    }
  } else if ( dispersion <= 0 ) {
    stop('dispersion must be a positie constant or "mle"')
  }

  ## check prior y0
  if ( length(prior.y0) != nrow(data) ) { stop('prior.y0 must have the same length as the data') }
  if ( (family$family == 'binomial') & ( any(prior.y0 < 0) | any(prior.y0 > 1) ) ) {
    stop('for binomial models, all elements of prior.y0 must be between 0 and 1')
  } else if ( family$family %in% c('poisson', 'Gamma') & (any(prior.y0 < 0 ) ) ) {
    stop(paste0('for', family$family, 'models, all elements of prior.y0 must be positive'))
  }

  ## if rhs formula, sample from prior; else sample from posterior
  if ( formula.tools::is.one.sided(formula) ) {
    post.precision = prior.precision
    post.y0        = prior.y0
  } else if ( formula.tools::is.two.sided(formula) ) {
    post.precision = 1 + prior.precision
    y       = data[[all.vars(formula)[1]]]                               ## dependent variable in glm
    post.y0 = (y + prior.precision * prior.y0) / (1 + prior.precision)   ## posterior location parameter
  }

  ## assemble stan data
  standat = list(
    'N'           = nrow(X),
    'K'           = ncol(X),
    'X'           = X,
    'm_post'      = post.y0,
    'lambda_post' = post.precision,
    'mu_link'     = mu_link,
    'dist'        = dist,
    'dispersion'  = dispersion
  )

  ## call stan and return stanobject
  rstan::sampling(
    object = stanmodels$sample_fixedDispersion,
    data   = standat,
    ...
  )
}
