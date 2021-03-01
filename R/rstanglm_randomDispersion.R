


#' Bayesian linear regression with Stan with a fixed dispersion parameter
#'
#' @param formula           an object of class \code{\link[stats]{formula}}.
#'                          Two-sided formulas (e.g., \code{y ~ x}) samples from the posterior density while
#'                          one-sided formulas (e.g., \code{~ x}) samples from the prior density.
#' @param family            an object of type \code{\link[stats]{family}} giving distribution and link function
#' @param data              a \code{\link[base]{data.frame}} giving all variables in \code{formula}
#' @param prior.precision   a scalar giving the prior precision parameter. This is denoted by \eqn{a_0} in Chen and Ibrahim (2003)
#' @param prior.y0          a vector of length \code{nrow(data)} giving the pseudo-data.
#'                          The range of each element of \code{prior.y0} should be same as the mean (e.g., for Bernoulli models, each element should
#'                          be between 0 and 1). This is denoted by \eqn{y_0} in Chen and Ibrahim (2003)
#' @param dispersion.shape  positive scalar giving the gamma prior shape hyperparameter for the inverse dispersion parameter (\code{dispersion^(-1) ~ Gamma(dispersion.shape, dispersion.rate)})
#' @param dispersion.rate   positive scalar giving the gamma prior rate hyperparameter for the inverse dispersion parameter (\code{dispersion^(-1) ~ Gamma(dispersion.shape, dispersion.rate)})
#' @param ...               arguments passed to \code{\link[rstan]{sampling}} (e.g. iter, chains).
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
#' rstan_options(auto_write = TRUE)
#' post_sample = rstanglm_randomDispersion(y ~ x, gaussian(), data = data, prior.y0 = rep(0, N), dispersion.shape = 1, dispersion.rate = .01)  ## sample from posterior with mle as dispersion parameter
rstanglm_randomDispersion = function(
  formula, family, data, prior.precision = 1, prior.y0, dispersion.shape = 1, dispersion.rate = 0.1, ...
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

  ## stop if family is poisson/binomial
  if ( family$family %in% c('poisson', 'binomial') ) {
    stop('dispersion parameter cannot be random for poisson and binomial models')
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

  ## get part of sum(c(y, phi)) that depends only on y; call sum_fy;
  sum_fy = 0
  if ( family$family == 'gaussian' ) { sum_fy = sum(y^2) }
  else if ( family$family == 'Gamma' ) { sum_fy = sum(log(y)) }
  else stop("Family must be 'gaussian' or 'Gamma' for random dispersion models")

  ## assemble stan data
  standat = list(
    'N'                = nrow(X),
    'K'                = ncol(X),
    'X'                = X,
    'm_post'           = post.y0,
    'lambda_post'      = post.precision,
    'mu_link'          = mu_link,
    'dist'             = dist,
    'dispersion_shape' = dispersion.shape,
    'dispersion_rate'  = dispersion.rate,
    'sum_fy'           = sum_fy
  )

  ## call stan and return stanobject
  rstan::sampling(
    object = stanmodels$sample_randomDispersion,
    data   = standat,
    ...
  )
}
