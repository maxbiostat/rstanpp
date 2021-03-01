



#' Bayesian linear regression with conjugate prior
#'
#' Fits a Bayesian generalized linear model using the conjugate prior of Chen and Ibrahim (2003).
#'
#' @export
#' @param formula          an object of class \code{\link[stats]{formula}}.
#'                         Two-sided formulas (e.g., \code{y ~ x}) samples from the posterior density while
#'                         one-sided formulas (e.g., \code{~ x}) samples from the prior density.
#' @param family           an object of type \code{\link[stats]{family}} giving distribution and link function
#' @param data             a \code{\link[base]{data.frame}} giving all variables in \code{formula}
#' @param prior.precision  a scalar giving the prior precision parameter. This is denoted by \eqn{a_0} in Chen and Ibrahim (2003)
#' @param prior.y0         a vector of length \code{nrow(data)} giving the pseudo-data.
#'                         The range of each element of \code{prior.y0} should be same as the mean (e.g., for Bernoulli models, each element should
#'                         be between 0 and 1). This is denoted by \eqn{y_0} in Chen and Ibrahim (2003)
#' @param dispersion       a scalar or string. If a scalar, assumes fixed dispersion parameter equal to a scalar. If a string, must be one of 'mle' or
#'                         random. If 'random' is specified or formula, the user must supply positive
#'                         scalar parameter values to \code{dispersion.shape} and \code{dispersion.rate}. Must be a positive scalar if
#'                         \code{formula} is one-sided and family is not \code{binomial} or \code{poisson}.
#' @param dispersion.shape positive scalar giving prior shape hyperparameter for dispersion parameter (if \code{dispersion=="random"}). Ignored otherwise.
#' @param dispersion.rate  positive scalar giving prior rate hyperparameter for dispersion parameter (if \code{dispersion=="random"}). Ignored otherwise.
#' @param ...              arguments passed to \code{\link[rstan]{sampling}} (e.g. iter, chains).
#'
#' @return an object of class [rstan::stanfit] returned by `rstan::sampling`
#'
#' @examples
#' N = 50
#' x = rnorm(N, sd = 5)
#' y = rnorm(0.5 + 0.25 * x, sd = 1)
#' data = data.frame(y = y, x = x)
#' ## NOT RUN--SAMPLE FROM PRIOR
#' # prior_sample = rstanglm_fixedDispersion(
#' #   ~ x, gaussian(), data = data,
#' #   prior.y0 = rep(0, N), dispersion = 1)
#'
#' ## SAMPLE FROM POSTERIOR
#' # options(mc.cores = parallel::detectCores())
#' rstan::rstan_options(auto_write = TRUE)
#' post_sample = rstanglm(
#' y ~ x, gaussian(), data = data, prior.y0 = rep(0, N),
#' dispersion = 'random', dispersion.shape = 1, dispersion.rate = .1)  ## sample from posterior with random dispersion parameter
#'
rstanglm = function(formula, family, data, prior.precision = 1, prior.y0, dispersion, dispersion.shape = NULL, dispersion.rate = NULL,  ...) {
  ## call sampler with fixed dispersion if dispersion is 'mle' or dispersion is numeric
  if ( dispersion == 'mle' | is.numeric(dispersion) ) {
    return( rstanglm_fixedDispersion(formula, family, data, prior.precision, prior.y0, dispersion, ...) )
  } else if ( dispersion == 'random' ) {
    if (is.null(dispersion.rate))      {stop('dispersion.rate must be a positive scalar if dispersion is random')}
    else if (is.null(dispersion.shape)){stop('dispersion.shape must be a positive scalar if dispersion is random')}
    return( rstanglm_randomDispersion(formula, family, data, prior.precision, prior.y0, dispersion.shape = dispersion.shape, dispersion.rate = dispersion.rate, ...) )
  }
  stop('dispersion must be one of mle, random, or a positive constant.')
}
