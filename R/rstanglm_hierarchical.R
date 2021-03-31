


#' Hierarchical Bayesian generalized linear models
#'
#' Sample from the posterior density of a hierarchical Bayesian GLM using Stan.
#' Currently only supports binomial and Poisson models.
#'
#' @param formula           a two-sided object of class \code{\link[stats]{formula}}.
#' @param family            an object of type \code{\link[stats]{family}} giving distribution and link function
#' @param data              a \code{\link[base]{data.frame}} giving all variables in \code{formula}
#' @param beta.precision    a scalar giving the prior precision parameter for the regression coefficients. This is denoted by \eqn{a_0} in Chen and Ibrahim (2003)
#' @param m.precision       a scalar giving the precision parameter for the hierarchical prior
#' @param m.location        a \eqn{n}-dimensional vector giving the prior prediction for the responses
#' @param start             a \eqn{p}-dimensional vector giving starting value for IRLS (used for computing the normalizing constant). The default utilizes the value of \code{m.location}
#' @param maxit             an integer giving the maximum number of iterations for Fisher scoring (Used for computing the normalizing constant).
#' @param tol               tolerance parameter in Fisher scoring algorithm
#' @param ...               arguments passed to \code{\link[rstan]{sampling}} (e.g. iter, chains, warmup, etc.).
#'
#' @return an object of class [rstan::stanfit] returned by `rstan::sampling`
#' @examples
#' ## binary regression example
#' set.seed(123)
#' n      = 100
#' X      = cbind(1, rnorm(n, mean = 6, sd = 2))
#' beta   = c(3, -0.5)
#' eta    = X %*% beta
#' mu     = 1 / (1 + exp(-eta))
#' y      = rbinom(n, 1, mu)
#' lambda = 20
#'
#' fit       = glm(y ~ 0 + X, family = binomial())
#' betahat   = coef(fit)
#' thetahat  = X %*% betahat
#'
#' b0      = c(3, -.25)
#' nu0     = X %*% b0
#' m0      = make.link('logit')$linkinv(nu0)
#' lambda0 = 10
#' data = data.frame(y = y, x = X[, 2])
#'
#' # options(mc.cores = parallel::detectCores())
#' rstan::rstan_options(auto_write = TRUE)
#' smpl = rstanglm_hierarchical(
#'   y ~ x, binomial(), data, 10, 20, as.vector(m0), chains = 1, iter = 100
#' )
#' @export
rstanglm_hierarchical = function(
  formula, family, data, beta.precision, m.precision, m.location, start = NULL, maxit = 100, tol = 1e-6, ...
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

  if ( is.null(start) ) {
    start = coef(glm(m.location ~ 0 + X))
  }

  ## assemble stan data
  standat = list(
    'N'                = nrow(X),
    'K'                = ncol(X),
    'y'                = data[, all.vars(formula)[1]],
    'X'                = X,
    'lambda'           = beta.precision,
    'lambda0'          = m.precision,
    'm0'               = m.location,
    'mu_link'          = mu_link,
    'start'            = start,
    'maxit'            = maxit,
    'tol'              = tol
  )

  ## call stan and return stanobject
  if ( family$family == 'binomial' ) {
    return(
      rstan::sampling(
      object = stanmodels$sample_hierarchical_binomial,
      data   = standat,
      ...
      )
    )
  }
  if ( family$family == 'poisson' ) {
    return(
      rstan::sampling(
        object = stanmodels$sample_hierarchical_poisson,
        data   = standat
      )
    )
  }
  stop("Invalid family")
}
