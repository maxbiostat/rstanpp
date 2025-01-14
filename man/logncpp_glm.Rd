% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/glm_lognc.R
\name{logncpp_glm}
\alias{logncpp_glm}
\title{Estimate logarithm of normalizing constant using bridge sampling}
\usage{
logncpp_glm(
  formula,
  family,
  histdata,
  a0,
  beta0 = NULL,
  Sigma0 = NULL,
  offset = NULL,
  disp.shape = 2.1,
  disp.scale = 1.1,
  method = "bridge",
  nsmpl = 1000,
  bridge.args = NULL,
  ...
)
}
\arguments{
\item{formula}{an object of class \code{\link[stats]{formula}}.}

\item{family}{an object of type \code{\link[stats]{family}} giving distribution and link function}

\item{histdata}{a \code{\link[base]{data.frame}} of historical data giving all variables in \code{formula}}

\item{a0}{either a positive integer giving the number of (equally spaced) values to compute the normalizing constant of the power prior, or a vector of values between 0 and 1 giving values of the normalizing constant to compute}

\item{beta0}{mean for initial prior on regression coefficients. Defaults to vector of 0s}

\item{Sigma0}{covariance matrix for initial prior on regression coefficients. Defaults to \code{diag(100, ncol(X))}}

\item{offset}{offset in GLM. If \code{NULL}, no offset is utilized}

\item{disp.shape}{shape parameter for inverse-gamma prior on dispersion parameter (for Gaussian and gamma models). Ignored for binomial and Poisson models}

\item{disp.scale}{rate parameter for inverse-gamma prior on dispersion parameter (for Gaussian and gamma models). Ignored for binomial and Poisson models}

\item{method}{character vector giving which method to use for importance sampling. Acceptable values are \code{"bridge"} or\code{"importance"}, corresponding to bridge sampling and importance sampling, respectively}

\item{nsmpl}{(optional) number of importance samples to take (ignored if \code{method == 'bridge'})}

\item{bridge.args}{(optional) parameters to pass onto \code{bridgesampling::bridge_sampler} (otherwise, default is performed) (ignored if \code{method == 'importance'})}

\item{...}{(optional) parameters to pass onto \code{rstan::sampling} (ignored if \code{method == 'importance'})}
}
\description{
Estimate the logarithm of the normalizing constant via importance sampling. Uses the \code{optimizing} function in Stan to compute
the posterior mode and Hessian, then uses a normal approximation
}
