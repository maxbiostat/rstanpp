% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/glm_lognc.R
\name{logncpp_glm_bridge}
\alias{logncpp_glm_bridge}
\title{Estimate logarithm of normalizing constant using bridge sampling}
\usage{
logncpp_glm_bridge(
  formula,
  family,
  histdata,
  a0,
  beta0,
  Sigma0,
  offset,
  disp.shape,
  disp.scale,
  bridge.args = NULL,
  ...
)
}
\arguments{
\item{formula}{an object of class \code{\link[stats]{formula}}.}

\item{family}{an object of type \code{\link[stats]{family}} giving distribution and link function}

\item{histdata}{a \code{\link[base]{data.frame}} of historical data giving all variables in \code{formula}}

\item{a0}{a positive scalar no larger than 1 giving the power prior parameter}

\item{beta0}{mean for initial prior on regression coefficients. Defaults to vector of 0s}

\item{Sigma0}{covariance matrix for initial prior on regression coefficients. Defaults to \code{diag(100, ncol(X))}}

\item{offset}{offset in GLM. If \code{NULL}, no offset is utilized}

\item{disp.shape}{shape parameter for inverse-gamma prior on dispersion parameter (for Gaussian and gamma models). Ignored for binomial and Poisson models}

\item{disp.scale}{rate parameter for inverse-gamma prior on dispersion parameter (for Gaussian and gamma models). Ignored for binomial and Poisson models}

\item{bridge.args}{(optional) parameters to pass onto \code{bridgesampling::bridge_sampler} (otherwise, default is performed)}

\item{...}{(optional) parameters to pass onto \code{rstan::sampling}}
}
\description{
Estimate the logarithm of the normalizing constant via importance sampling. Uses the \code{optimizing} function in Stan to compute
the posterior mode and Hessian, then uses a normal approximation
}
