#![allow(non_snake_case, non_camel_case_types)]
use serde::{Deserialize, Serialize};
use std::cmp::max;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]

pub struct rPCAHyperParams {
    /// lambda - regularization parameter, default = 1/sqrt(max(N,M))
    pub(crate) lambda: f64,
    /// mu - the augmented lagrangian parameter, default = 10*lambda
    pub(crate) mu : f64,
    /// tolerance - reconstruction error tolerance, default = 1e-6
    pub(crate) tolerance : f64,
    /// max_n_iterations - maximum number of iterations, default = 1000
    pub(crate) max_n_iterations : u64,
}

/// An helper struct used to construct a set of [valid hyperparameters]
pub struct rPCAHyperParamsBuilder {
    lambda: f64,
    mu : f64,
    tolerance : f64,
    max_n_iterations: u64,
}

impl rPCAHyperParamsBuilder {

    /// Set the value of `lambda`.
    ///
    /// Lambda is simply a regularization parameter used during the
    /// ADMM loop
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Set the value of `mu`.
    ///
    /// Mu is the augmented lagrangian multiplier
    pub fn mu(mut self, mu: f64) -> Self {
        self.mu = mu;
        self
    }

    /// Set the value of `tolerance`.
    ///
    /// The training is considered complete if the difference between
    /// the frobenius norm of the original matrix of data and
    /// the matrix created from the augmented lagrangian multiplier is less than the tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the value of `max_n_iterations`.
    ///
    /// We exit the ADMM (alternating direction method of multipliers) loop once
    /// the max_n_iterations have been reached or the error is less
    /// than the tolerance
    pub fn max_n_iterations(mut self, max_n_iterations: u64) -> Self {
        self.max_n_iterations = max_n_iterations;
        self
    }

    /// Return an instance of `rPCAMeansHyperParams` after
    /// having performed validation checks on all the specified hyperparamters.
    ///
    /// **Panics** if any of the validation checks fails.
    pub fn build(self) -> rPCAHyperParams {
        rPCAHyperParams::build(self.lambda, self.mu, self.tolerance, self.max_n_iterations)
    }
}

impl rPCAHyperParams {
    /// `new` lets us configure our training algorithm parameters:
    /// * we will be looking for `n_clusters` in the training dataset;
    /// * the training is considered complete if the euclidean distance
    ///   between the old set of centroids and the new set of centroids
    ///   after a training iteration is lower or equal than `tolerance`;
    /// * we exit the training loop when the number of training iterations
    ///   exceeds `max_n_iterations` even if the `tolerance` convergence
    ///   condition has not been met.
    ///
    /// `n_clusters` is mandatory.
    ///
    /// Defaults are provided if optional parameters are not specified:
    /// * `tolerance = 1e-4`;
    /// * `max_n_iterations = 300`.
    pub fn new(_n : u64, _m : u64) -> rPCAHyperParamsBuilder {
        rPCAHyperParamsBuilder {
            lambda : 1.0 / (max(_n, _m) as f64).sqrt() as f64,
            mu : 10.0 / (max(_n, _m) as f64).sqrt() as f64,
            tolerance : 1.0e-6,
            max_n_iterations : 1000,
        }
    }

    pub fn lambda(&self) -> f64 { self.lambda }

    /// The number of clusters we will be looking for in the training dataset.
    pub fn mu(&self) -> f64 { self.mu }

    /// The training is considered complete if the euclidean distance
    /// between the old set of centroids and the new set of centroids
    /// after a training iteration is lower or equal than `tolerance`.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// We exit the training loop when the number of training iterations
    /// exceeds `max_n_iterations` even if the `tolerance` convergence
    /// condition has not been met.
    pub fn max_n_iterations(&self) -> u64 {
        self.max_n_iterations
    }

    fn build(lambda : f64, mu : f64, tolerance : f64, max_n_iterations : u64) -> Self {
        if max_n_iterations == 0 {
            panic!("`max_n_iterations` cannot be 0!");
        }
        if tolerance <= 0. {
            panic!("`tolerance` must be greater than 0!");
        }
        if mu == 0.0 {
            panic!("`mu` cannot be 0!");
        }
        if lambda == 0.0 {
            panic!("`lambda` cannot be 0!");
        }
        rPCAHyperParams {
            lambda,
            mu,
            tolerance,
            max_n_iterations,
        }
    }
}
