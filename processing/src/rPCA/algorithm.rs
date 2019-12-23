#![allow(non_snake_case, non_camel_case_types)]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate openblas_src;

use crate::rPCA::hyperparameters::rPCAHyperParams;
use ndarray::{ArrayBase, Array2, Data, Ix2, Zip};
use ndarray_linalg::{SVD};


pub struct rPCA {
    L: Array2<f64>,
    S: Array2<f64>,
}

/// rPCA method based on https://statistics.stanford.edu/sites/g/files/sbiybj6031/f/2009-13.pdf
/// with matlab implementation, https://github.com/dlaptev/RobustPCA
impl rPCA {
    pub fn fit (
        hyperparameters: rPCAHyperParams,
        X : &ArrayBase<impl Data<Elem = f64>, Ix2>,
    ) -> Self {
        let (_n, _m) = X.dim();

        let normX = frobenius_norm(X);

        let mut L = Array2::zeros(X.dim()) as Array2<f64>;
        let mut S = Array2::zeros(X.dim()) as Array2<f64>;
        let mut Y = Array2::zeros(X.dim()) as Array2<f64>;
        let mut Z = Array2::zeros(X.dim()) as Array2<f64>;

        let mut tempY = Array2::zeros(X.dim()) as Array2<f64>;
        let mut tempDo = Array2::zeros(X.dim()) as Array2<f64>;
        let mut tempLo = Array2::zeros(X.dim()) as Array2<f64>;
        let mut err = 0.0;

        for i in 0..hyperparameters.max_n_iterations {
            tempY = Y.map(|x| x * (1.0/hyperparameters.mu));

            // ADMM step, update L and S

            Zip::from(&mut tempDo)
                .and(X)
                .and(&S)
                .and(&tempY)
                .apply(|a, &b, &c, &d| {
                    *a = b - c + d;
                });
            L = Do(1.0/hyperparameters.mu, &tempDo);

            Zip::from(&mut tempLo)
                .and(X)
                .and(&L)
                .and(&tempY)
                .apply(|a, &b, &c, &d| {
                    *a = b - c + d;
                });
            S = So(hyperparameters.lambda/hyperparameters.mu, &tempLo);

            // and augmented lagrangian multiplier
            Zip::from(&mut Z)
                .and(X)
                .and(&L)
                .and(&S)
                .apply(|a, &b, &c, &d| {
                    *a = b - c - d;
                });
            Y = Y + Z.map(|x| x * hyperparameters.mu);

            err = frobenius_norm(&Z) / normX;

            if err < hyperparameters.tolerance {
               break
            }
        }

        Self {
            L,
            S,
        }
    }

    pub fn L(&self) -> &Array2<f64> { &self.L }

    pub fn S(&self) -> &Array2<f64> { &self.S }
}


/// Shrinkage Operator for Singular Values
pub fn Do (
    tau : f64,
    X : &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array2<f64> {
    let temp:Array2<f64> = X.to_owned().clone();
    let (u, s, v) = temp.svd(true, true).unwrap();
    let s = Array2::from_diag(&s);
    let so = So(tau, &s);
    let u = u.unwrap() as Array2<f64>;
    let v = v.unwrap() as Array2<f64>;
    let mut r = u.dot(&so);
    r = r.dot(&v.t());
    r
}

/// Shrinkage Operator
pub fn So (
    tau : f64,
    X : &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array2<f64> {
    let temp = X.map(|x| x.abs()) - tau;
    for i in X.iter() {
        if *i > 0.0 {
            return sign(X) * *i
        }
    }
    sign(X) * 0.0
}

/// returns array the same size as X where
/// out[i, j] = 1 if x[i, j] > 0
/// out[i, j] = 0 if x[i, j] = 0
/// out[i, j] = -1 if x[i, j] < 0
pub fn sign (
    X: &ArrayBase<impl Data<Elem=f64>, Ix2>,
) -> Array2<f64> {
    let (_n, _m) = X.dim();
    let mut out = X.clone().to_owned();
    Zip::from(&mut out)
                .and(X)
                .apply(|a, &b| {
                    if b == 0.0 {
                        *a = 0.0;
                    }
                    else if b > 0.0 {
                        *a = 1.0;
                    }
                    else {
                        *a = -1.0;
                    }});
    out
}

pub fn frobenius_norm (
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> f64 {
    x.map(|x| x.powi(2)).sum().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, Array2};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use crate::rPCAHyperParams;

    #[test]
    fn test_rPCA() {

        // Let's setup a synthetic set of observations, composed of two clusters with known means
        let X: Array2<f64> = Array::random((100, 100), Uniform::new(-100., 100.));
        let (_n, _m) = X.dim();
        let hyperparams = rPCAHyperParams::new(_n as u64, _m as u64).build();

        let model = rPCA::fit(hyperparams, &X);

        /*
        // Does it work?
        let centroids = compute_centroids(2, &observations, &memberships);
        assert_abs_diff_eq!(
            centroids.index_axis(Axis(0), 0),
            expected_centroid_1,
            epsilon = 1e-5
        );
        assert_abs_diff_eq!(
            centroids.index_axis(Axis(0), 1),
            expected_centroid_2,
            epsilon = 1e-5
        );

        assert_eq!(centroids.len_of(Axis(0)), 2);
        */
    }
}