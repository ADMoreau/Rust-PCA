#![allow(non_snake_case, non_camel_case_types)]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate openblas_src;

use ndarray::{ArrayBase, Array2, Array1, Axis, Data, Ix2, arr2, stack, s, Zip};
use ndarray_linalg::{SVD, Trace};
use std::cmp::max;


pub struct rPCA {
    L: Array2<f64>,
    S: Array2<f64>,
}

/*todo : implement default values for rPCA fit
impl Default for <rPCA as Trait>::fit {
    fn default() -> Self {
        rPCA::fit {
            X: None,
            lambda: 0.0,
            mu: 0.0,
            tol: 1.0 * 10.0 ** -6,
            max_iter: 1000,
        }
    }
}
*/

impl rPCA {

    // X is a data matrix (of the size N x M) to be decomposed
    // lambda - regularization parameter, default = 1/sqrt(max(N,M))
    // mu - the augmented lagrangian parameter, default = 10*lambda
    // tol - reconstruction error tolerance, default = 1e-6
    // max_iter - maximum number of iterations, default = 1000
    pub fn fit (
        X : &ArrayBase<impl Data<Elem = f64>, Ix2>,
        mut lambda: f64,
        mut mu : f64,
        mut tol : f64,
        mut max_iter : i32,
    ) -> Self {
        let (_n, _m) = X.dim();

        if lambda == 0.0 {
            let max = max(_n, _m) as f64;
            lambda = 1.0 / max.sqrt() as f64;
        }
        if mu == 0.0 {
            mu = 10.0 * lambda;
        }
        if tol == 0.0 {
            tol = 1e-6;
        }
        if max_iter == 0 {
            max_iter = 1000;
        }

        let normX = frobenius_norm(X);

        let mut L = Array2::zeros(X.dim());
        let mut S = L.clone();
        let mut Y = L.clone();
        let mut Z = L.clone();

        let mut tempY = Array2::zeros(Y.dim());
        let mut tempDo = Array2::zeros(X.dim());
        let mut tempLo = Array2::zeros(X.dim());
        let mut err = 0.0;

        for i in 0..max_iter {
            tempY = Y.map(|x| x * (1.0/mu));

            // ADMM step, update L and S

            Zip::from(&mut tempDo)
                .and(X)
                .and(&S)
                .and(&tempY)
                .apply(|a, &b, &c, &d| {
                    *a = b - c + d;
                });
            L = Do(1.0/mu, &tempDo);

            Zip::from(&mut tempLo)
                .and(X)
                .and(&L)
                .and(&tempY)
                .apply(|a, &b, &c, &d| {
                    *a = b - c + d;
                });
            S = So(lambda/mu, &tempLo);

            // and augmented lagrangian multiplier
            Zip::from(&mut Z)
                .and(X)
                .and(&L)
                .and(&S)
                .apply(|a, &b, &c, &d| {
                    *a = b - c - d;
                });
            Y = Y + Z.map(|x| x * mu);

            err = frobenius_norm(&Z) / normX;

            if err < tol {
               break
            }
        }

        Self {
            L,
            S,
        }
    }
}


/// Shrinkage Operator for Singular Values
pub fn Do (
    tau : f64,
    X : &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array2<f64> {
    let (u, s, v) = X.svd(true, true).unwrap();
    let s = Array2::from_diag(&s);
    let So = So(tau, &s);
    let u = u.unwrap() as Array2<f64>;
    let v = v.unwrap() as Array2<f64>;
    let mut r = u.dot(&So);
    r = r.dot(&v.t());
    r
}

/// Shrinkage Operator
pub fn So (
    tau : f64,
    X : &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array2<f64> {
    let max = 0.0;
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
                    *a = b / b;
                });
    out
}

pub fn frobenius_norm (
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> f64 {
    x.map(|x| x.powi(2)).sum().sqrt()
}

pub struct PCA {
    components: Array2<f64>,
    mean : Array1<f64>,
}

impl PCA {

    pub fn fit (
        x : &ArrayBase<impl Data<Elem = f64>, Ix2>,
        n_components : f64,
    ) -> Self {

        let (_n, _m) = x.dim();

        //calculate the array of columnar means
        let mean = x.mean_axis(Axis(0)).unwrap();

        // subtract means from X
        let h:Array2<f64> = Array2::ones((_n, _m));
        let temp:Array2<f64> = h * &mean;
        let b:Array2<f64> = x - &temp;

        // compute SVD
        let (u, sigma, v) = b.svd(true, true).unwrap();

        let mut u = u.unwrap() as Array2<f64>;
        let mut v = v.unwrap() as Array2<f64>;
        let temp = (b.nrows() - 1) as f64;
        let explained_variance = sigma.map(|x| x.powi(2)).map(|x|  x / temp );
        let total_var = explained_variance.sum();
        let explained_variance_ratio = explained_variance.map(|x| x / total_var);
        let mut singular_values = sigma.clone();

        let mut ratio_cumsum = explained_variance_ratio.clone();
        ratio_cumsum.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

        //println!("{}", ratio_cumsum);

        // find the number of components to represent the variance ration passed in as n_components_ratio
        if n_components <= 1.0 {
            let mut covered_components = 0;
            loop {
                if ratio_cumsum[covered_components] > n_components {
                    covered_components += 1;
                    break;
                } else {
                    covered_components += 1;
                }
            }
            v = v.slice(s![.., ..covered_components]).to_owned();
            u = u.slice(s![.., ..covered_components]).to_owned();
            singular_values = singular_values.slice(s![..covered_components]).to_owned();
        }
        else {
            v = v.slice(s![.., ..n_components as i32]).to_owned();
            u = u.slice(s![.., ..n_components as i32]).to_owned();
            singular_values = singular_values.slice(s![..n_components as i32]).to_owned();
        }

        let components = u * singular_values;

        Self {
            components,
            mean,
        }
    }

    pub fn mean(&self) -> &Array1<f64> {
        &self.mean
    }

    pub fn components(&self) -> &Array2<f64> {
        &self.components
    }
}

/*
The laplace expansion is used to recursively find the determinant of
a square matrix.
*/
pub fn laplace_expansion(
    x: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> f64 {

    if x.is_square() == false {
        panic!("Cannot perform Laplace Expansion on non-square matrices");
    }

    if x.nrows() == 2 {
        let out = (x[[0,0]] * x[[1,1]]) - (x[[0,1]] * x[[1,0]]);
        return out;
    }

    else {
        let mut out = 0.;
        let mut cofactor:f64 = -1.0;
        let (n, _) = x.dim();
        // todo: better implementation looks for row or column
        // with most 0s to speed up process

        //simple implementation only looks at top row
        let row = x.slice(s![0, ..]);

        let mut j = 0;
        for elem in row.iter() {
            cofactor *= -1.0;
            let tempmat = delete_row_and_column(x, 0, j);
            j += 1;
            out += cofactor * (elem * laplace_expansion(&tempmat));
        }
        return out;
    }
}

pub fn delete_row_and_column(
    x:&ArrayBase<impl Data<Elem = f64>, Ix2>,
    i:i32,
    j:i32,
) -> ArrayBase<impl Data<Elem = f64>, Ix2> {
    let top_left = x.slice(s![..i, ..j]);
    let top_right = x.slice(s![..i, j+1..]);

    let bottom_left = x.slice(s![i+1.., ..j]);
    let bottom_right = x.slice(s![i+1.., j+1..]);

    let top = stack![Axis(1), top_left, top_right];
    let bottom = stack![Axis(1), bottom_left, bottom_right];

    let out = stack![Axis(0), top, bottom];
    out
}

pub fn get_covariance_matrix(
    b: Array2<f64>,
) -> Array2<f64> {
    let _n = &b.nrows();
    // let b_star = get_conjugate_transpose(b);
    let _div = _n - 1;
    let c = b.dot(&b.t()) / _div as f64;
    c
}

/* only necessary for complex numbers
pub fn get_conjugate_transpose(
    a: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> &ArrayBase<impl Data<Elem = f64>, Ix2> {
     let a = a.t();
    // todo: get conjugates for imaginary numbers
    // i.e conjugate of a + bi = a - bi
    a
}
*/

fn main() {
    let x: Array2<f64> = arr2(&[[1., 2., 3.],
                                    [3., 1., 2.],
                                    [2., 3., 1.]]);

    //let (n, m) = x.dim();
    //println!("{}", laplace_expansion(&x));
    //println!("{}", x)
    let model = PCA::fit(&x,3.0);
    //println!("{}", model.mean());
    //println!("{}", model.covariance_matrix());
}