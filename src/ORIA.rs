///This file is the rust implementation of
///Q. Zheng, Y. Wang, and P. Heng. Online robust image alignment
///via subspace learning from gradient orientations. IEEE International
///Conference on Computer Vision, ICCV 2017, 2017
#![allow(non_snake_case, non_camel_case_types)]

extern crate ndarray;
extern crate ndarray_linalg;
extern crate openblas_src;

use ndarray::{Array, Array2, Array3};
use std::cmp::max;


///ADMM solver for linearized convex optimization
pub fn ADMM (
    U: Array3<f64>, // Orthogonal basis
    x: Array2<f64>, // warped and vectorized gradient orientation
    J: Array3<f64>, // Jacobian matrix for x
    rho: f64, //penalty constant
    tol: f64, //tolerance
    iter: i32, //max iteration
) -> (
    /// weight vector w in R^k
    Array1<f64>,
    /// the sparse outliers e in R^d
    Array2<f64>,
    /// locally linearized parameter tao in R^p
    f64,
    /// dual vector y in R^d
    f64) {

    let mut w = 0.0;
    let mut tao = 0.0;
    let mut y = 0.0;
    let mut mu = 1.0;
    let mut e = Array2::zeros(x.dim());
    for i in 0..iter {
        //update sparse e
        e = soft_threshold(x + (J*tao) - (U*w) + (y/mu), mu);
        // update weight w
        w = U.t()
    }

    return (Array1::zeros(1), Array2::zeros(1,1), 0.0, 0.0);
}

pub fn soft_threshold (
    X: Array2<f64>,
    mu: f64,
) -> Array2<f64> {
    max((x - (1.0/mu), 0) - max((-1 * x) - (1.0/mu), 0)
}

/// based on python implementation -> https://github.com/welch/rasl/blob/master/rasl/rasl.py
pub fn RASL (
    /// array of images as ndarray(h, v)
    Image : Array3<f64>,
    /// maximum iterations to convergence
    maxiter : i32,
    /// stop iterating when objective change is less than this
    stop_delta : f64,
    /// if true, normalize transformed images and their gradients
    normalize : bool,
) -> (
    /// L : aligned low-rank images
    Array3<f64>,
    /// S : aligned sparse error images
    Array3<f64>,
    /// T : final transforms. Each will include initT's inset frame and the aligning paramv
    Array3<f64>,
) {

}

///configure a framing transformation.
///incorporate a framing transform such that when imtransform is
///applied to an image of the given shape, the image will first
///be inset to the specified interior boundary. This is
///convenient for zooming in on an area of interest prior to
///performing alignment (and an inset of at least 2 is necessary
///to avoid the boundary erosion of a bicubic transform)
///Parameters
///----------
///X : array2
///    input image
///frame : real or real(2) or (real(2), real(2))
///    pixel-width of inset boundary (single number), cropped image
///    size (tuple, centered) or boundary points (minimum and
///    maximum included points) as pixel offsets into the image,
///    ranging [0, max-1]. Negative values are subtracted from
///    the dimension size, as with python array indexing.
///    NOTE: frame tuples are organized [y,x] (ie, row, col) like all
///    other image indexing.
///crop : bool
///    if True, the output shape is set to the implied inset frame.
///    if False, the inset image will be zoomed to fill the input shape
pub fn inset(
    X : Array2<f64>,
    frame : i32,
    crop : bool, // default true
) -> Array2<f64> {
    let (_n, _m) = X.dim();
    let mut bounds = array!(frame);
    // assuming that frames is a single value, original code did not
    bounds = array!(((bounds, bounds), (bounds - 1, bounds - 1)));
    bounds = bounds.map(|x| )
}





