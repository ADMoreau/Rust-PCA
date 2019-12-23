///This file is the rust implementation of
///Q. Zheng, Y. Wang, and P. Heng. Online robust image alignment
///via subspace learning from gradient orientations. IEEE International
///Conference on Computer Vision, ICCV 2017, 2017
#![allow(non_snake_case, non_camel_case_types)]

extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_vision;
extern crate openblas_src;

use ndarray::{Array1, Array, Array2, Array3, arr2, Axis};
use ndarray_vision::processing::sobel::*;
use std::cmp::max;


///ADMM solver for linearized convex optimization
pub fn ADMM (
    U : Array3<f64>,    // Orthogonal basis
    x : Array2<f64>,    // warped and vectorized gradient orientation
    J : Array3<f64>,    // Jacobian matrix for x
    rho : f64,          //penalty constant
    tol : f64,          //tolerance
    iter : i32,         //max iteration
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

    return (Array1::zeros(1), Array2::zeros((1,1)), 0.0, 0.0);
}

pub fn soft_threshold (
    X: Array2<f64>,
    mu: f64,
) -> Array2<f64> {
    max(x - (1.0/mu), 0) - max((-1 * x) - (1.0/mu), 0)
}

/// based on python implementation -> https://github.com/welch/rasl/blob/master/rasl/rasl.py
/// and original matlab code
pub fn RASL (
    /// array of images as ndarray(h, v), assumes grayscale
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
    let (images, rows, columns) = Image.dim();
    let mut T = Array3::zeros((images, 3, 3));
    for i in 0..images {
        // default frame is 5
        T[i] = inset(index_axis(Axis(0), i), 5);
    }



    return (Array3::zeros((0,0,0)), Array3::zeros((0,0,0)), Array3::zeros((0,0,0)))
}

///basic implementation
/// incorporate a framing transform such that when imtransform is
/// applied to an image of the given shape, the image will first
/// be inset to the specified interior boundary. This is
/// convenient for zooming in on an area of interest prior to
/// performing alignment (and an inset of at least 2 is necessary
/// to avoid the boundary erosion of a bicubic transform)
pub fn inset(
    X : Array2<f64>,
    frame : i32,
) -> Array2<f64> {
    let (_n, _m) = X.dim();
    let mut bounds = array![[frame as f64, frame as f64], [-1.0 * frame as f64 + 1.0, -1.0 * frame as f64 + 1.0]];
    let scale = array![1.0, 1.0];
    // original -> parameters_to_projective_matrix, default ttype = 'affine'
    let T = array![[scale[1], 0.0, bounds[[0, 1]]], [0.0, scale[0], bounds[[0, 0]]], [0.0, 0.0, 1.0]];
    T
}

/// image gradient vectors under tform's framing transform
pub fn frame_gradient (
    tform : Array2<f64>, // current transform
    image : Array2<f64>, // original image
) -> (
    /// returns image gradients in the current frame
    Array2<f64>, // imagex
    Array2<f64>, // imagey
) {

}

/// apply sobel filter to the image to approximate the gradient
pub fn image_gradient (
    image : Array2<f64>, // image as an ndarray
    mut horv: String, // direction of gradient
) -> (
    Array2<f64> // directional gradient magnitude at each pixel
) {
    if horv = 'h'.to_string() {
        let axis = 1;
    }
    else {
        let axis = 0;
    }

    let grad = image.apply_sobel();

}



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


