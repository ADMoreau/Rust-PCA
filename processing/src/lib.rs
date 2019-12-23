extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_rand;
extern crate serde;


#[allow(clippy::new_ret_no_self)]
mod PCA;
mod rPCA;

pub use PCA::*;
pub use rPCA::*;