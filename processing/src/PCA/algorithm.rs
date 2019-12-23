#![allow(non_snake_case, non_camel_case_types)]
use ndarray::{ArrayBase, Array2, Array1, Axis, Data, Ix2, arr2, stack, s, Zip};
use ndarray_linalg::{SVD, Trace};
use std::cmp::max;

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
        if n_components < 1.0 {
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
        else if n_components == 1.0 {
            v = v.slice(s![.., ..]).to_owned();
            u = u.slice(s![.., ..]).to_owned();
            singular_values = singular_values.slice(s![..]).to_owned();
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, Array2};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_PCA() {

        // Let's setup a synthetic set of observations, composed of two clusters with known means
        let X: Array2<f64> = Array::random((100, 100), Uniform::new(-100., 100.));

        let model = PCA::fit(&X, 1.0);
    }
}