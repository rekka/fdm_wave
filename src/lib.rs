//! Finite difference method for the wave equation with Neumann boundary data.
#![cfg_attr(feature = "simd", feature(cfg_target_feature))]
#[cfg(feature = "rayon")]
extern crate rayon;

#[cfg(test)]
extern crate ndarray;
#[cfg(test)]
extern crate ndarray_rand;
#[cfg(test)]
extern crate rand;

#[cfg(feature = "simd")]
extern crate simd;

#[macro_use]
mod simd_utils;

macro_rules! wave_step_sub_impl {
        ($func: ident, $simd: ident, $ty: ident) => {
pub fn wave_step_sub(u: &[$ty],
                 v: &[$ty],
                 w: &mut [$ty],
                 dim: (usize, usize),
                 rows: (usize, usize),
                 mu: $ty) {
    let (ny, nx) = dim;
    let (rs, re) = rows;
    let n = nx * ny;
    // sanity checks
    debug_assert!(rs < re);
    debug_assert!(re <= ny);
    debug_assert_eq!(u.len(), n);
    debug_assert_eq!(v.len(), n);
    debug_assert_eq!(w.len(), (re - rs) * nx);

    let simd_width = $simd::width();
    let simd_nx = (nx / simd_width).saturating_sub(1);
    let w_offset = rs * nx;

    for i in rs..re {
        let s0 = nx * i.saturating_sub(1);
        let s1 = nx * i;
        let s2 = nx * ::std::cmp::min(i + 1, ny - 1);

        unsafe {
            if simd_nx > 0 {
                // initialize data
                let mut v1 = $simd::load_unchecked(v, s1);
                let mut vp1 = $simd::splat($ty::load_unchecked(v, s1));
                for j in 0..simd_nx {
                    let v0 = $simd::load_unchecked(v, s0 + j * simd_width);
                    let vn1 = $simd::load_unchecked(v, s1 + (j + 1) * simd_width);
                    let v2 = $simd::load_unchecked(v, s2 + j * simd_width);
                    let u1 = $simd::load_unchecked(u, s1 + j * simd_width);
                    let vl1 = v1.rotate_one_right(vp1);
                    let vr1 = v1.rotate_one_left(vn1);
                    let w1 = $simd::splat(2. - 4. * mu) * v1 - u1 +
                             $simd::splat(mu) * (vl1 + vr1 + v0 + v2);
                    w1.store_unchecked(w, s1 + j * simd_width - w_offset);
                    vp1 = v1;
                    v1 = vn1;
                }
            }
            // run the rest using single data instructions
            let js = simd_nx * simd_width;
            let mut v1 = $ty::load_unchecked(v, s1 + js);
            let mut vp1 = $ty::load_unchecked(v, s1 + js.saturating_sub(1));
            for j in js..nx {
                let jr = ::std::cmp::min(j + 1, nx - 1);
                let v0 = $ty::load_unchecked(v, s0 + j);
                let vn1 = $ty::load_unchecked(v, s1 + jr);
                let v2 = $ty::load_unchecked(v, s2 + j);
                let u1 = $ty::load_unchecked(u, s1 + j);
                let vl1 = vp1;
                let vr1 = vn1;
                let w1 = (2. - 4. * mu) * v1 - u1 + mu * (vl1 + vr1 + v0 + v2);
                w1.store_unchecked(w, s1 + j - w_offset);
                vp1 = v1;
                v1 = vn1;
            }
        }
    }
}
        }
}

#[cfg(not(feature = "simd"))]
mod scalar {
    use simd_utils::*;
    wave_step_sub_impl!(wave_step_sub, f64, f64);
}

#[cfg(feature = "simd")]
mod vector;

#[cfg(not(feature = "simd"))]
use scalar::wave_step_sub;
#[cfg(feature = "simd")]
use vector::wave_step_sub;

/// Performs one step of the finite difference scheme for the wave equation with zero Neumann
/// boundary condition.
///
/// The discretization is the standard central 2nd order difference in both space and time.
///
/// - `u`: value at step `n - 1`
/// - `v`: value at step `n`
/// - `w`: computed value at step `n + 1`
/// - `dim`: format `(ny, nx)` so that memory layout for `u`, `v` and `w` is `[[f64 ; nx]; ny]`
/// - `mu`: τ²/h², where τ is the time step and h is the space step
pub fn wave_step(u: &[f64], v: &[f64], w: &mut [f64], dim: (usize, usize), mu: f64) {
    let (ny, nx) = dim;
    let n = nx * ny;
    assert_eq!(u.len(), n);
    assert_eq!(v.len(), n);
    assert_eq!(w.len(), n);

    wave_step_sub(u, v, w, dim, (0, dim.0), mu);
}

/// Same as `wave_step`, attempting to run in parallel.
///
/// Does not increase performance :(.
#[cfg(feature = "rayon")]
pub fn wave_step_parallel(u: &[f64], v: &[f64], w: &mut [f64], dim: (usize, usize), mu: f64) {
    let (ny, nx) = dim;
    let n = nx * ny;
    assert_eq!(u.len(), n);
    assert_eq!(v.len(), n);
    assert_eq!(w.len(), n);

    fn inner(u: &[f64],
             v: &[f64],
             w: &mut [f64],
             dim: (usize, usize),
             rows: (usize, usize),
             mu: f64) {
        let elems_per_thread = 200 * 1024;
        let (row_start, row_end) = rows;
        if row_end - row_start > 1 && w.len() > elems_per_thread {
            let (_, nx) = dim;
            let mid = (row_start + row_end) / 2;
            let (left, right) = w.split_at_mut((mid - row_start) * nx);
            rayon::join(|| inner(u, v, left, dim, (row_start, mid), mu),
                        || inner(u, v, right, dim, (mid, row_end), mu));
        } else {
            wave_step_sub(u, v, w, dim, rows, mu);
        }
    };

    inner(u, v, w, dim, (0, ny), mu);
}

#[cfg(test)]
mod test {
    pub use super::*;
    use std::cmp::min;

    use ndarray::{Array, Ix};
    type Array2d = Array<f64, (Ix, Ix)>;


    /// Reference implementation.
    /// mu = τ²/h²
    fn wave_step_reference(w: &Array2d, u: &Array2d, v: &mut Array2d, mu: f64) {
        let (ny, nx) = u.dim();
        for j in 0..ny {
            for i in 0..nx {
                let uc = u[(j, i)];
                let ul = u[(j, i.saturating_sub(1))];
                let ur = u[(j, min(i + 1, nx - 1))];
                let ut = u[(j.saturating_sub(1), i)];
                let ub = u[(min(j + 1, ny - 1), i)];
                v[(j, i)] = 2. * uc - w[(j, i)] + mu * (ul + ur + ut + ub - 4. * uc);
            }
        }
    }

    macro_rules! test_stepper {
        ($modname:ident, $stepper:ident, $(($name:ident, $ny:expr, $nx:expr)),+,) => {

            mod $modname {
                use super::wave_step_reference;
                use ndarray::Array;
                use ndarray_rand::RandomExt;
                use rand::distributions::Range;
                use super::*;
                $(
                #[test]
                fn $name() {
                    let dim = ($ny, $nx);
                    let mu = 2.32;
                    let u = Array::random(dim, Range::new(0., 1.));
                    let v = Array::random(dim, Range::new(0., 1.));
                    let mut w_ref = Array::zeros(dim);

                    wave_step_reference(&u, &v, &mut w_ref, mu);

                    let mut w = Array::zeros(dim);
                    $stepper(u.as_slice().unwrap(),
                              v.as_slice().unwrap(),
                              w.as_slice_mut().unwrap(),
                              dim,
                              mu);

                    let d = &w - &w_ref;
                    let err = (&d * &d).scalar_sum();
                    if err > 1e-10 {
                        println!("{}\n", w);
                        println!("{}\n", w_ref);
                        println!("{}",
                                 d.mapv(|x| {
                            if x.abs() < 1e-14 {
                                0.
                            } else {
                                x
                            }
                        }));
                        panic!("Error too big: {}", err);
                    }
                }
                )+
            }
        }
    }

    test_stepper!{
        seq,
        wave_step,
        (s1x1, 1, 1),
        (s3x3, 3, 3),
        (s3x100, 3, 100),
        (s100x3, 100, 3),
        (s99x97, 99, 97),
    }

    #[cfg(feature = "rayon")]
    test_stepper!{
        rayon,
        wave_step_parallel,
        (s1x1, 1, 1),
        (s3x3, 3, 3),
        (s3x100, 3, 100),
        (s100x3, 100, 3),
        (s99x97, 99, 97),
    }
}
