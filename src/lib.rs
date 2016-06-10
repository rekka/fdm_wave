//! Finite difference method for the wave equation with Neumann boundary data.
#![feature(cfg_target_feature)]
extern crate rayon;

#[cfg(test)]
extern crate ndarray;
#[cfg(test)]
extern crate ndarray_rand;
#[cfg(test)]
extern crate rand;

extern crate simd;
use simd::Simd;
#[cfg(target_feature = "avx")]
use simd::x86::avx::f64x4;

use std::cmp::min;

#[repr(packed)]
#[derive(Debug, Copy, Clone)]
struct Unalign<T>(T);

trait UncheckedLoadStore: Simd {
    unsafe fn load_unchecked(array: &[Self::Elem], idx: usize) -> Self;
    unsafe fn store_unchecked(self, array: &[Self::Elem], idx: usize);
}

#[inline]
unsafe fn load_unchecked(array: &[f64], idx: usize) -> f64x4 {
    debug_assert!(idx + 4 <= array.len());
    let data = array.as_ptr().offset(idx as isize);
    let loaded = *(data as *const Unalign<f64x4>);
    loaded.0
}

#[inline]
unsafe fn store_unchecked(x: f64x4, array: &mut [f64], idx: usize) {
    debug_assert!(idx + 4 <= array.len());
    let place = array.as_mut_ptr().offset(idx as isize);
    *(place as *mut Unalign<f64x4>) = Unalign(x);
}

/// Same as `wave_step`, but runs two rows at the same time.
pub fn wave_step_double(u: &[f64], v: &[f64], w: &mut [f64], dim: (usize, usize), mu: f64) {
    let (ny, nx) = dim;
    let n = nx * ny;
    assert_eq!(u.len(), n);
    assert_eq!(v.len(), n);
    assert_eq!(w.len(), n);

    let simd_nx = (nx / 4).saturating_sub(1);

    let co = f64x4::splat(mu);
    let cc = f64x4::splat(2. - 4. * mu);

    // we do two rows at a time (to improve memory access/elem ratio from
    // (4 reads + 1 write) / elem to (6 reads + 2 writes) / elem
    for i in 0..ny / 2 {
        let s0 = nx * (2 * i).saturating_sub(1);
        let s1 = nx * 2 * i;
        let s2 = nx * (2 * i + 1);
        let s3 = nx * min(2 * i + 2, ny - 1);

        if simd_nx > 0 {
            // initialize data
            let mut v1 = f64x4::load(v, s1);
            let mut v2 = f64x4::load(v, s2);
            let mut vp1 = f64x4::splat(v[s1]);
            let mut vp2 = f64x4::splat(v[s2]);
            for j in 0..simd_nx {
                unsafe {
                    let v0 = load_unchecked(v, s0 + j * 4);
                    let vn1 = load_unchecked(v, s1 + j * 4 + 4);
                    let vn2 = load_unchecked(v, s2 + j * 4 + 4);
                    let v3 = load_unchecked(v, s3 + j * 4);
                    let u1 = load_unchecked(u, s1 + j * 4);
                    let u2 = load_unchecked(u, s2 + j * 4);
                    let vl1 =
                        f64x4::new(vp1.extract(3), v1.extract(0), v1.extract(1), v1.extract(2));
                    let vl2 =
                        f64x4::new(vp2.extract(3), v2.extract(0), v2.extract(1), v2.extract(2));
                    let vr1 =
                        f64x4::new(v1.extract(1), v1.extract(2), v1.extract(3), vn1.extract(0));
                    let vr2 =
                        f64x4::new(v2.extract(1), v2.extract(2), v2.extract(3), vn2.extract(0));
                    let w1 = cc * v1 - u1 + co * (vl1 + vr1 + v0 + v2);
                    let w2 = cc * v2 - u2 + co * (vl2 + vr2 + v1 + v3);
                    store_unchecked(w1, w, s1 + j * 4);
                    store_unchecked(w2, w, s2 + j * 4);
                    vp1 = v1;
                    vp2 = v2;
                    v1 = vn1;
                    v2 = vn2;
                }
            }
        }
        for j in simd_nx * 4..nx {
            let jl = j.saturating_sub(1);
            let jr = std::cmp::min(j + 1, nx - 1);
            w[s1 + j] = 2. * v[s1 + j] - u[s1 + j] +
                        mu * (v[s1 + jl] + v[s1 + jr] + v[s0 + j] + v[s2 + j] - 4. * v[s1 + j]);
            w[s2 + j] = 2. * v[s2 + j] - u[s2 + j] +
                        mu * (v[s2 + jl] + v[s2 + jr] + v[s1 + j] + v[s3 + j] - 4. * v[s2 + j]);
        }
    }
    // if ny is odd, we have to do the last row separately
    if ny % 2 != 0 {
        let s0 = nx * (ny - 1).saturating_sub(1);
        let s1 = nx * (ny - 1);
        let s2 = s1;

        if simd_nx > 0 {
            // initialize data
            let mut v1 = f64x4::load(v, s1);
            let mut vp1 = f64x4::splat(v[s1]);
            for j in 0..simd_nx {
                unsafe {
                    let v0 = load_unchecked(v, s0 + j * 4);
                    let vn1 = load_unchecked(v, s1 + j * 4 + 4);
                    let u1 = load_unchecked(u, s1 + j * 4);
                    let vl1 =
                        f64x4::new(vp1.extract(3), v1.extract(0), v1.extract(1), v1.extract(2));
                    let vr1 =
                        f64x4::new(v1.extract(1), v1.extract(2), v1.extract(3), vn1.extract(0));
                    let w1 = cc * v1 - u1 + co * (vl1 + vr1 + v0 + v1);
                    store_unchecked(w1, w, s1 + j * 4);
                    vp1 = v1;
                    v1 = vn1;
                }
            }
        }
        for j in simd_nx * 4..nx {
            let jl = j.saturating_sub(1);
            let jr = std::cmp::min(j + 1, nx - 1);
            w[s1 + j] = 2. * v[s1 + j] - u[s1 + j] +
                        mu * (v[s1 + jl] + v[s1 + jr] + v[s0 + j] + v[s2 + j] - 4. * v[s1 + j]);
        }
    }
}

fn wave_step_sub(u: &[f64],
                 v: &[f64],
                 w: &mut [f64],
                 dim: (usize, usize),
                 rows: (usize, usize),
                 mu: f64) {
    let (ny, nx) = dim;
    let (rs, re) = rows;
    let n = nx * ny;
    debug_assert!(rs < re);
    debug_assert!(re <= ny);
    debug_assert_eq!(u.len(), n);
    debug_assert_eq!(v.len(), n);
    debug_assert_eq!(w.len(), (re - rs) * nx);

    let simd_nx = (nx / 4).saturating_sub(1);

    let co = f64x4::splat(mu);
    let cc = f64x4::splat(2. - 4. * mu);
    let w_offset = rs * nx;

    for i in rs..re {
        let s0 = nx * i.saturating_sub(1);
        let s1 = nx * i;
        let s2 = nx * min(i + 1, ny - 1);

        if simd_nx > 0 {
            // initialize data
            let mut v1 = f64x4::load(v, s1);
            let mut vp1 = f64x4::splat(v[s1]);
            for j in 0..simd_nx {
                unsafe {
                    let v0 = load_unchecked(v, s0 + j * 4);
                    let vn1 = load_unchecked(v, s1 + j * 4 + 4);
                    let v2 = load_unchecked(v, s2 + j * 4);
                    let u1 = load_unchecked(u, s1 + j * 4);
                    let vl1 =
                        f64x4::new(vp1.extract(3), v1.extract(0), v1.extract(1), v1.extract(2));
                    let vr1 =
                        f64x4::new(v1.extract(1), v1.extract(2), v1.extract(3), vn1.extract(0));
                    let w1 = cc * v1 - u1 + co * (vl1 + vr1 + v0 + v2);
                    store_unchecked(w1, w, s1 + j * 4 - w_offset);
                    vp1 = v1;
                    v1 = vn1;
                }
            }
        }
        for j in simd_nx * 4..nx {
            let jl = j.saturating_sub(1);
            let jr = std::cmp::min(j + 1, nx - 1);
            w[s1 + j - w_offset] =
                2. * v[s1 + j] - u[s1 + j] +
                mu * (v[s1 + jl] + v[s1 + jr] + v[s0 + j] + v[s2 + j] - 4. * v[s1 + j]);
        }
    }
}

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
/// Does not increase the performance :(.
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
    use rand::distributions::Range;
    use ndarray_rand::RandomExt;
    use super::*;
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

    fn test_dim(ny: usize, nx: usize) {
        let dim = (ny, nx);
        let mu = 2.32;
        let u = Array::random(dim, Range::new(0., 1.));
        let v = Array::random(dim, Range::new(0., 1.));
        // let u = Array::from_elem(dim, 1.);
        // let v = Array::from_elem(dim, 1.);
        let mut w_ref = Array::zeros(dim);

        wave_step_reference(&u, &v, &mut w_ref, mu);

        let mut w = Array::zeros(dim);
        wave_step(u.as_slice().unwrap(),
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

        let mut w = Array::zeros(dim);
        wave_step_double(u.as_slice().unwrap(),
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

        let mut w = Array::zeros(dim);
        wave_step_parallel(u.as_slice().unwrap(),
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

    #[test]
    fn test_equal() {
        test_dim(1, 1);
        test_dim(3, 3);
        test_dim(3, 20);
        test_dim(20, 3);
        test_dim(20, 20);
        test_dim(2048, 2048);
    }
}
