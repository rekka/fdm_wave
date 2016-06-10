
#[cfg(target_feature = "avx")]
pub use self::avx::wave_step_sub;
#[cfg(not(target_feature = "avx"))]
pub use self::sse2::wave_step_sub;

#[cfg(target_feature = "avx")]
pub mod avx {
    use simd::x86::avx::f64x4;
    use simd_utils::*;

    impl_unchecked_load_store!{ f64x4 f64 4, }
    impl_simd_width!{ f64x4 4, }

    impl RotateOne for f64x4 {
        #[inline]
        fn rotate_one_left(self, filler: Self) -> Self {
            Self::new(self.extract(1),
                      self.extract(2),
                      self.extract(3),
                      filler.extract(0))
        }
        #[inline]
        fn rotate_one_right(self, filler: Self) -> Self {
            Self::new(filler.extract(3),
                      self.extract(0),
                      self.extract(1),
                      self.extract(2))
        }
    }
    wave_step_sub_impl!(wave_step_sub, f64x4, f64);

    /// Same as `wave_step`, but runs two rows at the same time.
    #[allow(dead_code)]
    pub fn wave_step_double(u: &[f64], v: &[f64], w: &mut [f64], dim: (usize, usize), mu: f64) {
        use std::cmp::min;
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
                        let v0 = f64x4::load_unchecked(v, s0 + j * 4);
                        let vn1 = f64x4::load_unchecked(v, s1 + j * 4 + 4);
                        let vn2 = f64x4::load_unchecked(v, s2 + j * 4 + 4);
                        let v3 = f64x4::load_unchecked(v, s3 + j * 4);
                        let u1 = f64x4::load_unchecked(u, s1 + j * 4);
                        let u2 = f64x4::load_unchecked(u, s2 + j * 4);
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
                        w1.store_unchecked(w, s1 + j * 4);
                        w2.store_unchecked(w, s2 + j * 4);
                        vp1 = v1;
                        vp2 = v2;
                        v1 = vn1;
                        v2 = vn2;
                    }
                }
            }
            for j in simd_nx * 4..nx {
                let jl = j.saturating_sub(1);
                let jr = min(j + 1, nx - 1);
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
                        let v0 = f64x4::load_unchecked(v, s0 + j * 4);
                        let vn1 = f64x4::load_unchecked(v, s1 + j * 4 + 4);
                        let u1 = f64x4::load_unchecked(u, s1 + j * 4);
                        let vl1 =
                            f64x4::new(vp1.extract(3), v1.extract(0), v1.extract(1), v1.extract(2));
                        let vr1 =
                            f64x4::new(v1.extract(1), v1.extract(2), v1.extract(3), vn1.extract(0));
                        let w1 = cc * v1 - u1 + co * (vl1 + vr1 + v0 + v1);
                        w1.store_unchecked(w, s1 + j * 4);
                        vp1 = v1;
                        v1 = vn1;
                    }
                }
            }
            for j in simd_nx * 4..nx {
                let jl = j.saturating_sub(1);
                let jr = min(j + 1, nx - 1);
                w[s1 + j] = 2. * v[s1 + j] - u[s1 + j] +
                            mu * (v[s1 + jl] + v[s1 + jr] + v[s0 + j] + v[s2 + j] - 4. * v[s1 + j]);
            }
        }
    }
}

#[cfg(not(target_feature = "avx"))]
pub mod sse2 {
    use simd_utils::*;
    use simd::x86::sse2::f64x2;

    impl RotateOne for f64x2 {
        #[inline]
        fn rotate_one_left(self, filler: Self) -> Self {
            Self::new(self.extract(1), filler.extract(0))
        }
        #[inline]
        fn rotate_one_right(self, filler: Self) -> Self {
            Self::new(filler.extract(1), self.extract(0))
        }
    }

    impl_unchecked_load_store!{ f64x2 f64 2, }
    impl_simd_width!{ f64x2 2, }

    wave_step_sub_impl!(wave_step_sub, f64x2, f64);
}
