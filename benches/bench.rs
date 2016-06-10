#![feature(test)]
extern crate test;
// extern crate ndarray;
extern crate fdm_wave;

mod bench {
    // mod ndarray {
    //     use test::{black_box, Bencher};
    //     use ndarray::Array;
    //     use fdm_wave::*;
    //
    //     fn bench_ndarray(b: &mut Bencher, ny: usize, nx: usize) {
    //         let dim = (ny, nx);
    //         let u = Array::from_elem(dim, 1.);
    //         let v = black_box(Array::from_elem(dim, 1.));
    //         let mut w = Array::from_elem(dim, 1.);
    //         let mu = 2.32;
    //
    //         b.iter(|| {
    //             wave_step_ndarray(&u, &v, &mut w, mu);
    //         });
    //     }
    //
    //     #[bench]
    //     fn s128(b: &mut Bencher) {
    //         bench_ndarray(b, 128, 128);
    //     }
    //
    //     #[bench]
    //     fn s512(b: &mut Bencher) {
    //         bench_ndarray(b, 512, 512);
    //     }
    //
    //     #[bench]
    //     fn s2048(b: &mut Bencher) {
    //         bench_ndarray(b, 2048, 2048);
    //     }
    // }

    macro_rules! bench_step {
        ($modname:ident, $func: ident, $(($name:ident, $ny:expr, $nx:expr))+) => {
            mod $modname {
                use test::{black_box, Bencher};
                use fdm_wave::*;
                $(
                #[bench]
                fn $name(b: &mut Bencher)
                {
                    let dim = ($ny, $nx);
                    let l = dim.0 * dim.1;
                    let u = vec![1.; l];
                    let v = vec![2.; l];
                    let mut w = black_box(vec![1.; l]);
                    let mu = 2.32;

                    b.iter(|| {
                        $func(&u, &v, &mut w, dim, mu);
                    });
                }
                )+
            }
        }
    }

    bench_step!{simd_single, wave_step,
                (s128, 128, 128)
                (s512, 512, 512)
                (s2048, 2048, 2048)
                }

    // bench_step!{simd_double, wave_step_double,
    //             (s128, 128, 128)
    //             (s512, 512, 512)
    //             (s2048, 2048, 2048)
    //             }

    bench_step!{simd_parallel, wave_step_parallel,
                (s128, 128, 128)
                (s512, 512, 512)
                (s2048, 2048, 2048)
                }

}
