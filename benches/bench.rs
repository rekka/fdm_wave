#![feature(test)]
extern crate test;
extern crate ndarray;
extern crate fdm_wave;

use test::Bencher;

use ndarray::Array;

use fdm_wave::*;

fn bench_ndarray(b: &mut Bencher, ny: usize, nx: usize) {
    let dim = (ny, nx);
    let u = Array::from_elem(dim, 1.);
    let v = test::black_box(Array::from_elem(dim, 1.));
    let mut w = Array::from_elem(dim, 1.);
    let mu = 2.32;

    b.iter(|| {
        wave_step_ndarray(&u, &v, &mut w, mu);
    });
}

fn bench_simd(b: &mut Bencher, ny: usize, nx: usize) {
    let dim = (ny, nx);
    let l = dim.0 * dim.1;
    let u = vec![1.; l];
    let v = vec![1.; l];
    let mut w = test::black_box(vec![1.; l]);
    let mu = 2.32;

    b.iter(|| {
        wave_step(&u, &v, &mut w, dim, mu);
    });
}

#[bench]
fn bench_ndarray_128(b: &mut Bencher) {
    bench_ndarray(b, 128, 128);
}

#[bench]
fn bench_ndarray_512(b: &mut Bencher) {
    bench_ndarray(b, 512, 512);
}

#[bench]
fn bench_ndarray_2048(b: &mut Bencher) {
    bench_ndarray(b, 2048, 2048);
}

#[bench]
fn bench_simd_128(b: &mut Bencher) {
    bench_simd(b, 128, 128);
}

#[bench]
fn bench_simd_512(b: &mut Bencher) {
    bench_simd(b, 512, 512);
}

#[bench]
fn bench_simd_2048(b: &mut Bencher) {
    bench_simd(b, 2048, 2048);
}
