# fdm_wave

Implementation of a finite difference stepper for the wave equation with
SIMD optimization.

Currently only the following are supported:

- methods

    - central-in-time, central-in-space finite difference discretization

- boundary data

    - zero Neumann boundary data

## Simple example

The following code advances the solution by performing one step of the
finite difference method for the wave equation with zero Neumann
boundary data, with given `dt` and `dx`.  `v` is the current solution,
and `u` is the solution at the previous time step, and `w` is the new
solution. The dimension of the mesh is `(2, 2) = (rows, columns)`.
Arrays are stored in the row-major order (C order).

```rust
extern crate fdm_wave;

fn main() {
    let u = vec![1., 2., 3., 4.];
    let v = vec![3., 2., 1., 4.];
    let mut w = vec![0.; 4];

    let dt = 1.;
    let dx = 2.;

    fdm_wave::wave_step(&u, &v, &mut w, (2, 2), dt * dt / dx * dx);

    println!("{:?}", w);
}
```

## Usage

This crate works with Cargo. Add it to `Cargo.toml` as

```toml
[dependencies.fdm_wave]
git = "https://github.com/rekka/fdm_wave.git"
```

To use the SIMD-optimized routines, you need to use a nightly compiler and
add the `simd` feature:

```toml
[dependencies.fdm_wave]
git = "https://github.com/rekka/fdm_wave.git"
features = ["simd"]
```

Supports SSE2 and AVX. Make sure that you are passing correct flags to
Cargo/rustc (`target-cpu=native`; possibly using
[`RUSTFLAGS`](https://github.com/rust-lang/cargo/pull/2241)).

The SIMD versions are mainly useful for small problem sizes. The
algorithm is limited by memory bandwidth if it spills out of caches.

## License

MIT license: see the `LICENSE` file for details.
