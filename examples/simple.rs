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
