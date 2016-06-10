#[repr(packed)]
#[derive(Debug, Copy, Clone)]
pub struct Unalign<T>(pub T);

/// Unaligned loads and stores, unchecked in the release build.
pub trait UncheckedLoadStore {
    type Elem;
    unsafe fn load_unchecked(array: &[Self::Elem], idx: usize) -> Self;
    unsafe fn store_unchecked(self, array: &mut [Self::Elem], idx: usize);
}

impl UncheckedLoadStore for f64 {
    type Elem = f64;
    #[inline]
    unsafe fn load_unchecked(array: &[Self::Elem], idx: usize) -> Self {
        debug_assert!(idx < array.len());
        *array.get_unchecked(idx)
    }

    #[inline]
    unsafe fn store_unchecked(self, array: &mut [Self::Elem], idx: usize) {
        debug_assert!(idx < array.len());
        *array.get_unchecked_mut(idx) = self;
    }
}

macro_rules! impl_unchecked_load_store {
    ($($simd:ident $elem:ident $n:expr),+,) => {
        $(
        impl UncheckedLoadStore for $simd {
            type Elem = $elem;
            #[inline]
            unsafe fn load_unchecked(array: &[Self::Elem], idx: usize) -> Self {
                debug_assert!(idx + $n <= array.len());
                let data = array.as_ptr().offset(idx as isize);
                let loaded = *(data as *const Unalign<Self>);
                loaded.0
            }

            #[inline]
            unsafe fn store_unchecked(self, array: &mut [Self::Elem], idx: usize) {
                debug_assert!(idx + $n <= array.len());
                let place = array.as_mut_ptr().offset(idx as isize);
                *(place as *mut Unalign<Self>) = Unalign(self);
            }
        }
        )+
    }
}

/// Rotate values by one lane left or right, filling the empty value by the value from `filler`.
pub trait RotateOne {
    fn rotate_one_left(self, filler: Self) -> Self;
    fn rotate_one_right(self, filler: Self) -> Self;
}

impl RotateOne for f64 {
    #[inline]
    fn rotate_one_left(self, filler: Self) -> Self {
        filler
    }
    #[inline]
    fn rotate_one_right(self, filler: Self) -> Self {
        filler
    }
}

pub trait Splat<T> {
    fn splat(x: T) -> Self;
}

impl Splat<f64> for f64 {
    fn splat(x: f64) -> f64 {
        x
    }
}

pub trait SimdWidth {
    fn width() -> usize;
}

macro_rules! impl_simd_width {
    ($($simd:ident $n:expr),+,) => {
        $(
        impl SimdWidth for $simd {
            #[inline]
            fn width() -> usize {
                $n
            }
        }
        )+
    }
}

impl_simd_width!{
    f64 1,
}
