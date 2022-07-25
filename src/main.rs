//! In memory data copy
//! # Findings:
//!
//! * clone_from_slice, copy_from_slice, optimised avx copy are all on par with memcpy
//! * default zero-based initialisation does not happen at vector declaration time
//! * memory pages are not activated and initialised until one of their elements is touched
//! * copying into an uninitialised buffer is slower than into an initialised one
//! * touching one element every <page size> elements before copying data between buffers makes the
//!   first copy operation much faster
//! * page-locking memory with mlock for 1GiB buffers first is slower than initialising the buffers
//!   explicitly and the copy operation is not faster
//! * for a 1GiB buffer no difference between aligned, unaligned and page-locked memory and almost no
//!   difference between debug and release builds
//! * same results with C and Rust
//! * multithreaded copy is faster but dependent on number of memory channels and number of cores

use std::ffi::*;
use std::time::Instant;

//-----------------------------------------------------------------------------
// Need to move pointer to buffer element across threads
#[derive(Copy, Clone)]
struct Movable<T>(*const T);

impl<T> Movable<T> {
    fn get(&self) -> Option<*const T> {
        if self.0.is_null() {
            return None;
        }
        Some(self.0)
    }
}
#[derive(Copy, Clone)]
struct MovableMut<T>(*mut T);

impl<T> MovableMut<T> {
    fn get(&self) -> Option<*mut T> {
        if self.0.is_null() {
            return None;
        }
        Some(self.0)
    }
}

unsafe impl<T> Send for Movable<T> {}
unsafe impl<T> Send for MovableMut<T> {}
//-----------------------------------------------------------------------------
fn par_cp<T: 'static + Copy>(src: &[T], dst: &mut [T], n: usize) {
    assert!(src.len() % n == 0);
    let mut th = vec![];
    let cs = src.len() / n;
    for i in 0..n {
        unsafe {
            let idx = (n * i) as isize;
            let s = Movable(src.as_ptr().offset(idx));
            let d = MovableMut(dst.as_mut_ptr().offset(idx));
            // no difference in performance using memmcpy
            th.push(std::thread::spawn(move || {
                let src = std::slice::from_raw_parts(s.get().unwrap(), cs);
                let dst = std::slice::from_raw_parts_mut(d.get().unwrap(), cs);
                dst.copy_from_slice(src);
            }))
        }
    }
    for t in th {
        t.join().unwrap();
    }
}
//-----------------------------------------------------------------------------
fn par_cp2<T: 'static + Copy>(src: &[T], dst: &mut [T], n: usize) {
    assert!(src.len() % n == 0);
    let mut th = vec![];
    let cs = src.len() / n;
    for i in 0..n {
        unsafe {
            let idx = (n * i) as isize;
            let s = Movable(src.as_ptr().offset(idx));
            let d = MovableMut(dst.as_mut_ptr().offset(idx));
            th.push(std::thread::spawn(move || {
                memcpy(
                    d.get().unwrap() as *mut c_void,
                    s.get().unwrap() as *const c_void,
                    cs,
                );
            }))
        }
    }
    for t in th {
        t.join().unwrap();
    }
}
//-----------------------------------------------------------------------------
type size_t = usize;
extern "C" {
    fn memcpy(dest: *mut c_void, src: *const c_void, n: size_t) -> *mut c_void;
}
fn cp(src: &[u8], dst: &mut [u8]) {
    unsafe {
        memcpy(
            dst.as_mut_ptr() as *mut c_void,
            src.as_ptr() as *const c_void,
            dst.len(),
        );
    }
}
//-----------------------------------------------------------------------------
fn cp2(src: &[u8], dst: &mut [u8]) {
    dst.copy_from_slice(src);
}
//-----------------------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
fn cp_avx<T: 'static + Copy>(src: &[T], dst: &mut [T]) {
    let addr = src.as_ptr() as usize;
    if addr % page_size::get() != 0 || addr % 256 != 0 {
        panic!("AVX2 copy requires memory to be aligned to page size and a multiple of 256");
    }
    //let base = 256;
    for i in (0..src.len()).step_by(256) {
        unsafe {
            // Playing with pre-fetching does have minor effects but looks pretty much useless
            // if i < (src.len() - 256) {
            //  std::arch::x86_64::_mm_prefetch(src.as_ptr().offset(i as isize + base) as *const i8, std::arch::x86_64::_MM_HINT_T1);
            //  std::arch::x86_64::_mm_prefetch(src.as_ptr().offset(i as isize + base + 32) as *const i8, std::arch::x86_64::_MM_HINT_T1);
            //  std::arch::x86_64::_mm_prefetch(src.as_ptr().offset(i as isize + base + 64) as *const i8, std::arch::x86_64::_MM_HINT_T1);
            //  std::arch::x86_64::_mm_prefetch(src.as_ptr().offset(i as isize + base + 96) as *const i8, std::arch::x86_64::_MM_HINT_T1);
            //  std::arch::x86_64::_mm_prefetch(src.as_ptr().offset(i as isize + base + 128) as *const i8, std::arch::x86_64::_MM_HINT_T1);
            //  std::arch::x86_64::_mm_prefetch(src.as_ptr().offset(i as isize + base + 160) as *const i8, std::arch::x86_64::_MM_HINT_T1);
            //  std::arch::x86_64::_mm_prefetch(src.as_ptr().offset(i as isize + base + 192) as *const i8, std::arch::x86_64::_MM_HINT_T1);
            //  std::arch::x86_64::_mm_prefetch(src.as_ptr().offset(i as isize + base + 224) as *const i8, std::arch::x86_64::_MM_HINT_T1);
            // }
            let d1 = dst.as_mut_ptr().offset(i as isize) as *mut std::arch::x86_64::__m256i;
            let d2 = dst.as_mut_ptr().offset(i as isize + 32) as *mut std::arch::x86_64::__m256i;
            let d3 = dst.as_mut_ptr().offset(i as isize + 64) as *mut std::arch::x86_64::__m256i;
            let d4 = dst.as_mut_ptr().offset(i as isize + 96) as *mut std::arch::x86_64::__m256i;
            let d5 = dst.as_mut_ptr().offset(i as isize + 128) as *mut std::arch::x86_64::__m256i;
            let d6 = dst.as_mut_ptr().offset(i as isize + 160) as *mut std::arch::x86_64::__m256i;
            let d7 = dst.as_mut_ptr().offset(i as isize + 192) as *mut std::arch::x86_64::__m256i;
            let d8 = dst.as_mut_ptr().offset(i as isize + 224) as *mut std::arch::x86_64::__m256i;
            let s1 = std::arch::x86_64::_mm256_load_si256(
                src.as_ptr().offset(i as isize) as *const std::arch::x86_64::__m256i
            );
            let s2 = std::arch::x86_64::_mm256_load_si256(
                src.as_ptr().offset(i as isize + 32) as *const std::arch::x86_64::__m256i
            );
            let s3 = std::arch::x86_64::_mm256_load_si256(
                src.as_ptr().offset(i as isize + 64) as *const std::arch::x86_64::__m256i
            );
            let s4 = std::arch::x86_64::_mm256_load_si256(
                src.as_ptr().offset(i as isize + 96) as *const std::arch::x86_64::__m256i
            );
            let s5 = std::arch::x86_64::_mm256_load_si256(
                src.as_ptr().offset(i as isize + 128) as *const std::arch::x86_64::__m256i
            );
            let s6 = std::arch::x86_64::_mm256_load_si256(
                src.as_ptr().offset(i as isize + 160) as *const std::arch::x86_64::__m256i
            );
            let s7 = std::arch::x86_64::_mm256_load_si256(
                src.as_ptr().offset(i as isize + 192) as *const std::arch::x86_64::__m256i
            );
            let s8 = std::arch::x86_64::_mm256_load_si256(
                src.as_ptr().offset(i as isize + 224) as *const std::arch::x86_64::__m256i
            );
            std::arch::x86_64::_mm256_store_si256(d1, s1);
            std::arch::x86_64::_mm256_store_si256(d2, s2);
            std::arch::x86_64::_mm256_store_si256(d3, s3);
            std::arch::x86_64::_mm256_store_si256(d4, s4);
            std::arch::x86_64::_mm256_store_si256(d5, s5);
            std::arch::x86_64::_mm256_store_si256(d6, s6);
            std::arch::x86_64::_mm256_store_si256(d7, s7);
            std::arch::x86_64::_mm256_store_si256(d8, s8);
        }
    }
}
//-----------------------------------------------------------------------------
fn main() {
    let size = std::env::args()
        .nth(1)
        .expect("Missing buffer size")
        .parse::<usize>()
        .expect("Wrong number format");
    let num_threads = std::env::args()
        .nth(2)
        .map_or(1, |a| a.parse::<usize>().expect("Wrong number format"));
    let init_value = 42_u8; // 0 results in an order of magnitude higher copy time and ~zero
                            // initialization time when using vec![]
    println!("Initializing...");
    let t = Instant::now();
    // same performance with standard initialisation and page aligned buffers;
    // page aligned buffers are required for AVX2 copy
    //let src = vec![init_value; SIZE_U8];
    //let mut dest = vec![init_value; SIZE_U8];
    let src = page_aligned_vec::<u8>(size, size, Some(init_value));
    let mut dest = page_aligned_vec::<u8>(size, size, Some(init_value));
    let init_time = t.elapsed().as_secs_f64();
    println!("Copying...");
    let e = if num_threads == 1 {
        let t = Instant::now();
        cp2(&src, &mut dest);
        t.elapsed().as_secs_f64()
    } else {
        let t = Instant::now();
        par_cp2(&src, &mut dest, num_threads);
        t.elapsed().as_secs_f64()
    };
    println!("output element at {}: {}", page_size::get(), dest[page_size::get()]);
    // R/W 2 * measured BW
    println!(
        "{:.0} ms, {:.2} GiB/s, init: {:.2} s",
        e * 1000.0,
        2. * (size as f64 / 0x40000000 as f64) / e,
        init_time
    );
}
//-----------------------------------------------------------------------------
fn aligned_vec<T: Copy>(
    size: usize,
    capacity: usize,
    align: usize,
    touch: Option<T>,
) -> Vec<T> {
    unsafe {
        if size == 0 {
            Vec::<T>::new()
        } else {
            let size = size * std::mem::size_of::<T>();
            let capacity = (capacity * std::mem::size_of::<T>()).max(size);

            let layout = std::alloc::Layout::from_size_align_unchecked(size, align);
            let raw_ptr = std::alloc::alloc(layout) as *mut T;
            if let Some(x) = touch {
                let mut v = Vec::from_raw_parts(raw_ptr, size, capacity);
                for i in (0..size).step_by(page_size::get()) {
                    v[i] = x;
                }
                v
            } else {
                //SLOW!
                //nix::sys::mman::mlock(raw_ptr as *const c_void, size).unwrap();
                Vec::from_raw_parts(raw_ptr, size, capacity)
            }
        }
    }
}
//-----------------------------------------------------------------------------
fn page_aligned_vec<T: Copy>(size: usize, capacity: usize, touch: Option<T>) -> Vec<T> {
    aligned_vec::<T>(size, capacity, page_size::get(), touch)
}
