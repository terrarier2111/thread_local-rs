// Copyright 2017 Amanieu d'Antras
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Per-object thread-local storage
//!
//! This library provides the `ThreadLocal` type which allows a separate copy of
//! an object to be used for each thread. This allows for per-object
//! thread-local storage, unlike the standard library's `thread_local!` macro
//! which only allows static thread-local storage.
//!
//! Per-thread objects are not destroyed when a thread exits. Instead, objects
//! are only destroyed when the `ThreadLocal` containing them is destroyed.
//!
//! You can also iterate over the thread-local values of all thread in a
//! `ThreadLocal` object using the `iter_mut` and `into_iter` methods. This can
//! only be done if you have mutable access to the `ThreadLocal` object, which
//! guarantees that you are the only thread currently accessing it.
//!
//! Note that since thread IDs are recycled when a thread exits, it is possible
//! for one thread to retrieve the object of another thread. Since this can only
//! occur after a thread has exited this does not lead to any race conditions.
//!
//! # Examples
//!
//! Basic usage of `ThreadLocal`:
//!
//! ```rust
//! use thread_local::ThreadLocal;
//! let tls: ThreadLocal<u32> = ThreadLocal::new();
//! assert_eq!(tls.get(), None);
//! assert_eq!(tls.get_or(|| 5), &5);
//! assert_eq!(tls.get(), Some(&5));
//! ```
//!
//! Combining thread-local values into a single result:
//!
//! ```rust
//! use thread_local::ThreadLocal;
//! use std::sync::Arc;
//! use std::cell::Cell;
//! use std::thread;
//!
//! let tls = Arc::new(ThreadLocal::new());
//!
//! // Create a bunch of threads to do stuff
//! for _ in 0..5 {
//!     let tls2 = tls.clone();
//!     thread::spawn(move || {
//!         // Increment a counter to count some event...
//!         let cell = tls2.get_or(|| Cell::new(0));
//!         cell.set(cell.get() + 1);
//!     }).join().unwrap();
//! }
//!
//! // Once all threads are done, collect the counter values and return the
//! // sum of all thread-local counter values.
//! let tls = Arc::try_unwrap(tls).unwrap();
//! let total = tls.into_iter().fold(0, |x, y| x + y.get());
//! assert_eq!(total, 5);
//! ```

#![warn(missing_docs)]
#![allow(clippy::mutex_atomic)]
#![cfg_attr(feature = "nightly", feature(thread_local))]

mod cached;
mod thread_id;
mod unreachable;

#[allow(deprecated)]
pub use cached::{CachedIntoIter, CachedIterMut, CachedThreadLocal};

#[cfg(feature = "nightly")]
use crate::thread_id::ThreadHolder;
use std::cell::UnsafeCell;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::iter::FusedIterator;
use std::mem;
use std::mem::{MaybeUninit, transmute};
use std::ops::{Deref, DerefMut};
use std::panic::UnwindSafe;
use std::ptr;
use std::ptr::{addr_of, read};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, AtomicUsize, Ordering};
use thread_id::Thread;
use unreachable::UncheckedResultExt;

// Use usize::BITS once it has stabilized and the MSRV has been bumped.
#[cfg(target_pointer_width = "16")]
const POINTER_WIDTH: u8 = 16;
#[cfg(target_pointer_width = "32")]
const POINTER_WIDTH: u8 = 32;
#[cfg(target_pointer_width = "64")]
const POINTER_WIDTH: u8 = 64;

/// The total number of buckets stored in each thread local.
const BUCKETS: usize = (POINTER_WIDTH + 1) as usize;

/// Thread-local variable wrapper
///
/// See the [module-level documentation](index.html) for more.
pub struct ThreadLocal<T: Send> {
    /// The buckets in the thread local. The nth bucket contains `2^(n-1)`
    /// elements. Each bucket is lazily allocated.
    buckets: [AtomicPtr<Option<NoNiche<T>>>; BUCKETS],

    /// The number of values in the thread local. This can be less than the real number of values,
    /// but is never more.
    values: AtomicUsize,
}

struct Entry<T> {
    // we use the option in here to indicate whether the value is present or not
    // and its discriminant should be updated atomically
    value: UnsafeCell<Option<NoNiche<T>>>,
}

impl<T> Entry<T> {

    const ORDINAL_BIT: usize = {
        let tmp = Some(MaybeUninit::zeroed());
        let ptr = addr_of!(tmp);
        let mut global_bit = 0;
        let mut result = usize::MAX;
        while global_bit < mem::size_of::<T>() * 8 {
            let byte = global_bit / 8;
            let local_bit = global_bit % 8;
            if (unsafe { ((ptr as usize + byte) as *const u8).read() } & local_bit) != 0 {
                result = global_bit;
            }
        }
        result
    };

    #[inline]
    fn empty() -> Self {
        Entry {
            value: UnsafeCell::new(None),
        }
    }

    #[inline]
    fn update_present(&self, val: bool, ordering: Ordering) {
        let byte_ptr = unsafe { ((((self as *const Self) as usize) + Self::ORDINAL_BIT / 8) as *const AtomicU8).as_ref().unwrap_unchecked() };
        let bit = (Self::ORDINAL_BIT % 8) as u8;
        if val {
            byte_ptr.fetch_or(1 << bit, ordering);
        } else {
            byte_ptr.fetch_and(!(1 << bit), ordering);
        }
    }

    fn is_present(&self, ordering: Ordering) -> bool {
        let byte_ptr = unsafe { ((((self as *const Self) as usize) + Self::ORDINAL_BIT / 8) as *const AtomicU8).as_ref().unwrap_unchecked() };
        let bit = (Self::ORDINAL_BIT % 8) as u8;
        byte_ptr.load(ordering) & bit != 0
    }

}

// ThreadLocal is always Sync, even if T isn't
unsafe impl<T: Send> Sync for ThreadLocal<T> {}

impl<T: Send> Default for ThreadLocal<T> {
    fn default() -> ThreadLocal<T> {
        ThreadLocal::new()
    }
}

impl<T: Send> Drop for ThreadLocal<T> {
    fn drop(&mut self) {
        let mut bucket_size = 1;

        // Free each non-null bucket
        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            let bucket_ptr = *bucket.get_mut();

            let this_bucket_size = bucket_size;
            if i != 0 {
                bucket_size <<= 1;
            }

            if bucket_ptr.is_null() {
                continue;
            }

            unsafe { deallocate_bucket(bucket_ptr, this_bucket_size) };
        }
    }
}

impl<T: Send> ThreadLocal<T> {
    /// Creates a new empty `ThreadLocal`.
    pub fn new() -> ThreadLocal<T> {
        Self::with_capacity(2)
    }

    /// Creates a new `ThreadLocal` with an initial capacity. If less than the capacity threads
    /// access the thread local it will never reallocate. The capacity may be rounded up to the
    /// nearest power of two.
    pub fn with_capacity(capacity: usize) -> ThreadLocal<T> {
        let allocated_buckets = capacity
            .checked_sub(1)
            .map(|c| usize::from(POINTER_WIDTH) - (c.leading_zeros() as usize) + 1)
            .unwrap_or(0);

        let mut buckets = [ptr::null_mut(); BUCKETS];
        let mut bucket_size = 1;
        for (i, bucket) in buckets[..allocated_buckets].iter_mut().enumerate() {
            *bucket = allocate_bucket::<T>(bucket_size);

            if i != 0 {
                bucket_size <<= 1;
            }
        }

        ThreadLocal {
            // Safety: AtomicPtr has the same representation as a pointer and arrays have the same
            // representation as a sequence of their inner type.
            buckets: unsafe { mem::transmute(buckets) },
            values: AtomicUsize::new(0),
        }
    }

    /// Returns the element for the current thread, if it exists.
    #[cfg(feature = "nightly")]
    pub fn get(&self) -> &Option<NoNiche<T>> {
        match unsafe { thread_id::THREAD_HOLDER.as_ref() } {
            None => &None,
            Some(x) => self.get_inner(&x.0),
        }
    }

    /// Returns the element for the current thread, if it exists.
    #[cfg(not(feature = "nightly"))]
    pub fn get(&self) -> Option<&T> {
        let thread = thread_id::get();
        self.get_inner(&thread)
    }

    /// Returns the element for the current thread, or creates it if it doesn't
    /// exist.
    pub fn get_or<F>(&self, create: F) -> &T
    where
        F: FnOnce() -> T,
    {
        unsafe {
            self.get_or_try(|| Ok::<T, ()>(create()))
                .unchecked_unwrap_ok()
        }
    }

    /// Returns the element for the current thread, or creates it if it doesn't
    /// exist. If `create` fails, that error is returned and no element is
    /// added.
    #[cfg(feature = "nightly")]
    pub fn get_or_try<F, E>(&self, create: F) -> Result<&T, E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        if let Some(thread) = unsafe { thread_id::THREAD_HOLDER.as_ref() } {
            if let Some(inner) = self.get_inner(&thread.0) {
                return Ok(inner);
            }
        }
        let thread = ThreadHolder::new();
        Ok(self.insert(thread, create()?))
    }

    /// Returns the element for the current thread, or creates it if it doesn't
    /// exist. If `create` fails, that error is returned and no element is
    /// added.
    #[cfg(not(feature = "nightly"))]
    pub fn get_or_try<F, E>(&self, create: F) -> Result<&T, E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        let thread = thread_id::get();
        match self.get_inner(&thread) {
            Some(x) => Ok(x),
            None => Ok(self.insert(thread, create()?)),
        }
    }

    fn get_inner(&self, thread: &Thread) -> &Option<NoNiche<T>> {
        let bucket_ptr =
            unsafe { self.buckets.get_unchecked(thread.bucket) }.load(Ordering::Acquire);
        if bucket_ptr.is_null() {
            return &None;
        }
        unsafe {
            let entry = &*bucket_ptr.add(thread.index);
            // Read without atomic operations as only this thread can set the value.
            entry
        }
    }

    #[cold]
    #[cfg(feature = "nightly")]
    fn insert(&self, thread: ThreadHolder, data: T) -> &T {
        unsafe {
            thread_id::THREAD_HOLDER = Some(thread);
        }
        let thread = &unsafe { thread_id::THREAD_HOLDER.as_ref().unwrap_unchecked() }.0;

        self.insert_inner(thread, data)
    }

    #[cold]
    #[cfg(not(feature = "nightly"))]
    fn insert(&self, thread: Thread, data: T) -> &T {
        self.insert_inner(&thread, data)
    }

    fn insert_inner(&self, thread: &Thread, data: T) -> &T {
        let bucket_atomic_ptr = unsafe { self.buckets.get_unchecked(thread.bucket) };
        let bucket_ptr: *const _ = bucket_atomic_ptr.load(Ordering::Acquire);

        // If the bucket doesn't already exist, we need to allocate it
        let bucket_ptr = if bucket_ptr.is_null() {
            let new_bucket = allocate_bucket(thread.bucket_size);

            match bucket_atomic_ptr.compare_exchange(
                ptr::null_mut(),
                new_bucket,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => new_bucket,
                // If the bucket value changed (from null), that means
                // another thread stored a new bucket before we could,
                // and we can free our bucket and use that one instead
                Err(bucket_ptr) => {
                    unsafe { deallocate_bucket(new_bucket, thread.bucket_size) }
                    bucket_ptr
                }
            }
        } else {
            bucket_ptr
        };

        // Insert the new element into the bucket
        let entry = unsafe { &*bucket_ptr.add(thread.index) };
        let value_ptr = entry.value.get();
        unsafe { value_ptr.write(Some(MaybeUninit::new(data))) };
        entry.update_present(true, Ordering::Release);

        self.values.fetch_add(1, Ordering::Release);

        unsafe { value_ptr.as_ref().unwrap_unchecked().as_ref().unwrap_unchecked() } // FIXME: turn this into a wrapper type!
    }

    /// Returns an iterator over the local values of all threads in unspecified
    /// order.
    ///
    /// This call can be done safely, as `T` is required to implement [`Sync`].
    pub fn iter(&self) -> Iter<'_, T>
    where
        T: Sync,
    {
        Iter {
            thread_local: self,
            raw: RawIter::new(),
        }
    }

    /// Returns a mutable iterator over the local values of all threads in
    /// unspecified order.
    ///
    /// Since this call borrows the `ThreadLocal` mutably, this operation can
    /// be done safely---the mutable borrow statically guarantees no other
    /// threads are currently accessing their associated values.
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            thread_local: self,
            raw: RawIter::new(),
        }
    }

    /// Removes all thread-specific values from the `ThreadLocal`, effectively
    /// resetting it to its original state.
    ///
    /// Since this call borrows the `ThreadLocal` mutably, this operation can
    /// be done safely---the mutable borrow statically guarantees no other
    /// threads are currently accessing their associated values.
    pub fn clear(&mut self) {
        *self = ThreadLocal::new();
    }
}

pub struct NoNiche<T>(MaybeUninit<T>);

impl<T> NoNiche<T> {

    #[inline]
    fn new(val: T) -> Self {
        Self(MaybeUninit::new(val))
    }

    #[inline]
    unsafe fn new_uninit() -> Self {
        Self(MaybeUninit::uninit())
    }

    #[inline]
    unsafe fn new_zeroed() -> Self {
        Self(MaybeUninit::zeroed())
    }

    #[inline]
    fn write(&mut self, val: T) {
        self.0.write(val);
    }

}

impl<T> Deref for NoNiche<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.assume_init_ref() }
    }
}

impl<T> AsRef<T> for NoNiche<T> {
    fn as_ref(&self) -> &T {
        unsafe { self.0.assume_init_ref() }
    }
}

impl<T> DerefMut for NoNiche<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.0.assume_init_mut() }
    }
}

impl<T> AsMut<T> for NoNiche<T> {
    fn as_mut(&mut self) -> &mut T {
        unsafe { self.0.assume_init_mut() }
    }
}

impl<T: Debug> Debug for NoNiche<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        unsafe { self.0.assume_init_ref() }.fmt(f)
    }
}

impl<T: Display> Display for NoNiche<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        unsafe { self.0.assume_init_ref() }.fmt(f)
    }
}

impl<T: Send> IntoIterator for ThreadLocal<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        IntoIter {
            thread_local: self,
            raw: RawIter::new(),
        }
    }
}

impl<'a, T: Send + Sync> IntoIterator for &'a ThreadLocal<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: Send> IntoIterator for &'a mut ThreadLocal<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

impl<T: Send + Default> ThreadLocal<T> {
    /// Returns the element for the current thread, or creates a default one if
    /// it doesn't exist.
    pub fn get_or_default(&self) -> &T {
        self.get_or(Default::default)
    }
}

impl<T: Send + fmt::Debug> fmt::Debug for ThreadLocal<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ThreadLocal {{ local_data: {:?} }}", self.get())
    }
}

impl<T: Send + UnwindSafe> UnwindSafe for ThreadLocal<T> {}

#[derive(Debug)]
struct RawIter {
    yielded: usize,
    bucket: usize,
    bucket_size: usize,
    index: usize,
}
impl RawIter {
    #[inline]
    fn new() -> Self {
        Self {
            yielded: 0,
            bucket: 0,
            bucket_size: 1,
            index: 0,
        }
    }

    fn next<'a, T: Send + Sync>(&mut self, thread_local: &'a ThreadLocal<T>) -> Option<&'a T> {
        while self.bucket < BUCKETS {
            let bucket = unsafe { thread_local.buckets.get_unchecked(self.bucket) };
            let bucket = bucket.load(Ordering::Acquire);

            if !bucket.is_null() {
                while self.index < self.bucket_size {
                    let entry = unsafe { &*bucket.add(self.index) };
                    self.index += 1;
                    if entry.present.load(Ordering::Acquire) {
                        self.yielded += 1;
                        return Some(unsafe { &*(&*entry.value.get()).as_ptr() });
                    }
                }
            }

            self.next_bucket();
        }
        None
    }
    fn next_mut<'a, T: Send>(
        &mut self,
        thread_local: &'a mut ThreadLocal<T>,
    ) -> Option<&'a mut Entry<T>> {
        if *thread_local.values.get_mut() == self.yielded {
            return None;
        }

        loop {
            let bucket = unsafe { thread_local.buckets.get_unchecked_mut(self.bucket) };
            let bucket = *bucket.get_mut();

            if !bucket.is_null() {
                while self.index < self.bucket_size {
                    let entry = unsafe { &mut *bucket.add(self.index) };
                    self.index += 1;
                    if *entry.present.get_mut() {
                        self.yielded += 1;
                        return Some(entry);
                    }
                }
            }

            self.next_bucket();
        }
    }

    #[inline]
    fn next_bucket(&mut self) {
        if self.bucket != 0 {
            self.bucket_size <<= 1;
        }
        self.bucket += 1;
        self.index = 0;
    }

    fn size_hint<T: Send>(&self, thread_local: &ThreadLocal<T>) -> (usize, Option<usize>) {
        let total = thread_local.values.load(Ordering::Acquire);
        (total - self.yielded, None)
    }
    fn size_hint_frozen<T: Send>(&self, thread_local: &ThreadLocal<T>) -> (usize, Option<usize>) {
        let total = unsafe { *(&thread_local.values as *const AtomicUsize as *const usize) };
        let remaining = total - self.yielded;
        (remaining, Some(remaining))
    }
}

/// Iterator over the contents of a `ThreadLocal`.
#[derive(Debug)]
pub struct Iter<'a, T: Send + Sync> {
    thread_local: &'a ThreadLocal<T>,
    raw: RawIter,
}

impl<'a, T: Send + Sync> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.raw.next(self.thread_local)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.raw.size_hint(self.thread_local)
    }
}
impl<T: Send + Sync> FusedIterator for Iter<'_, T> {}

/// Mutable iterator over the contents of a `ThreadLocal`.
pub struct IterMut<'a, T: Send> {
    thread_local: &'a mut ThreadLocal<T>,
    raw: RawIter,
}

impl<'a, T: Send> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<&'a mut T> {
        self.raw
            .next_mut(self.thread_local)
            .map(|entry| unsafe { &mut *(&mut *entry.value.get()).as_mut_ptr() })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.raw.size_hint_frozen(self.thread_local)
    }
}

impl<T: Send> ExactSizeIterator for IterMut<'_, T> {}
impl<T: Send> FusedIterator for IterMut<'_, T> {}

// Manual impl so we don't call Debug on the ThreadLocal, as doing so would create a reference to
// this thread's value that potentially aliases with a mutable reference we have given out.
impl<'a, T: Send + fmt::Debug> fmt::Debug for IterMut<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IterMut").field("raw", &self.raw).finish()
    }
}

/// An iterator that moves out of a `ThreadLocal`.
#[derive(Debug)]
pub struct IntoIter<T: Send> {
    thread_local: ThreadLocal<T>,
    raw: RawIter,
}

impl<T: Send> Iterator for IntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.raw.next_mut(&mut self.thread_local).map(|entry| {
            unsafe { mem::replace(&mut *entry.value.get(), None).assume_init() }
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.raw.size_hint_frozen(&self.thread_local)
    }
}

impl<T: Send> ExactSizeIterator for IntoIter<T> {}
impl<T: Send> FusedIterator for IntoIter<T> {}

fn allocate_bucket<T>(size: usize) -> *mut Option<NoNiche<T>> {
    Box::into_raw(
        (0..size)
            .map(|_| None::<T>)
            .collect(),
    ) as *mut _
}

unsafe fn deallocate_bucket<T>(bucket: *mut Option<NoNiche<T>>, size: usize) {
    let _ = Box::from_raw(std::slice::from_raw_parts_mut(bucket, size));
}

#[cfg(test)]
mod tests {
    use super::ThreadLocal;
    use std::cell::RefCell;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::Arc;
    use std::thread;

    fn make_create() -> Arc<dyn Fn() -> usize + Send + Sync> {
        let count = AtomicUsize::new(0);
        Arc::new(move || count.fetch_add(1, Relaxed))
    }

    #[test]
    fn same_thread() {
        let create = make_create();
        let mut tls = ThreadLocal::new();
        assert_eq!(None, tls.get());
        assert_eq!("ThreadLocal { local_data: None }", format!("{:?}", &tls));
        assert_eq!(0, *tls.get_or(|| create()));
        assert_eq!(Some(&0), tls.get());
        assert_eq!(0, *tls.get_or(|| create()));
        assert_eq!(Some(&0), tls.get());
        assert_eq!(0, *tls.get_or(|| create()));
        assert_eq!(Some(&0), tls.get());
        assert_eq!("ThreadLocal { local_data: Some(0) }", format!("{:?}", &tls));
        tls.clear();
        assert_eq!(None, tls.get());
    }

    #[test]
    fn different_thread() {
        let create = make_create();
        let tls = Arc::new(ThreadLocal::new());
        assert_eq!(None, tls.get());
        assert_eq!(0, *tls.get_or(|| create()));
        assert_eq!(Some(&0), tls.get());

        let tls2 = tls.clone();
        let create2 = create.clone();
        thread::spawn(move || {
            assert_eq!(None, tls2.get());
            assert_eq!(1, *tls2.get_or(|| create2()));
            assert_eq!(Some(&1), tls2.get());
        })
        .join()
        .unwrap();

        assert_eq!(Some(&0), tls.get());
        assert_eq!(0, *tls.get_or(|| create()));
    }

    #[test]
    fn iter() {
        let tls = Arc::new(ThreadLocal::new());
        tls.get_or(|| Box::new(1));

        let tls2 = tls.clone();
        thread::spawn(move || {
            tls2.get_or(|| Box::new(2));
            let tls3 = tls2.clone();
            thread::spawn(move || {
                tls3.get_or(|| Box::new(3));
            })
            .join()
            .unwrap();
            drop(tls2);
        })
        .join()
        .unwrap();

        let mut tls = Arc::try_unwrap(tls).unwrap();

        let mut v = tls.iter().map(|x| **x).collect::<Vec<i32>>();
        v.sort_unstable();
        assert_eq!(vec![1, 2, 3], v);

        let mut v = tls.iter_mut().map(|x| **x).collect::<Vec<i32>>();
        v.sort_unstable();
        assert_eq!(vec![1, 2, 3], v);

        let mut v = tls.into_iter().map(|x| *x).collect::<Vec<i32>>();
        v.sort_unstable();
        assert_eq!(vec![1, 2, 3], v);
    }

    #[test]
    fn test_drop() {
        let local = ThreadLocal::new();
        struct Dropped(Arc<AtomicUsize>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.fetch_add(1, Relaxed);
            }
        }

        let dropped = Arc::new(AtomicUsize::new(0));
        local.get_or(|| Dropped(dropped.clone()));
        assert_eq!(dropped.load(Relaxed), 0);
        drop(local);
        assert_eq!(dropped.load(Relaxed), 1);
    }

    #[test]
    fn is_sync() {
        fn foo<T: Sync>() {}
        foo::<ThreadLocal<String>>();
        foo::<ThreadLocal<RefCell<String>>>();
    }
}
