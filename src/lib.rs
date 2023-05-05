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
//! let tls = Arc::new(ThreadLocal::<Cell<i32>, ()>::new());
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
//! // sum of all thread-local counter values, but because all threads are done
//! // there shouldn't be any entries left.
//! let tls = Arc::try_unwrap(tls).unwrap();
//! let total = tls.into_iter().fold(0, |x, y| x + y.get());
//! assert_eq!(total, 0);
//! ```

#![warn(missing_docs)]
#![allow(clippy::mutex_atomic)]
#![feature(thread_local)]

mod cached;
mod thread_id;
mod unreachable;

#[allow(deprecated)]
pub use cached::{CachedIntoIter, CachedIterMut, CachedThreadLocal};

use crate::thread_id::{EntryData, FreeList, global_tid_manager, Thread, ThreadIdManager};
use crossbeam_utils::Backoff;
use smallvec::{smallvec, SmallVec};
use std::cell::UnsafeCell;
use std::fmt;
use std::iter::FusedIterator;
use std::mem;
use std::mem::{size_of, transmute, MaybeUninit};
use std::panic::UnwindSafe;
use std::ptr;
use std::ptr::{null_mut, NonNull, null};
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, TryLockResult};
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
pub struct ThreadLocal<T: Send, M: Metadata = (), const AUTO_FREE_IDS: bool = true> {
    /// The buckets in the thread local. The nth bucket contains `2^(n-1)`
    /// elements. Each bucket is lazily allocated.
    buckets: [AtomicPtr<Entry<T, M, AUTO_FREE_IDS>>; BUCKETS],

    alternative_buckets: [AtomicPtr<Entry<T, M, AUTO_FREE_IDS>>; BUCKETS],
    alternative_entry_ids: Box<Mutex<ThreadIdManager>>,
}

const GUARD_UNINIT: usize = 0;
// there is nothing to guard, so the guard can't be used
const GUARD_EMPTY: usize = 1;
// the guard is ready to be used (changed to active/empty)
const GUARD_READY: usize = 2;
// the guard is currently active and anybody wanting to use it has to wait
const GUARD_ACTIVE_INTERNAL: usize = 3;
const GUARD_ACTIVE_EXTERNAL: usize = 4;
const GUARD_FREE_MANUALLY: usize = 5;

#[derive(Copy, Clone)]
pub struct EntryToken<T, M: Metadata, const AUTO_FREE_IDS: bool>(NonNull<Entry<T, M, AUTO_FREE_IDS>>);

impl<T, M: Metadata, const AUTO_FREE_IDS: bool> EntryToken<T, M, AUTO_FREE_IDS> {

    #[inline]
    pub fn value(&self) -> &T {
        unsafe { (&*self.0.as_ref().value.get()).assume_init_ref() }
    }

    #[inline]
    pub fn meta(&self) -> &M {
        unsafe { &self.0.as_ref().meta }
    }

    /// SAFETY: This may only be called after the thread associated with this thread local has finished.
    pub fn destruct(&self) {
        unsafe { self.0.as_ref().free_id(); }
    }

}

// FIXME: should we primarily determine whether an entry is empty via the free_list ptr or the guard value?
struct Entry<T, M: Metadata = (), const AUTO_FREE_IDS: bool = true> {
    /// this will be null if this entry isn't an alternative entry.
    tid_manager: *const Mutex<ThreadIdManager>,
    id: usize,
    guard: AtomicUsize,
    alternative_entry: AtomicPtr<Entry<T, M, AUTO_FREE_IDS>>,
    free_list: AtomicPtr<FreeList>,
    outstanding_refs: AtomicPtr<AtomicUsize>,
    meta: M,
    value: UnsafeCell<MaybeUninit<T>>,
}

impl<T, M: Metadata, const AUTO_FREE_IDS: bool> Entry<T, M, AUTO_FREE_IDS> {
    /// This should only be called when the central `ThreadLocal`
    /// struct gets dropped.
    pub(crate) unsafe fn try_detach_thread<const N: usize>(
        &self,
        cleanup: &mut SmallVec<[*const Entry<T, M, AUTO_FREE_IDS>; N]>,
    ) {
        let alt = self.alternative_entry.load(Ordering::Acquire);

        if let Some(alt) = alt.as_ref() {
            if !alt.free_list.load(Ordering::Acquire).is_null() {
                if !alt.try_detach_thread_locally() {
                    cleanup.push(alt as *const _);
                }
            }
        }

        if !self.try_detach_thread_locally() {
            cleanup.push(self as *const _);
        }
    }

    /// Returns whether the entry has been cleaned up or not
    pub(crate) unsafe fn try_detach_thread_locally(&self) -> bool {
        let guard = self.guard.load(Ordering::Acquire);
        // the entry is either empty or the guard is currently active
        if guard != GUARD_READY {
            return guard != GUARD_ACTIVE_EXTERNAL;
        }
        // the entry is `Ready`, so we know that we can get exclusive access to it
        if let Err(val) = self.guard.compare_exchange(
            GUARD_READY,
            GUARD_ACTIVE_INTERNAL,
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            return val != GUARD_ACTIVE_EXTERNAL;
        }

        let free_list = self.free_list.load(Ordering::Acquire);

        let mut backoff = Backoff::new();
        loop {
            match unsafe { &*free_list }.free_list.try_lock() {
                Ok(mut guard) => {
                    // we got the lock and can now remove our entry from the free list
                    guard.remove(&(self as *const Entry<T, M, AUTO_FREE_IDS> as usize));
                    return true;
                }
                Err(_) => {
                    // check if the thread declared that it was dropping, if so give up.
                    if unsafe { &*free_list }.dropping.load(Ordering::Acquire) {
                        self.guard.store(GUARD_READY, Ordering::Release);
                        return false;
                    }
                    // FIXME: do we have a better way to wait (but be able to see a change in dropping)
                    backoff.snooze();
                }
            }
        }
    }

    pub(crate) unsafe fn cleanup(slf: *const Entry<()>, remaining_cnt: *const AtomicUsize) -> bool {
        let slf = unsafe { &*slf.cast::<Entry<T, M, AUTO_FREE_IDS>>() };
        let mut backoff = Backoff::new();
        while slf
            .guard
            .compare_exchange_weak(
                GUARD_READY,
                GUARD_ACTIVE_EXTERNAL,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_err()
        {
            backoff.snooze();
        }
        // clean up the value as we already know at this point that there is a value present
        let val = unsafe { &mut *slf.value.get() }.as_mut_ptr();
        unsafe {
            ptr::drop_in_place(val);
        }

        if !AUTO_FREE_IDS {
            slf.outstanding_refs.store(remaining_cnt.cast_mut(), Ordering::Release);
        }

        // signal that there is no thread associated with the entry anymore.
        // this also disables the cleanup of this entry in the `normal` entry
        // on destruction of the central struct.
        slf.free_list.store(null_mut(), Ordering::Release);
        slf.guard.store(if AUTO_FREE_IDS { GUARD_EMPTY } else { GUARD_FREE_MANUALLY }, Ordering::Release); // FIXME: is it okay to store GUARD_EMPTY even if there is still an alternative_entry present?
        AUTO_FREE_IDS
    }

    /// SAFETY: This may only be called after the thread associated with this thread local has finished.
    unsafe fn free_id(&self) {
        // check if we are a "main" entry and our thread is finished
        if let Some(outstanding) = unsafe { self.outstanding_refs.load(Ordering::Acquire).as_ref() } {
            if outstanding.fetch_sub(1, Ordering::AcqRel) != 1 {
                // there are outstanding references left, so we can't free the id yet.
                return;
            }
        }
        // the tid_manager is either an `alternative` id manager or the `global` tid manager.
        unsafe { self.tid_manager.as_ref().unwrap_unchecked() }.lock().unwrap().free(self.id);
    }

}

impl<T, M: Metadata, const AUTO_FREE_IDS: bool> Drop for Entry<T, M, AUTO_FREE_IDS> {
    fn drop(&mut self) {
        let guard = *self.guard.get_mut();
        if guard == GUARD_READY || guard == GUARD_ACTIVE_INTERNAL { // FIXME: is `GUARD_ACTIVE_INTERNAL` valid here if an iterator is currently active?
            let val = unsafe { &mut *self.value.get() }.as_mut_ptr();
            unsafe {
                ptr::drop_in_place(val);
            }
        }
    }
}

// ThreadLocal is always Sync, even if T isn't
unsafe impl<T: Send, M: Metadata, const AUTO_FREE_IDS: bool> Sync for ThreadLocal<T, M, AUTO_FREE_IDS> {}

impl<T: Send, M: Metadata, const AUTO_FREE_IDS: bool> Default for ThreadLocal<T, M, AUTO_FREE_IDS> {
    fn default() -> ThreadLocal<T, M, AUTO_FREE_IDS> {
        ThreadLocal::new()
    }
}

impl<T: Send, M: Metadata, const AUTO_FREE_IDS: bool> Drop for ThreadLocal<T, M, AUTO_FREE_IDS> {
    fn drop(&mut self) {
        let mut bucket_size = 1;

        // Free each non-null bucket
        for (i, bucket) in self.buckets.iter().enumerate() {
            let bucket_ptr = bucket.load(Ordering::Relaxed);

            let this_bucket_size = bucket_size;
            if i != 0 {
                bucket_size <<= 1;
            }

            if bucket_ptr.is_null() {
                // we went up high enough to find an empty bucket, we now know that
                // there are no more non-empty buckets.
                break;
            }

            // capture all unfinished entries and iterate over them after trying to
            // detach all threads once in order to ensure dropping is valid
            let mut unfinished = smallvec![];
            for offset in 0..this_bucket_size {
                let entry = unsafe { bucket_ptr.add(offset).as_ref().unwrap_unchecked() };
                unsafe {
                    entry.try_detach_thread::<8>(&mut unfinished);
                }
            }

            for entry in unfinished.into_iter() {
                let entry = unsafe { &*entry };
                let mut backoff = Backoff::new();
                while entry.guard.load(Ordering::Acquire) == GUARD_ACTIVE_EXTERNAL {
                    backoff.snooze();
                }
            }
        }

        bucket_size = 1;

        // free alternative buckets
        for (i, bucket) in self.alternative_buckets.iter().enumerate() {
            let bucket_ptr = bucket.load(Ordering::Relaxed);

            let this_bucket_size = bucket_size;
            if i != 0 {
                bucket_size <<= 1;
            }

            if bucket_ptr.is_null() {
                // we went up high enough to find an empty bucket, we now know that
                // there are no more non-empty buckets.
                break;
            }

            // FIXME: do we even need to check these alternative buckets - as we should already have awaited
            // FIXME: all of them while awaiting the "normal" buckets.
            for offset in 0..this_bucket_size {
                let entry = unsafe { bucket_ptr.add(offset).as_ref().unwrap_unchecked() };
                let mut backoff = Backoff::new();
                while entry.guard.load(Ordering::Acquire) == GUARD_ACTIVE_EXTERNAL {
                    backoff.snooze();
                }
            }
        }

        bucket_size = 1;

        // Free each non-null bucket
        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            let bucket_ptr = *bucket.get_mut();

            let this_bucket_size = bucket_size;
            if i != 0 {
                bucket_size <<= 1;
            }

            if bucket_ptr.is_null() {
                // we went up high enough to find an empty bucket, we now know that
                // there are no more non-empty buckets.
                break;
            }

            unsafe { deallocate_bucket(bucket_ptr, this_bucket_size) };
        }

        bucket_size = 1;

        // free alternative buckets
        for (i, bucket) in self.alternative_buckets.iter_mut().enumerate() {
            let bucket_ptr = *bucket.get_mut();

            let this_bucket_size = bucket_size;
            if i != 0 {
                bucket_size <<= 1;
            }

            if bucket_ptr.is_null() {
                // we went up high enough to find an empty bucket, we now know that
                // there are no more non-empty buckets.
                break;
            }

            unsafe { deallocate_bucket(bucket_ptr, this_bucket_size) };
        }
    }
}

impl<T: Send, M: Metadata, const AUTO_FREE_IDS: bool> ThreadLocal<T, M, AUTO_FREE_IDS> {
    /// Creates a new empty `ThreadLocal`.
    pub fn new() -> ThreadLocal<T, M, AUTO_FREE_IDS> {
        Self::with_capacity(2)
    }

    /// Creates a new `ThreadLocal` with an initial capacity. If less than the capacity threads
    /// access the thread local it will never reallocate. The capacity may be rounded up to the
    /// nearest power of two.
    pub fn with_capacity(capacity: usize) -> ThreadLocal<T, M, AUTO_FREE_IDS> {
        let allocated_buckets = capacity
            .checked_sub(1)
            .map(|c| usize::from(POINTER_WIDTH) - (c.leading_zeros() as usize) + 1)
            .unwrap_or(0);

        let mut buckets = [null_mut(); BUCKETS];
        let mut bucket_size = 1;
        for (i, bucket) in buckets[..allocated_buckets].iter_mut().enumerate() {
            *bucket = allocate_bucket::<false, AUTO_FREE_IDS, T, M>(bucket_size, global_tid_manager(), 0);

            if i != 0 {
                bucket_size <<= 1;
            }
        }

        let tid_manager = Box::new(Mutex::new(ThreadIdManager::new()));

        let mut alternative_buckets = [null_mut(); BUCKETS];
        let mut bucket_size = 1;
        for (i, bucket) in alternative_buckets[..allocated_buckets]
            .iter_mut()
            .enumerate()
        {
            *bucket = allocate_bucket::<true, AUTO_FREE_IDS, T, M>(bucket_size, tid_manager.as_ref() as *const _, i);

            if i != 0 {
                bucket_size <<= 1;
            }
        }

        Self {
            // Safety: AtomicPtr has the same representation as a pointer and arrays have the same
            // representation as a sequence of their inner type.
            buckets: unsafe { transmute(buckets) },
            alternative_buckets: unsafe { transmute(alternative_buckets) },
            alternative_entry_ids: tid_manager,
        }
    }

    /// Returns the element for the current thread, if it exists.
    pub fn get(&self) -> Option<&T> {
        self.get_inner(thread_id::get())
    }

    /// Returns the meta of the element for the current thread, if it exists.
    pub fn get_meta(&self) -> Option<&M> {
        self.get_inner_meta(thread_id::get())
    }

    /// Returns the meta and value of the element for the current thread, if it exists.
    pub fn get_val_and_meta(&self) -> Option<(&T, &M)> {
        self.get_inner_val_and_meta(thread_id::get())
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

    /// Returns the meta and value of the element for the current thread, or creates it if it doesn't
    /// exist.
    pub fn get_val_and_meta_or<F, MF>(&self, create: F, meta: MF) -> (&T, &M)
    where
        F: FnOnce(*const M) -> T,
        MF: FnOnce(&M),
    {
        let thread = thread_id::get();
        if let Some(val) = self.get_inner_val_and_meta(thread) {
            return val;
        }

        self.insert_with_meta(create, meta)
    }

    /// Returns the element for the current thread, or creates it if it doesn't
    /// exist. If `create` fails, that error is returned and no element is
    /// added.
    pub fn get_or_try<F, E>(&self, create: F) -> Result<&T, E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        let thread = thread_id::get();
        if let Some(val) = self.get_inner(thread) {
            return Ok(val);
        }

        Ok(self.insert(create()?))
    }

    fn get_inner(&self, thread: Thread) -> Option<&T> {
        let bucket_ptr =
            unsafe { self.buckets.get_unchecked(thread.bucket) }.load(Ordering::Acquire);
        if bucket_ptr.is_null() {
            return None;
        }
        let entry = unsafe { &*bucket_ptr.add(thread.index) };
        let free_list = entry.free_list.load(Ordering::Acquire);
        // check if the entry is usable (it has an init value and it's not a pseudo-present value)
        if free_list.cast_const() == thread.free_list {
            Some(unsafe { &*(&*entry.value.get()).as_ptr() })
        } else {
            let alt = entry.alternative_entry.load(Ordering::Acquire);
            // check if the entry has an alternative entry (and thus a pseudo-present value)
            if let Some(alt) = unsafe { alt.as_ref() } {
                Some(unsafe { &*(&*alt.value.get()).as_ptr() })
            } else {
                None
            }
        }
    }

    fn get_inner_meta(&self, thread: Thread) -> Option<&M> {
        let bucket_ptr =
            unsafe { self.buckets.get_unchecked(thread.bucket) }.load(Ordering::Acquire);
        if bucket_ptr.is_null() {
            return None;
        }

        let entry = unsafe { &*bucket_ptr.add(thread.index) };
        let free_list = entry.free_list.load(Ordering::Acquire);
        // check if the entry is usable (it has an init value and it's not a pseudo-present value)
        if free_list.cast_const() == thread.free_list {
            Some(&entry.meta)
        } else {
            let alt = entry.alternative_entry.load(Ordering::Acquire);
            // check if the entry has an alternative entry (and thus a pseudo-present value)
            if let Some(alt) = unsafe { alt.as_ref() } {
                Some(&alt.meta)
            } else {
                None
            }
        }
    }

    fn get_inner_val_and_meta(&self, thread: Thread) -> Option<(&T, &M)> {
        let bucket_ptr =
            unsafe { self.buckets.get_unchecked(thread.bucket) }.load(Ordering::Acquire);
        if bucket_ptr.is_null() {
            return None;
        }

        let entry = unsafe { &*bucket_ptr.add(thread.index) };
        let free_list = entry.free_list.load(Ordering::Acquire);
        // check if the entry is usable (it has an init value and it's not a pseudo-present value)
        if free_list.cast_const() == thread.free_list {
            Some((unsafe { &*(&*entry.value.get()).as_ptr() }, &entry.meta))
        } else {
            let alt = entry.alternative_entry.load(Ordering::Acquire);
            // check if the entry has an alternative entry (and thus a pseudo-present value)
            if let Some(alt) = unsafe { alt.as_ref() } {
                Some((unsafe { &*(&*alt.value.get()).as_ptr() }, &alt.meta))
            } else {
                None
            }
        }
    }

    // FIXME: support insertion in alternative entries!
    #[cold]
    fn insert(&self, data: T) -> &T {
        let thread = thread_id::get();
        let bucket_atomic_ptr = unsafe { self.buckets.get_unchecked(thread.bucket) };
        let bucket_ptr: *const _ = bucket_atomic_ptr.load(Ordering::Acquire);

        // If the bucket doesn't already exist, we need to allocate it
        let bucket_ptr = if bucket_ptr.is_null() {
            let new_bucket = allocate_bucket::<false, AUTO_FREE_IDS, T, M>(thread.bucket_size(), global_tid_manager(), 0);

            match bucket_atomic_ptr.compare_exchange(
                null_mut(),
                new_bucket,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => new_bucket,
                // If the bucket value changed (from null), that means
                // another thread stored a new bucket before we could,
                // and we can free our bucket and use that one instead
                Err(bucket_ptr) => {
                    unsafe { deallocate_bucket(new_bucket, thread.bucket_size()) }
                    bucket_ptr
                }
            }
        } else {
            bucket_ptr
        };

        // Insert the new element into the bucket
        let mut entry = unsafe { &*bucket_ptr.add(thread.index) };

        if entry.guard.load(Ordering::Acquire) == GUARD_FREE_MANUALLY {
            // FIXME: do we have to support traversing alternative entries recursively?
            let alt = self.acquire_alternative_entry();
            entry.alternative_entry.store(alt.cast_mut(), Ordering::Release);
            entry = unsafe { &*alt };
        }

        let value_ptr = entry.value.get();
        unsafe { value_ptr.write(MaybeUninit::new(data)) };
        if size_of::<M>() > 0 {
            entry.meta.set_default();
        }
        let cleanup_fn = Entry::<T, M, AUTO_FREE_IDS>::cleanup;
        unsafe { thread.free_list.as_ref().unwrap_unchecked() }
            .free_list
            .lock()
            .unwrap()
            .insert(
                entry as *const Entry<T, M, AUTO_FREE_IDS> as usize,
                EntryData {
                    drop_fn: unsafe { transmute(cleanup_fn as *const ()) },
                },
            );
        entry
            .free_list
            .store(thread.free_list.cast_mut(), Ordering::Release);
        entry.guard.store(GUARD_READY, Ordering::Release);

        unsafe { &*(&*value_ptr).as_ptr() }
    }

    // FIXME: support insertion in alternative entries!
    #[cold]
    fn insert_with_meta<F: FnOnce(*const M) -> T, FM: FnOnce(&M)>(&self, f: F, fm: FM) -> (&T, &M) {
        let thread = thread_id::get();
        let bucket_atomic_ptr = unsafe { self.buckets.get_unchecked(thread.bucket) };
        let bucket_ptr: *const _ = bucket_atomic_ptr.load(Ordering::Acquire);

        // If the bucket doesn't already exist, we need to allocate it
        let bucket_ptr = if bucket_ptr.is_null() {
            let new_bucket = allocate_bucket::<false, AUTO_FREE_IDS, T, M>(thread.bucket_size(), global_tid_manager(), 0);

            match bucket_atomic_ptr.compare_exchange(
                null_mut(),
                new_bucket,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => new_bucket,
                // If the bucket value changed (from null), that means
                // another thread stored a new bucket before we could,
                // and we can free our bucket and use that one instead
                Err(bucket_ptr) => {
                    unsafe { deallocate_bucket(new_bucket, thread.bucket_size()) }
                    bucket_ptr
                }
            }
        } else {
            bucket_ptr
        };

        // Insert the new element into the bucket
        let entry = unsafe { &*bucket_ptr.add(thread.index) };
        let value_ptr = entry.value.get();
        unsafe { value_ptr.write(MaybeUninit::new(f(&entry.meta as *const M))) };
        if size_of::<M>() > 0 {
            fm(&entry.meta);
        }
        entry
            .free_list
            .store(thread.free_list.cast_mut(), Ordering::Release);
        entry.guard.store(GUARD_READY, Ordering::Release);

        (unsafe { &*(&*value_ptr).as_ptr() }, &entry.meta)
    }

    fn acquire_alternative_entry(&self) -> *const Entry<T, M, AUTO_FREE_IDS> {
        let id = self.alternative_entry_ids.lock().unwrap().alloc();
        let (bucket, bucket_size, index) = thread_id::id_into_parts(id);

        let bucket_atomic_ptr = unsafe { self.buckets.get_unchecked(bucket) };
        let bucket_ptr: *const _ = bucket_atomic_ptr.load(Ordering::Acquire);

        // If the bucket doesn't already exist, we need to allocate it
        let bucket_ptr = if bucket_ptr.is_null() {
            let new_bucket = allocate_bucket::<true, AUTO_FREE_IDS, T, M>(bucket_size, self.alternative_entry_ids.as_ref() as *const _, bucket);

            match bucket_atomic_ptr.compare_exchange(
                null_mut(),
                new_bucket,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => new_bucket,
                // If the bucket value changed (from null), that means
                // another thread stored a new bucket before we could,
                // and we can free our bucket and use that one instead
                Err(bucket_ptr) => {
                    unsafe { deallocate_bucket(new_bucket, bucket_size) }
                    bucket_ptr
                }
            }
        } else {
            bucket_ptr
        };

        // Insert the new element into the bucket
        unsafe { bucket_ptr.add(index) }
    }

    /// Returns an iterator over the local values of all threads in unspecified
    /// order.
    ///
    /// This call can be done safely, as `T` is required to implement [`Sync`].
    pub fn iter(&self) -> Iter<'_, T, M, AUTO_FREE_IDS>
    where
        T: Sync,
    {
        Iter {
            thread_local: self,
            raw: RawIter::new(),
            prev: null(),
        }
    }

    /// Returns a mutable iterator over the local values of all threads in
    /// unspecified order.
    ///
    /// Since this call borrows the `ThreadLocal` mutably, this operation can
    /// be done safely---the mutable borrow statically guarantees no other
    /// threads are currently accessing their associated values.
    pub fn iter_mut(&mut self) -> IterMut<T, M, AUTO_FREE_IDS> {
        IterMut {
            thread_local: self,
            raw: RawIter::new(),
            prev: null(),
        }
    }

    /// Returns an iterator over the local values and metadata of all threads in unspecified
    /// order.
    ///
    /// This call can be done safely, as `T` is required to implement [`Sync`].
    pub fn iter_meta(&self) -> IterMeta<'_, T, M, AUTO_FREE_IDS>
    where
        T: Sync,
    {
        IterMeta {
            thread_local: self,
            raw: RawIter::new(),
            prev: null(),
        }
    }

    /// Returns a mutable iterator over the local values and metadata of all threads in
    /// unspecified order.
    ///
    /// Since this call borrows the `ThreadLocal` mutably, this operation can
    /// be done safely---the mutable borrow statically guarantees no other
    /// threads are currently accessing their associated values.
    pub fn iter_mut_meta(&mut self) -> IterMutMeta<T, M, AUTO_FREE_IDS> {
        IterMutMeta {
            thread_local: self,
            raw: RawIter::new(),
            prev: null(),
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

    /// SAFETY: The provided `val` has to point to an instance of
    /// `T` that was returned by some call to this `ThreadLocal`.
    #[inline]
    pub unsafe fn val_meta_ptr_from_val(val: NonNull<T>) -> ValMetaPtr<T, M, AUTO_FREE_IDS> {
        ValMetaPtr(NonNull::new_unchecked(
            val.as_ptr()
                .cast::<u8>()
                .offset(memoffset::offset_of!(Entry::<T, M, AUTO_FREE_IDS>, value) as isize * -1)
                .cast::<Entry<T, M, AUTO_FREE_IDS>>(),
        ))
    }
}

pub struct ValMetaPtr<T, M: Metadata, const AUTO_FREE_IDS: bool = true>(NonNull<Entry<T, M, AUTO_FREE_IDS>>);

impl<T, M: Metadata, const AUTO_FREE_IDS: bool> ValMetaPtr<T, M, AUTO_FREE_IDS> {
    #[inline]
    pub fn val_ptr(self) -> NonNull<T> {
        unsafe {
            NonNull::new_unchecked(
                self.0
                    .as_ptr()
                    .cast::<u8>()
                    .add(memoffset::offset_of!(Entry::<T, M, AUTO_FREE_IDS>, value))
                    .cast::<T>(),
            )
        }
    }

    #[inline]
    pub fn meta_ptr(self) -> NonNull<M> {
        unsafe {
            NonNull::new_unchecked(
                self.0
                    .as_ptr()
                    .cast::<u8>()
                    .add(memoffset::offset_of!(Entry::<T, M, AUTO_FREE_IDS>, meta))
                    .cast::<M>(),
            )
        }
    }
}

impl<T: Send, M: Metadata, const AUTO_FREE_IDS: bool> IntoIterator for ThreadLocal<T, M, AUTO_FREE_IDS> {
    type Item = T;
    type IntoIter = IntoIter<T, M, AUTO_FREE_IDS>;

    fn into_iter(self) -> IntoIter<T, M, AUTO_FREE_IDS> {
        IntoIter {
            thread_local: self,
            raw: RawIter::new(),
        }
    }
}

impl<'a, T: Send + Sync, M: Metadata, const AUTO_FREE_IDS: bool> IntoIterator for &'a ThreadLocal<T, M, AUTO_FREE_IDS> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T, M, AUTO_FREE_IDS>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: Send, M: Metadata, const AUTO_FREE_IDS: bool> IntoIterator for &'a mut ThreadLocal<T, M, AUTO_FREE_IDS> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T, M, AUTO_FREE_IDS>;

    fn into_iter(self) -> IterMut<'a, T, M, AUTO_FREE_IDS> {
        self.iter_mut()
    }
}

impl<T: Send + Default, M: Metadata, const AUTO_FREE_IDS: bool> ThreadLocal<T, M, AUTO_FREE_IDS> {
    /// Returns the element for the current thread, or creates a default one if
    /// it doesn't exist.
    pub fn get_or_default(&self) -> &T {
        self.get_or(Default::default)
    }
}

impl<T: Send + fmt::Debug, M: Metadata, const AUTO_FREE_IDS: bool> fmt::Debug for ThreadLocal<T, M, AUTO_FREE_IDS> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ThreadLocal {{ local_data: {:?} }}", self.get())
    }
}

impl<T: Send + UnwindSafe, M: Metadata, const AUTO_FREE_IDS: bool> UnwindSafe for ThreadLocal<T, M, AUTO_FREE_IDS> {}

// FIXME: support alternative_entry
#[derive(Debug)]
struct RawIter<const NEW_GUARD: usize> {
    bucket: usize,
    bucket_size: usize,
    index: usize,
}

impl<const NEW_GUARD: usize> RawIter<NEW_GUARD> {
    #[inline]
    fn new() -> Self {
        Self {
            bucket: 0,
            bucket_size: 1,
            index: 0,
        }
    }

    fn next<'a, T: Send + Sync, M: Metadata, const AUTO_FREE_IDS: bool>(
        &mut self,
        thread_local: &'a ThreadLocal<T, M, AUTO_FREE_IDS>,
    ) -> Option<&'a Entry<T, M, AUTO_FREE_IDS>> {
        while self.bucket < BUCKETS {
            let bucket = unsafe { thread_local.buckets.get_unchecked(self.bucket) };
            let bucket = bucket.load(Ordering::Acquire);

            if bucket.is_null() {
                return None;
            }

            while self.index < self.bucket_size {
                let entry = unsafe { &*bucket.add(self.index) };
                self.index += 1;
                match entry.guard.compare_exchange(
                    GUARD_READY,
                    NEW_GUARD,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        return Some(entry);
                    }
                    Err(guard) => {
                        if guard == GUARD_UNINIT {
                            return None;
                        }
                    }
                }
            }

            self.next_bucket();
        }
        None
    }

    fn next_mut<T: Send, M: Metadata, const AUTO_FREE_IDS: bool>(
        &mut self,
        thread_local: &mut ThreadLocal<T, M, AUTO_FREE_IDS>,
    ) -> Option<*mut Entry<T, M, AUTO_FREE_IDS>> {
        while self.bucket < BUCKETS {
            let bucket = unsafe { thread_local.buckets.get_unchecked_mut(self.bucket) };
            let bucket = *bucket.get_mut();

            if bucket.is_null() {
                return None;
            }

            while self.index < self.bucket_size {
                let entry = unsafe { &*bucket.add(self.index) };
                self.index += 1;
                match entry.guard.compare_exchange(
                    GUARD_READY,
                    NEW_GUARD,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        return Some((entry as *const Entry<T, M, AUTO_FREE_IDS>).cast_mut());
                    }
                    Err(guard) => {
                        if guard == GUARD_UNINIT {
                            return None;
                        }
                    }
                }
            }

            self.next_bucket();
        }
        None
    }

    #[inline]
    fn next_bucket(&mut self) {
        if self.bucket != 0 {
            self.bucket_size <<= 1;
        }
        self.bucket += 1;
        self.index = 0;
    }
}

/// Iterator over the contents of a `ThreadLocal`.
#[derive(Debug)]
pub struct Iter<'a, T: Send + Sync, M: Metadata = (), const AUTO_FREE_IDS: bool = true> {
    thread_local: &'a ThreadLocal<T, M, AUTO_FREE_IDS>,
    raw: RawIter<GUARD_ACTIVE_INTERNAL>,
    prev: *const Entry<T, M, AUTO_FREE_IDS>,
}

impl<'a, T: Send + Sync, M: Metadata, const AUTO_FREE_IDS: bool> Iterator for Iter<'a, T, M, AUTO_FREE_IDS> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.raw.next(self.thread_local).map(|entry| {
            if let Some(prev) = unsafe { self.prev.as_ref() } {
                prev.guard.store(GUARD_READY, Ordering::Release);
            }
            self.prev = entry as *const _;

            unsafe { &*(&*entry.value.get()).as_ptr() }
        })
    }
}

impl<T: Send + Sync, M: Metadata, const AUTO_FREE_IDS: bool> FusedIterator for Iter<'_, T, M, AUTO_FREE_IDS> {}

impl<T: Send + Sync, M: Metadata, const AUTO_FREE_IDS: bool> Drop for Iter<'_, T, M, AUTO_FREE_IDS> {
    fn drop(&mut self) {
        if let Some(prev) = unsafe { self.prev.as_ref() } {
            prev.guard.store(GUARD_READY, Ordering::Release);
        }
    }
}

/// Mutable iterator over the contents of a `ThreadLocal`.
pub struct IterMut<'a, T: Send, M: Metadata = (), const AUTO_FREE_IDS: bool = true> {
    thread_local: &'a mut ThreadLocal<T, M, AUTO_FREE_IDS>,
    raw: RawIter<GUARD_ACTIVE_INTERNAL>,
    prev: *const Entry<T, M, AUTO_FREE_IDS>,
}

impl<'a, T: Send, M: Metadata, const AUTO_FREE_IDS: bool> Iterator for IterMut<'a, T, M, AUTO_FREE_IDS> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        self.raw.next_mut(self.thread_local).map(|entry| {
            if let Some(prev) = unsafe { self.prev.as_ref() } {
                prev.guard.store(GUARD_READY, Ordering::Release);
            }
            self.prev = entry as *const _;

            unsafe { &mut *(&mut *(&*entry).value.get()).as_mut_ptr() }
        })
    }
}

impl<T: Send, M: Metadata, const AUTO_FREE_IDS: bool> FusedIterator for IterMut<'_, T, M, AUTO_FREE_IDS> {}

impl<T: Send, M: Metadata, const AUTO_FREE_IDS: bool> Drop for IterMut<'_, T, M, AUTO_FREE_IDS> {
    fn drop(&mut self) {
        if let Some(prev) = unsafe { self.prev.as_ref() } {
            prev.guard.store(GUARD_READY, Ordering::Release);
        }
    }
}

// Manual impl so we don't call Debug on the ThreadLocal, as doing so would create a reference to
// this thread's value that potentially aliases with a mutable reference we have given out.
impl<'a, T: Send + fmt::Debug, M: Metadata, const AUTO_FREE_IDS: bool> fmt::Debug for IterMut<'a, T, M, AUTO_FREE_IDS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IterMut").field("raw", &self.raw).finish()
    }
}

/// An iterator that moves out of a `ThreadLocal`.
#[derive(Debug)]
pub struct IntoIter<T: Send, M: Metadata = (), const AUTO_FREE_IDS: bool = true> {
    thread_local: ThreadLocal<T, M, AUTO_FREE_IDS>,
    raw: RawIter<GUARD_EMPTY>,
}

impl<T: Send, M: Metadata, const AUTO_FREE_IDS: bool> Iterator for IntoIter<T, M, AUTO_FREE_IDS> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        loop {
            match self.raw.next_mut(&mut self.thread_local) {
                None => return None,
                Some(entry) => {
                    // remove the entry from the freelist as we are freeing it already.

                    // FIXME: reduce code duplication with detachment helpers of entries from threads in `Entry`
                    let free_list = unsafe { &*entry }.free_list.load(Ordering::Acquire);

                    let mut backoff = Backoff::new();
                    loop {
                        match unsafe { &*free_list }.free_list.try_lock() {
                            // FIXME: this can deref an already dealloced piece of memory.
                            Ok(mut guard) => {
                                // we got the lock and can now remove our entry from the free list
                                guard.remove(&(entry as usize));
                                let ret = unsafe {
                                    mem::replace(
                                        &mut *(&*entry).value.get(),
                                        MaybeUninit::uninit(),
                                    )
                                    .assume_init()
                                };
                                return Some(ret);
                            }
                            Err(_) => {
                                // check if the thread declared that it was dropping, if so give up and try finding a new entry
                                if unsafe { &*free_list }.dropping.load(Ordering::Acquire) {
                                    unsafe { &*entry }
                                        .guard
                                        .store(GUARD_READY, Ordering::Release);
                                    // try finding a new entry
                                    break;
                                }
                                // FIXME: do we have a better way to wait (but be able to see a change in dropping)
                                backoff.snooze();
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<T: Send, M: Metadata, const AUTO_FREE_IDS: bool> FusedIterator for IntoIter<T, M, AUTO_FREE_IDS> {}

/// Iterator over the contents of a `ThreadLocal`.
#[derive(Debug)]
pub struct IterMeta<'a, T: Send + Sync, M: Metadata = (), const AUTO_FREE_IDS: bool = true> {
    thread_local: &'a ThreadLocal<T, M, AUTO_FREE_IDS>,
    raw: RawIter<GUARD_ACTIVE_INTERNAL>,
    prev: *const Entry<T, M, AUTO_FREE_IDS>,
}

impl<'a, T: Send + Sync, M: Metadata, const AUTO_FREE_IDS: bool> Iterator for IterMeta<'a, T, M, AUTO_FREE_IDS> {
    type Item = (&'a T, &'a M);

    fn next(&mut self) -> Option<Self::Item> {
        self.raw.next(self.thread_local).map(|entry| {
            if let Some(prev) = unsafe { self.prev.as_ref() } {
                prev.guard.store(GUARD_READY, Ordering::Release);
            }
            self.prev = entry as *const _;

            (unsafe { &*(&*entry.value.get()).as_ptr() }, &entry.meta)
        })
    }
}

impl<T: Send + Sync, M: Metadata, const AUTO_FREE_IDS: bool> FusedIterator for IterMeta<'_, T, M, AUTO_FREE_IDS> {}

impl<T: Send + Sync, M: Metadata, const AUTO_FREE_IDS: bool> Drop for IterMeta<'_, T, M, AUTO_FREE_IDS> {
    fn drop(&mut self) {
        if let Some(prev) = unsafe { self.prev.as_ref() } {
            prev.guard.store(GUARD_READY, Ordering::Release);
        }
    }
}

/// Mutable iterator over the contents of a `ThreadLocal`.
pub struct IterMutMeta<'a, T: Send, M: Metadata = (), const AUTO_FREE_IDS: bool = true> {
    thread_local: &'a mut ThreadLocal<T, M, AUTO_FREE_IDS>,
    raw: RawIter<GUARD_ACTIVE_INTERNAL>,
    prev: *const Entry<T, M, AUTO_FREE_IDS>,
}

impl<'a, T: Send, M: Metadata, const AUTO_FREE_IDS: bool> Iterator for IterMutMeta<'a, T, M, AUTO_FREE_IDS> {
    type Item = (&'a mut T, &'a M);

    fn next(&mut self) -> Option<(&'a mut T, &'a M)> {
        self.raw.next_mut(self.thread_local).map(move |entry| {
            if let Some(prev) = unsafe { self.prev.as_ref() } {
                prev.guard.store(GUARD_READY, Ordering::Release);
            }
            self.prev = entry as *const _;

            (
                unsafe { &mut *(&mut *(&*entry).value.get()).as_mut_ptr() },
                unsafe { &(&*entry).meta },
            )
        })
    }
}

impl<T: Send, M: Metadata, const AUTO_FREE_IDS: bool> FusedIterator for IterMutMeta<'_, T, M, AUTO_FREE_IDS> {}

impl<T: Send, M: Metadata, const AUTO_FREE_IDS: bool> Drop for IterMutMeta<'_, T, M, AUTO_FREE_IDS> {
    fn drop(&mut self) {
        if let Some(prev) = unsafe { self.prev.as_ref() } {
            prev.guard.store(GUARD_READY, Ordering::Release);
        }
    }
}

// Manual impl so we don't call Debug on the ThreadLocal, as doing so would create a reference to
// this thread's value that potentially aliases with a mutable reference we have given out.
impl<'a, T: Send + fmt::Debug, M: Metadata, const AUTO_FREE_IDS: bool> fmt::Debug for IterMutMeta<'a, T, M, AUTO_FREE_IDS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IterMutMeta")
            .field("raw", &self.raw)
            .finish()
    }
}

fn allocate_bucket<const ALTERNATIVE: bool, const AUTO_FREE_IDS: bool, T, M: Metadata>(size: usize, tid_manager: *const Mutex<ThreadIdManager>, bucket: usize) -> *mut Entry<T, M, AUTO_FREE_IDS> {
    Box::into_raw(
        (0..size)
            .map(|n| Entry::<T, M, AUTO_FREE_IDS> {
                tid_manager,
                id: if ALTERNATIVE {
                    // we need to offset all entries by the number of all entries of previous buckets
                    (1 << (bucket + 1)) - 1 + n
                } else { usize::MAX },
                guard: AtomicUsize::new(GUARD_UNINIT),
                alternative_entry: AtomicPtr::new(null_mut()),
                free_list: Default::default(),
                outstanding_refs: Default::default(),
                meta: Default::default(),
                value: UnsafeCell::new(MaybeUninit::uninit()),
            })
            .collect(),
    ) as *mut _
}

unsafe fn deallocate_bucket<T, M: Metadata, const AUTO_FREE_IDS: bool>(bucket: *mut Entry<T, M, AUTO_FREE_IDS>, size: usize) {
    let _ = Box::from_raw(std::slice::from_raw_parts_mut(bucket, size));
}

pub trait Metadata: Send + Sync + Default {
    fn set_default(&self);
}

impl Metadata for () {
    #[inline(always)]
    fn set_default(&self) {}
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
        let mut tls: ThreadLocal<usize, ()> = ThreadLocal::new();
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
        let tls = Arc::new(ThreadLocal::<usize, ()>::new());
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
        let tls = Arc::new(ThreadLocal::<Box<i32>, ()>::new());
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

        // FIXME: there is a race condition with one of these 3 iterators!
        let mut v = tls
            .iter()
            .map(|x| {
                println!("found: {}", x);
                **x
            })
            .collect::<Vec<i32>>();
        v.sort_unstable();
        assert_eq!(vec![1], v);

        let mut v = tls.iter_mut().map(|x| **x).collect::<Vec<i32>>();
        v.sort_unstable();
        assert_eq!(vec![1], v);

        let mut v = tls.into_iter().map(|x| *x).collect::<Vec<i32>>();
        v.sort_unstable();
        assert_eq!(vec![1], v);
    }

    #[test]
    fn test_drop() {
        let local: ThreadLocal<Dropped, ()> = ThreadLocal::new();
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
