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
//! let entry = tls.get();
//! assert_eq!(entry.as_ref().map(|entry| entry.value()), None);
//! assert_eq!(tls.get_or(|_| 5, |_| {}).value(), &5);
//! let entry = tls.get();
//! assert_eq!(entry.as_ref().map(|entry| entry.value()), Some(&5));
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
//!         let cell = tls2.get_or(|_| Cell::new(0), |_| {});
//!         cell.value().set(cell.value().get() + 1);
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

mod thread_id;
mod unreachable;

use crate::thread_id::{EntryData, FreeList, global_tid_manager, Thread, ThreadIdManager};
use crossbeam_utils::Backoff;
use smallvec::{smallvec, SmallVec};
use std::cell::UnsafeCell;
use std::fmt;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem;
use std::mem::{size_of, transmute, MaybeUninit};
use std::ops::Deref;
use std::panic::UnwindSafe;
use std::ptr;
use std::ptr::{null_mut, NonNull, null};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Mutex;

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
const GUARD_ACTIVE_ITERATOR: usize = 5;
const GUARD_FREE_MANUALLY: usize = 6; // FIXME: if we store this once, when are we able to reuse the entry again?

#[derive(Clone)]
pub struct UnsafeToken<T, M: Metadata, const AUTO_FREE_IDS: bool>(NonNull<Entry<T, M, AUTO_FREE_IDS>>);

impl<T, M: Metadata, const AUTO_FREE_IDS: bool> UnsafeToken<T, M, AUTO_FREE_IDS> {

    /// This is `MaybeUninit<T>` instead of `T` because when `UnsafeToken` is handed out
    /// in the first closure of`get_or` the value is still uninitialized.
    #[inline]
    pub unsafe fn value_ptr(&self) -> *const MaybeUninit<T> {
        unsafe { &*self.0.as_ref().value.get() as *const MaybeUninit<T> }
    }

    /// This is `MaybeUninit<T>` instead of `T` because when `UnsafeToken` is handed out
    /// in the first closure of`get_or` the value is still uninitialized.
    #[inline]
    pub unsafe fn value(&self) -> &MaybeUninit<T> {
        unsafe { &&*self.0.as_ref().value.get() }
    }

    #[inline]
    pub unsafe fn meta_ptr(&self) -> *const M {
        self.meta() as *const M
    }

    #[inline]
    pub unsafe fn meta(&self) -> &M {
        unsafe { &self.0.as_ref().meta }
    }

    /// SAFETY: This may only be called after the thread associated with this thread local has finished.
    pub unsafe fn destruct(self) {
        unsafe { self.0.as_ref().free_id(); }
    }

    #[inline]
    pub fn duplicate(&self) -> Self {
        Self(self.0.clone())
    }

}

#[derive(PartialEq)]
pub struct EntryToken<'a, ACCESS, T, M: Metadata = (), const AUTO_FREE_IDS: bool = true>(NonNull<Entry<T, M, AUTO_FREE_IDS>>, PhantomData<&'a ACCESS>);

impl<'a, ACCESS, T, M: Metadata, const AUTO_FREE_IDS: bool> EntryToken<'a, ACCESS, T, M, AUTO_FREE_IDS> {

    #[inline]
    pub fn value(&self) -> &T {
        unsafe { (&*self.0.as_ref().value.get()).assume_init_ref() }
    }

    #[inline]
    pub fn meta(&self) -> &M {
        unsafe { &self.0.as_ref().meta }
    }

    /// SAFETY: This may only be called after the thread associated with this thread local has finished.
    pub unsafe fn destruct(self) {
        unsafe { self.0.as_ref().free_id(); }
    }

    #[inline]
    pub unsafe fn into_unsafe_token(self) -> UnsafeToken<T, M, AUTO_FREE_IDS> {
        UnsafeToken(self.0)
    }

}

impl<'a, T, M: Metadata, const AUTO_FREE_IDS: bool> EntryToken<'a, MutRefAccess, T, M, AUTO_FREE_IDS> {

    #[inline]
    pub fn value_mut(&mut self) -> &mut T {
        unsafe { (&mut *self.0.as_ref().value.get()).assume_init_mut() }
    }

}

impl<'a, T, M: Metadata, const AUTO_FREE_IDS: bool> Clone for EntryToken<'a, RefAccess, T, M, AUTO_FREE_IDS> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData::default())
    }
}

impl<'a, T, M: Metadata, const AUTO_FREE_IDS: bool> Copy for EntryToken<'a, RefAccess, T, M, AUTO_FREE_IDS> {}

pub struct MutRefAccess;

pub struct RefAccess;

// FIXME: should we primarily determine whether an entry is empty via the free_list ptr or the guard value?
struct Entry<T, M: Metadata = (), const AUTO_FREE_IDS: bool = true> {
    tid_manager: NonNull<Mutex<ThreadIdManager>>,
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
        let mut alt = self.alternative_entry.load(Ordering::Acquire);

        while let Some(alt_ref) = alt.as_ref() {
            // when its freelist is non-null, we know that the entry is active
            if !alt_ref.free_list.load(Ordering::Acquire).is_null() {
                break;
            }

            if !alt_ref.try_detach_thread_locally() {
                cleanup.push(alt.cast_const());
            }

            alt = alt_ref.alternative_entry.load(Ordering::Acquire);
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

        self.remove_from_freelist()
    }

    fn remove_from_freelist(&self) -> bool {
        let free_list = self.free_list.load(Ordering::Acquire);

        let backoff = Backoff::new();
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

    /// This will get called when the thread associated with this entry exits.
    pub(crate) unsafe fn cleanup(slf: *const Entry<()>, remaining_cnt: *const AtomicUsize) -> bool {
        let slf = unsafe { &*slf.cast::<Entry<T, M, AUTO_FREE_IDS>>() };
        let backoff = Backoff::new();
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
        // signal that there is no more manual cleanup required for future threads that get assigned this
        // entry's id so they can use the actual entry and don't always fall back to an alternative_entry
        // even though the entry is completely unused.
        self.guard.store(GUARD_EMPTY, Ordering::Release);
        // check if we are a "main" entry and our thread is finished
        if let Some(outstanding) = unsafe { self.outstanding_refs.load(Ordering::Acquire).as_ref() } {
            if outstanding.fetch_sub(1, Ordering::AcqRel) != 1 {
                // there are outstanding references left, so we can't free the id yet.
                return;
            }
        }
        // the tid_manager is either an `alternative` id manager or the `global` tid manager.
        unsafe { self.tid_manager.as_ref() }.lock().unwrap().free(self.id);
    }

}

impl<T, M: Metadata, const AUTO_FREE_IDS: bool> Drop for Entry<T, M, AUTO_FREE_IDS> {
    fn drop(&mut self) {
        let guard = *self.guard.get_mut();
        if guard == GUARD_READY || guard == GUARD_ACTIVE_INTERNAL {
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
                let backoff = Backoff::new();
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
                let backoff = Backoff::new();
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
            *bucket = allocate_bucket::<false, AUTO_FREE_IDS, T, M>(bucket_size, global_tid_manager(), i);

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
            let tid_manager = unsafe { NonNull::new_unchecked((tid_manager.as_ref() as *const Mutex<ThreadIdManager>).cast_mut()) };
            *bucket = allocate_bucket::<true, AUTO_FREE_IDS, T, M>(bucket_size, tid_manager, i);

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
    pub fn get(&self) -> Option<EntryToken<RefAccess, T, M, AUTO_FREE_IDS>> {
        self.get_inner(thread_id::get())
    }

    /// Returns the element for the current thread, or creates it if it doesn't
    /// exist.
    pub fn get_or<F, MF>(&self, create: F, meta: MF) -> EntryToken<'_, RefAccess, T, M, AUTO_FREE_IDS>
    where
        F: FnOnce(UnsafeToken<T, M, AUTO_FREE_IDS>) -> T,
        MF: FnOnce(EntryToken<RefAccess, T, M, AUTO_FREE_IDS>),
    {
        let thread = thread_id::get();
        if let Some(val) = self.get_inner(thread) {
            return val;
        }

        println!("failed fetching entry of: {}", thread.id);

        self.insert(create, meta)
    }

    fn get_inner(&self, thread: Thread) -> Option<EntryToken<'_, RefAccess, T, M, AUTO_FREE_IDS>> {
        let bucket_ptr =
            unsafe { self.buckets.get_unchecked(thread.bucket) }.load(Ordering::Acquire);
        if bucket_ptr.is_null() {
            return None;
        }
        let mut entry_ptr = unsafe { bucket_ptr.add(thread.index) };
        let mut entry = unsafe { &*entry_ptr };
        let mut free_list = entry.free_list.load(Ordering::Acquire);

        // check if the entry is unusable (it does not have an init value and it's not a pseudo-present value)
        while free_list.cast_const() != thread.free_list {
            let alt_ptr = entry.alternative_entry.load(Ordering::Acquire);

            // check if there is an alternative entry present
            if alt_ptr.is_null() {
                return None;
            }

            entry_ptr = alt_ptr;
            entry = unsafe { &*entry_ptr };
            free_list = entry.free_list.load(Ordering::Acquire);
        }

        Some(EntryToken(unsafe { NonNull::new_unchecked(alt_ptr) }, Default::default()))
    }

    #[cold]
    fn insert<F: FnOnce(UnsafeToken<T, M, AUTO_FREE_IDS>) -> T, FM: FnOnce(EntryToken<RefAccess, T, M, AUTO_FREE_IDS>)>(&self, f: F, fm: FM) -> EntryToken<RefAccess, T, M, AUTO_FREE_IDS> {
        let thread = thread_id::get();
        let bucket_atomic_ptr = unsafe { self.buckets.get_unchecked(thread.bucket) };
        let bucket_ptr: *const _ = bucket_atomic_ptr.load(Ordering::Acquire);

        // If the bucket doesn't already exist, we need to allocate it
        let bucket_ptr = if bucket_ptr.is_null() {
            let new_bucket = allocate_bucket::<false, AUTO_FREE_IDS, T, M>(thread.bucket_size(), global_tid_manager(), thread.bucket);

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
        let mut entry = unsafe { &*unsafe { bucket_ptr.add(thread.index) } };
        let entry_ptr = entry as *const Entry<T, M, AUTO_FREE_IDS>;

        // check if the entry isn't cleaned up automatically
        if entry.guard.load(Ordering::Acquire) == GUARD_FREE_MANUALLY {
            // traverse alternative entries recursively until we find the end and can insert our new entry
            while let Some(alt) = unsafe { entry.alternative_entry.load(Ordering::Acquire).as_ref() } {
                entry = alt;
            }
            let alt = self.acquire_alternative_entry();
            entry.alternative_entry.store(alt.cast_mut(), Ordering::Release);
            entry = unsafe { &*alt };
            println!("acquired alternative entry: {:?} (master entry: {:?})", alt, entry_ptr);
        }

        let value_ptr = entry.value.get();
        unsafe { value_ptr.write(MaybeUninit::new(f(UnsafeToken(NonNull::new_unchecked((entry as *const Entry<T, M, AUTO_FREE_IDS>).cast_mut()))))) };
        if size_of::<M>() > 0 {
            fm(EntryToken(unsafe { NonNull::new_unchecked((entry as *const Entry<T, M, AUTO_FREE_IDS>).cast_mut()) }, Default::default()));
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

        EntryToken(unsafe { NonNull::new_unchecked((entry as *const Entry<T, M, AUTO_FREE_IDS>).cast_mut()) }, Default::default())
    }

    fn acquire_alternative_entry(&self) -> *const Entry<T, M, AUTO_FREE_IDS> {
        let id = self.alternative_entry_ids.lock().unwrap().alloc();
        let (bucket, bucket_size, index) = thread_id::id_into_parts(id);

        let bucket_atomic_ptr = unsafe { self.alternative_buckets.get_unchecked(bucket) };
        let bucket_ptr: *const _ = bucket_atomic_ptr.load(Ordering::Acquire);

        // If the bucket doesn't already exist, we need to allocate it
        let bucket_ptr = if bucket_ptr.is_null() {
            let tid_manager = unsafe { NonNull::new_unchecked((self.alternative_entry_ids.as_ref() as *const Mutex<ThreadIdManager>).cast_mut()) };
            let new_bucket = allocate_bucket::<true, AUTO_FREE_IDS, T, M>(bucket_size, tid_manager, bucket);

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
    type Item = EntryToken<'a, RefAccess, T, M, AUTO_FREE_IDS>;
    type IntoIter = Iter<'a, T, M, AUTO_FREE_IDS>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: Send, M: Metadata, const AUTO_FREE_IDS: bool> IntoIterator for &'a mut ThreadLocal<T, M, AUTO_FREE_IDS> {
    type Item = EntryToken<'a, MutRefAccess, T, M, AUTO_FREE_IDS>;
    type IntoIter = IterMut<'a, T, M, AUTO_FREE_IDS>;

    fn into_iter(self) -> IterMut<'a, T, M, AUTO_FREE_IDS> {
        self.iter_mut()
    }
}

impl<T: Send + Default, const AUTO_FREE_IDS: bool> ThreadLocal<T, (), AUTO_FREE_IDS> {
    /// Returns the element for the current thread, or creates a default one if
    /// it doesn't exist.
    pub fn get_or_default(&self) -> EntryToken<RefAccess, T, (), AUTO_FREE_IDS> {
        self.get_or(|_| Default::default(), |_| ())
    }
}

impl<T: Send + fmt::Debug, M: Metadata, const AUTO_FREE_IDS: bool> fmt::Debug for ThreadLocal<T, M, AUTO_FREE_IDS> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let entry = self.get();
        write!(f, "ThreadLocal {{ local_data: {:?} }}", entry.as_ref().map(|entry| entry.value()))
    }
}

impl<T: Send + UnwindSafe, M: Metadata, const AUTO_FREE_IDS: bool> UnwindSafe for ThreadLocal<T, M, AUTO_FREE_IDS> {}

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
                let backoff = Backoff::new();
                // this loop will only ever loop multiple times if the guard of the current entry
                // is `GUARD_ACTIVE_ITERATOR`. We have to loop here as the entry is still valid but
                // we have to wait on another iterator to use it.
                loop {
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
                            if guard != GUARD_ACTIVE_ITERATOR {
                                break;
                            }
                            backoff.snooze();
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
    raw: RawIter<GUARD_ACTIVE_ITERATOR>,
    prev: *const Entry<T, M, AUTO_FREE_IDS>,
}

impl<'a, T: Send + Sync, M: Metadata, const AUTO_FREE_IDS: bool> Iterator for Iter<'a, T, M, AUTO_FREE_IDS> {
    type Item = EntryToken<'a, RefAccess, T, M, AUTO_FREE_IDS>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(prev) = unsafe { self.prev.as_ref() } {
            // mark the previous entry as `READY` again as we are done using it at this point.
            prev.guard.store(GUARD_READY, Ordering::Release);
            let alt_ptr = prev.alternative_entry.load(Ordering::Acquire);
            if let Some(alt) = unsafe { alt_ptr.as_ref() } {
                let backoff = Backoff::new();
                // this loop will only ever loop multiple times if the guard of the current entry
                // is `GUARD_ACTIVE_ITERATOR`. We have to loop here as the entry is still valid but
                // we have to wait on another iterator to use it.
                loop {
                    match alt.guard.compare_exchange(
                        GUARD_READY,
                        GUARD_ACTIVE_ITERATOR,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            return Some(EntryToken(unsafe { NonNull::new_unchecked(alt_ptr) }, Default::default()));
                        }
                        Err(guard) => {
                            if guard == GUARD_UNINIT {
                                return None;
                            }
                            if guard != GUARD_ACTIVE_ITERATOR {
                                break;
                            }
                            backoff.snooze();
                        }
                    }
                }

                self.prev = alt as *const _;

                return Some(EntryToken(unsafe { NonNull::new_unchecked(alt_ptr) }, Default::default()));
            }
        }

        self.raw.next(self.thread_local).map(|entry| {
            self.prev = entry as *const _;

            EntryToken(unsafe { NonNull::new_unchecked((entry as *const Entry<T, M, AUTO_FREE_IDS>).cast_mut()) }, Default::default())
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
    raw: RawIter<GUARD_ACTIVE_ITERATOR>,
    prev: *const Entry<T, M, AUTO_FREE_IDS>,
}

impl<'a, T: Send, M: Metadata, const AUTO_FREE_IDS: bool> Iterator for IterMut<'a, T, M, AUTO_FREE_IDS> {
    type Item = EntryToken<'a, MutRefAccess, T, M, AUTO_FREE_IDS>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(prev) = unsafe { self.prev.as_ref() } {
            // mark the previous entry as `READY` again as we are done using it at this point.
            prev.guard.store(GUARD_READY, Ordering::Release);
            let alt_ptr = prev.alternative_entry.load(Ordering::Acquire);
            if let Some(alt) = unsafe { alt_ptr.as_ref() } {
                let backoff = Backoff::new();
                // this loop will only ever loop multiple times if the guard of the current entry
                // is `GUARD_ACTIVE_ITERATOR`. We have to loop here as the entry is still valid but
                // we have to wait on another iterator to use it.
                loop {
                    match alt.guard.compare_exchange(
                        GUARD_READY,
                        GUARD_ACTIVE_ITERATOR,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => {
                            return Some(EntryToken(unsafe { NonNull::new_unchecked(alt_ptr) }, Default::default()));
                        }
                        Err(guard) => {
                            if guard == GUARD_UNINIT {
                                return None;
                            }
                            if guard != GUARD_ACTIVE_ITERATOR {
                                break;
                            }
                            backoff.snooze();
                        }
                    }
                }

                self.prev = alt as *const _;

                return Some(EntryToken(unsafe { NonNull::new_unchecked(alt_ptr) }, Default::default()));
            }
        }

        self.raw.next_mut(self.thread_local).map(|entry| {
            self.prev = entry.cast_const();

            EntryToken(unsafe { NonNull::new_unchecked(entry) }, Default::default())
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

                    if unsafe { &*entry }.remove_from_freelist() {
                        let ret = unsafe {
                            mem::replace(
                                &mut *(&*entry).value.get(),
                                MaybeUninit::uninit(),
                            )
                                .assume_init()
                        };
                        return Some(ret);
                    }
                }
            }
        }
    }
}

impl<T: Send, M: Metadata, const AUTO_FREE_IDS: bool> FusedIterator for IntoIter<T, M, AUTO_FREE_IDS> {}

fn allocate_bucket<const ALTERNATIVE: bool, const AUTO_FREE_IDS: bool, T, M: Metadata>(size: usize, tid_manager: NonNull<Mutex<ThreadIdManager>>, bucket: usize) -> *mut Entry<T, M, AUTO_FREE_IDS> {
    Box::into_raw(
        (0..size)
            .map(|n| Entry::<T, M, AUTO_FREE_IDS> {
                tid_manager,
                id: {
                    // we need to offset all entries by the number of all entries of previous buckets
                    (1 << (bucket + 1)) - 1 + n
                },
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
        assert_eq!(None, tls.get().map(|entry| *entry.value()));
        assert_eq!("ThreadLocal { local_data: None }", format!("{:?}", &tls));
        assert_eq!(0, *tls.get_or(|_| create(), |_| {}).value());
        let tmp = tls.get();
        assert_eq!(Some(&0), tmp.as_ref().map(|entry| entry.value()));
        assert_eq!(0, *tls.get_or(|_| create(), |_| {}).value());
        let tmp = tls.get();
        assert_eq!(Some(&0), tmp.as_ref().map(|entry| entry.value()));
        assert_eq!(0, *tls.get_or(|_| create(), |_| {}).value());
        let tmp = tls.get();
        assert_eq!(Some(&0), tmp.as_ref().map(|entry| entry.value()));
        assert_eq!("ThreadLocal { local_data: Some(0) }", format!("{:?}", &tls));
        tls.clear();
        assert_eq!(None, tls.get().map(|entry| *entry.value()));
    }

    #[test]
    fn different_thread() {
        let create = make_create();
        let tls = Arc::new(ThreadLocal::<usize, ()>::new());
        let tmp = tls.get();
        assert_eq!(None, tmp.as_ref().map(|entry| entry.value()));
        assert_eq!(0, *tls.get_or(|_| create(), |_| {}).value());
        let tmp = tls.get();
        assert_eq!(Some(&0), tmp.as_ref().map(|entry| entry.value()));

        let tls2 = tls.clone();
        let create2 = create.clone();
        thread::spawn(move || {
            let tmp = tls2.get();
            assert_eq!(None, tmp.as_ref().map(|entry| entry.value()));
            assert_eq!(1, *tls2.get_or(|_| create2(), |_| {}).value());
            let tmp = tls2.get();
            assert_eq!(Some(&1), tmp.as_ref().map(|entry| entry.value()));
        })
        .join()
        .unwrap();

        let tmp = tls.get();
        assert_eq!(Some(&0), tmp.as_ref().map(|entry| entry.value()));
        assert_eq!(0, *tls.get_or(|_| create(), |_| {}).value());
    }

    #[test]
    fn iter() {
        let tls = Arc::new(ThreadLocal::<Box<i32>, ()>::new());
        tls.get_or(|_| Box::new(1), |_| {});

        let tls2 = tls.clone();
        thread::spawn(move || {
            tls2.get_or(|_| Box::new(2), |_| {});
            let tls3 = tls2.clone();
            thread::spawn(move || {
                tls3.get_or(|_| Box::new(3), |_| {});
            })
            .join()
            .unwrap();
            drop(tls2);
        })
        .join()
        .unwrap();

        let mut tls = Arc::try_unwrap(tls).unwrap();

        let mut v = tls
            .iter()
            .map(|x| {
                println!("found: {}", x.value());
                **x.value()
            })
            .collect::<Vec<i32>>();
        v.sort_unstable();
        assert_eq!(vec![1], v);

        let mut v = tls.iter_mut().map(|x| **x.value()).collect::<Vec<i32>>();
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
        local.get_or(|_| Dropped(dropped.clone()), |_| {});
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
