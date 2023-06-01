// Copyright 2017 Amanieu d'Antras
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{Entry, POINTER_WIDTH};
use once_cell::sync::Lazy;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::mem;
use std::ops::Deref;
use std::ptr::NonNull;

/// Thread ID manager which allocates thread IDs. It attempts to aggressively
/// reuse thread IDs where possible to avoid cases where a ThreadLocal grows
/// indefinitely when it is used by many short-lived threads.
pub(crate) struct ThreadIdManager {
    free_from: usize,
    free_list: BinaryHeap<Reverse<usize>>,
}

impl ThreadIdManager {
    pub fn new() -> Self {
        Self {
            free_from: 0,
            free_list: BinaryHeap::new(),
        }
    }

    pub(crate) fn alloc(&mut self) -> usize {
        if let Some(id) = self.free_list.pop() {
            id.0
        } else {
            let id = self.free_from;
            self.free_from = self
                .free_from
                .checked_add(1)
                .expect("Ran out of thread IDs");
            id
        }
    }

    pub(crate) fn free(&mut self, id: usize) {
        if self.free_list.iter().find(|x| x.0 == id).is_some() {
            panic!("double freed tid!");
        }
        if self.free_from <= id {
            panic!("freed tid although tid was never handed out {} max {} glob {:?} local {:?}", id, self.free_from, global_tid_manager(), self as *const ThreadIdManager);
        }

        self.free_list.push(Reverse(id));
    }
}

static THREAD_ID_MANAGER: Lazy<Mutex<ThreadIdManager>> =
    Lazy::new(|| Mutex::new(ThreadIdManager::new()));

/// Data which is unique to the current thread while it is running.
/// A thread ID may be reused after a thread exits.
#[derive(Copy, Clone)]
pub(crate) struct Thread {
    /// The thread ID obtained from the thread ID manager.
    pub(crate) id: usize,
    /// The bucket this thread's local storage will be in.
    pub(crate) bucket: usize,
    /// The index into the bucket this thread's local storage is in.
    pub(crate) index: usize,
    pub(crate) free_list: *const FreeList,
}

impl Thread {
    fn new(id: usize, free_list: *const FreeList) -> Self {
        let bucket = usize::from(POINTER_WIDTH) - id.leading_zeros() as usize;
        let bucket_size = 1 << bucket.saturating_sub(1);
        let index = if id != 0 { id ^ bucket_size } else { 0 };

        Self {
            id,
            bucket,
            index,
            free_list,
        }
    }

    /// The size of the bucket this thread's local storage will be in.
    #[inline]
    pub(crate) fn bucket_size(&self) -> usize {
        1 << self.bucket.saturating_sub(1)
    }
}

/// returns the bucket, bucket size and index of the given id
#[inline]
pub(crate) fn id_into_parts(id: usize) -> (usize, usize, usize) {
    let bucket = usize::from(POINTER_WIDTH) - id.leading_zeros() as usize;
    let bucket_size = 1 << bucket.saturating_sub(1);
    let index = if id != 0 { id ^ bucket_size } else { 0 };

    (bucket, bucket_size, index)
}

#[inline]
pub(crate) fn global_tid_manager() -> NonNull<Mutex<ThreadIdManager>> {
    unsafe { NonNull::new_unchecked((THREAD_ID_MANAGER.deref() as *const Mutex<ThreadIdManager>).cast_mut()) }
}

pub(crate) struct FreeList {
    pub(crate) dropping: AtomicBool,
    pub(crate) free_list: Mutex<HashMap<usize, EntryData>>,
}

impl FreeList {
    fn new() -> Self {
        Self {
            dropping: Default::default(),
            free_list: Mutex::new(Default::default()),
        }
    }

    fn cleanup(&self) {
        self.dropping.store(true, Ordering::Release);
        let free_list = self.free_list.lock();
        let outstanding_shared = Box::new(AtomicUsize::new(usize::MAX));
        let mut outstanding = 0;
        for entry in free_list.unwrap().iter() {
            // sum up all the "failed" cleanups
            if unsafe { !entry.1.cleanup(*entry.0 as *const Entry<()>, &outstanding_shared as *const Box<AtomicUsize> as *const AtomicUsize) } {
                outstanding += 1;
            }
        }

        // store the actual number of outstanding references and ensure that all
        // updates that happened before are applied to the actual value after
        // the initial store and cleanup the id if there are no actual outstanding
        // references left after syncing the references.
        let prev = outstanding_shared.swap(outstanding, Ordering::Release);
        let diff = usize::MAX - prev;
        if diff > 0 {
            if outstanding_shared.fetch_sub(diff, Ordering::AcqRel) == diff {
                // perform the actual cleanup of the id
                let id = unsafe { THREAD.as_ref().unwrap_unchecked() }.id;
                THREAD_ID_MANAGER.lock().unwrap().free(id); // FIXME: this is not okay if we are an alternative_id or is it?
            }
        }

        mem::forget(outstanding);
    }
}

pub(crate) struct EntryData {
    pub(crate) drop_fn: unsafe fn(*const Entry<()>, *const AtomicUsize) -> bool,
}

impl EntryData {
    #[inline]
    unsafe fn cleanup(&self, data: *const Entry<()>, outstanding: *const AtomicUsize) -> bool {
        let dfn = self.drop_fn;
        dfn(data, outstanding)
    }
}

// This is split into 2 thread-local variables so that we can check whether the
// thread is initialized without having to register a thread-local destructor.
//
// This makes the fast path smaller.
#[thread_local]
static mut THREAD: Option<Thread> = None;
#[thread_local]
static mut FREE_LIST: Option<FreeList> = None;
thread_local! { static THREAD_GUARD: ThreadGuard = const { ThreadGuard }; }

// Guard to ensure the thread ID is released on thread exit.
struct ThreadGuard;

impl Drop for ThreadGuard {
    fn drop(&mut self) {
        unsafe {
            // first clean up all entries in the freelist
            FREE_LIST.as_ref().unwrap_unchecked().cleanup(); // FIXME: this causes invalid drops (maybe)
            // ... then clean up the freelist itself.
            FREE_LIST.take();
        }
        // Release the thread ID. Any further accesses to the thread ID
        // will go through get_slow which will either panic or
        // initialize a new ThreadGuard.
        THREAD_ID_MANAGER
            .lock()
            .unwrap()
            .free(unsafe { THREAD.as_ref().unwrap_unchecked().id }); // FIXME: doesn't this lead to a double free of thread ids?
    }
}

/// Returns a thread ID for the current thread, allocating one if needed.
#[inline]
pub(crate) fn get() -> Thread {
    if let Some(thread) = unsafe { THREAD } {
        thread
    } else {
        get_slow()
    }
}

/// Out-of-line slow path for allocating a thread ID.
#[cold]
fn get_slow() -> Thread {
    unsafe {
        FREE_LIST = Some(FreeList::new());
    }
    let new = Thread::new(THREAD_ID_MANAGER.lock().unwrap().alloc(), unsafe {
        FREE_LIST.as_ref().unwrap_unchecked() as *const FreeList
    });
    unsafe {
        THREAD = Some(new);
    }
    THREAD_GUARD.with(|_| {});
    new
}

#[test]
fn test_thread() {
    use std::ptr::null;
    let thread = Thread::new(0, null());
    assert_eq!(thread.id, 0);
    assert_eq!(thread.bucket, 0);
    assert_eq!(thread.bucket_size(), 1);
    assert_eq!(thread.index, 0);

    let thread = Thread::new(1, null());
    assert_eq!(thread.id, 1);
    assert_eq!(thread.bucket, 1);
    assert_eq!(thread.bucket_size(), 1);
    assert_eq!(thread.index, 0);

    let thread = Thread::new(2, null());
    assert_eq!(thread.id, 2);
    assert_eq!(thread.bucket, 2);
    assert_eq!(thread.bucket_size(), 2);
    assert_eq!(thread.index, 0);

    let thread = Thread::new(3, null());
    assert_eq!(thread.id, 3);
    assert_eq!(thread.bucket, 2);
    assert_eq!(thread.bucket_size(), 2);
    assert_eq!(thread.index, 1);

    let thread = Thread::new(19, null());
    assert_eq!(thread.id, 19);
    assert_eq!(thread.bucket, 5);
    assert_eq!(thread.bucket_size(), 16);
    assert_eq!(thread.index, 3);
}
