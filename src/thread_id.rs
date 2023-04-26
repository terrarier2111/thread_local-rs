// Copyright 2017 Amanieu d'Antras
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{Entry, INVALID_THREAD_ID, POINTER_WIDTH};
use once_cell::sync::Lazy;
use std::cell::{Cell, RefCell, UnsafeCell};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::process::abort;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::usize;

/// Thread ID manager which allocates thread IDs. It attempts to aggressively
/// reuse thread IDs where possible to avoid cases where a ThreadLocal grows
/// indefinitely when it is used by many short-lived threads.
struct ThreadIdManager {
    free_from: usize,
    free_list: BinaryHeap<Reverse<usize>>,
}
impl ThreadIdManager {
    fn new() -> Self {
        Self {
            free_from: 0,
            free_list: BinaryHeap::new(),
        }
    }
    fn alloc(&mut self) -> usize {
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
    fn free(&mut self, id: usize) {
        self.free_list.push(Reverse(id));
    }
}
static THREAD_ID_MANAGER: Lazy<Mutex<ThreadIdManager>> =
    Lazy::new(|| Mutex::new(ThreadIdManager::new()));

fn thread_id() -> usize {
    let tid = thread_id::get();

    if tid == INVALID_THREAD_ID {
        abort();
    }

    tid
}

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
    fn new(id: usize, free_list: *const FreeList) -> Thread {
        let bucket = usize::from(POINTER_WIDTH) - id.leading_zeros() as usize;
        let bucket_size = 1 << bucket.saturating_sub(1);
        let index = if id != 0 { id ^ bucket_size } else { 0 };

        Thread {
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
        println!("start cleaning up...");
        self.dropping.store(true, Ordering::Release);
        let mut free_list = self.free_list.lock();
        for entry in free_list.unwrap().iter() {
            unsafe {
                entry.1.cleanup(*entry.0 as *const Entry<()>);
            }
        }
        println!("cleaned up!");
    }
}

pub(crate) struct EntryData {
    pub(crate) drop_fn: unsafe fn(*const Entry<()>),
}

impl EntryData {
    #[inline]
    unsafe fn cleanup(&self, data: *const Entry<()>) {
        let dfn = self.drop_fn;
        unsafe { dfn(data) };
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
            FREE_LIST.as_ref().unwrap_unchecked().cleanup();
            // ... then clean up the freelist itself.
            FREE_LIST.take();
        }
        // Release the thread ID. Any further accesses to the thread ID
        // will go through get_slow which will either panic or
        // initialize a new ThreadGuard.
        THREAD_ID_MANAGER
            .lock()
            .unwrap()
            .free(unsafe { THREAD.as_ref().unwrap_unchecked().id });
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
