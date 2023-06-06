// Copyright 2017 Amanieu d'Antras
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::alloc::{alloc, alloc_zeroed, dealloc, Layout};
use std::cell::Cell;
use crate::{BUCKETS, Entry, POINTER_WIDTH};
use once_cell::sync::Lazy;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::mem;
use std::mem::transmute;
use std::ops::Deref;
use std::ptr::{NonNull, null};

/// Thread ID manager which allocates thread IDs. It attempts to aggressively
/// reuse thread IDs where possible to avoid cases where a ThreadLocal grows
/// indefinitely when it is used by many short-lived threads.
pub(crate) struct ThreadIdManager {
    free_from: usize,
    free_list: BinaryHeap<Reverse<usize>>,
}

impl ThreadIdManager {
    fn new() -> Self {
        SHARED_IDS[0].set(alloc_shared(1));
        Self {
            free_from: 0,
            free_list: BinaryHeap::new(),
        }
    }

    pub(crate) fn alloc(&mut self) -> usize {
        if let Some(id) = self.free_list.pop() {
            println!("alloced tid: {}", id.0);
            return id.0;
        }

        // we don't allow 1 before MAX to be returned because our buckets can only contain
        // up to usize::MAX - 1 elements, but this shouldn't have any impact in practice.
        if self.free_from >= usize::MAX - 1 {
            panic!("Ran out of thread IDs");
        }

        let id = self.free_from;
        self.free_from += 1;

        if id % 2 == 0 {
            let bucket = POINTER_WIDTH as usize - id.leading_zeros() as usize + 1;
            let bucket_size = 1 << bucket;
            SHARED_IDS[bucket].set(alloc_shared(bucket_size));
        }

        println!("alloced tid: {}", id);
        id
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

impl Drop for ThreadIdManager {
    fn drop(&mut self) {
        let buckets = POINTER_WIDTH as usize - self.free_from.next_power_of_two().leading_zeros() as usize + 1;
        for bucket in 0..buckets {
            let ptr = SHARED_IDS[bucket].get().cast_mut();
            unsafe { dealloc(ptr.cast(), Layout::array::<AtomicUsize>(1 << bucket).unwrap_unchecked()); }
        }
        println!("dealloced!");
    }
}

fn alloc_shared(size: usize) -> *const AtomicUsize {
    let ret = unsafe { alloc_zeroed(Layout::array::<AtomicUsize>(size).unwrap()) };
    if ret.is_null() {
        panic!("There was an error allocating shared counters!");
    }
    ret.cast::<AtomicUsize>().cast_const()
}

static THREAD_ID_MANAGER: Lazy<Mutex<ThreadIdManager>> =
    Lazy::new(|| Mutex::new(ThreadIdManager::new()));

pub(crate) static SHARED_IDS: [PtrCell<AtomicUsize>; BUCKETS] = {
    unsafe { transmute([null::<AtomicUsize>(); BUCKETS]) }
};

pub(crate) unsafe fn shared_id_ptr(id: usize) -> *const AtomicUsize {
    let (bucket, _, index) = id_into_parts(id);
    SHARED_IDS[bucket].get().offset(index as isize)
}

#[derive(Clone)]
pub(crate) struct PtrCell<T>(Cell<usize>, PhantomData<T>);

impl<T> PtrCell<T> {

    #[inline]
    pub(crate) fn new(val: *const T) -> Self {
        Self(Cell::new(val as usize), PhantomData)
    }

    #[inline]
    pub(crate) fn set(&self, val: *const T) {
        self.0.set(val as usize);
    }

    #[inline]
    pub(crate) fn get(&self) -> *const T {
        self.0.get() as *const T
    }

}

unsafe impl<T: Send> Send for PtrCell<T> {}
unsafe impl<T: Sync> Sync for PtrCell<T> {}

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
        let bucket = usize::from(POINTER_WIDTH) - ((id + 1).leading_zeros() as usize) - 1;
        let bucket_size = 1 << bucket;
        let index = id - (bucket_size - 1);

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
        1 << self.bucket
    }
}

/// returns the bucket, bucket size and index of the given id
#[inline]
pub(crate) fn id_into_parts(id: usize) -> (usize, usize, usize) {
    let bucket = usize::from(POINTER_WIDTH) - ((id + 1).leading_zeros() as usize) - 1;
    let bucket_size = 1 << bucket;
    let index = id - (bucket_size - 1);

    (bucket, bucket_size, index)
}

#[inline]
pub(crate) fn global_tid_manager() -> NonNull<Mutex<ThreadIdManager>> {
    unsafe { NonNull::new_unchecked((THREAD_ID_MANAGER.deref() as *const Mutex<ThreadIdManager>).cast_mut()) }
}

pub(crate) struct FreeList {
    id: usize,
    pub(crate) dropping: AtomicBool,
    pub(crate) free_list: Mutex<HashMap<usize, EntryData>>,
}

impl FreeList {
    fn new(id: usize) -> Self {
        Self {
            id,
            dropping: Default::default(),
            free_list: Mutex::new(Default::default()),
        }
    }

    fn cleanup(&self) {
        self.dropping.store(true, Ordering::Release);
        let free_list = self.free_list.lock();
        println!("alloced shared counter!");
        let outstanding_shared = unsafe { shared_id_ptr(self.id) };
        let mut outstanding = 0;
        for entry in free_list.unwrap().iter() {
            // sum up all the "failed" cleanups
            if unsafe { !entry.1.cleanup(*entry.0 as *const Entry<()>) } {
                outstanding += 1;
            }
        }

        if outstanding > 0 {
            // store the number of outstanding references
            unsafe { outstanding_shared.as_ref().unwrap_unchecked() }.store(outstanding, Ordering::Release);
        } else {
            // perform the actual cleanup of the id
            let id = unsafe { THREAD.as_ref().unwrap_unchecked() }.id;
            // Release the thread ID. Any further accesses to the thread ID
            // will go through get_slow which will either panic or
            // initialize a new ThreadGuard.
            THREAD_ID_MANAGER.lock().unwrap().free(id); // FIXME: this panicked with a poison error!
        }
    }
}

pub(crate) struct EntryData {
    pub(crate) drop_fn: unsafe fn(*const Entry<()>) -> bool,
}

impl EntryData {
    #[inline]
    unsafe fn cleanup(&self, data: *const Entry<()>) -> bool {
        let dfn = self.drop_fn;
        dfn(data)
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
    let tid = THREAD_ID_MANAGER.lock().unwrap().alloc();
    unsafe {
        FREE_LIST = Some(FreeList::new(tid));
    }
    let new = Thread::new(tid, unsafe {
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
    assert_eq!(thread.bucket_size(), 2);
    assert_eq!(thread.index, 0);

    let thread = Thread::new(2, null());
    assert_eq!(thread.id, 2);
    assert_eq!(thread.bucket, 1);
    assert_eq!(thread.bucket_size(), 2);
    assert_eq!(thread.index, 1);

    let thread = Thread::new(3, null());
    assert_eq!(thread.id, 3);
    assert_eq!(thread.bucket, 2);
    assert_eq!(thread.bucket_size(), 4);
    assert_eq!(thread.index, 0);

    let thread = Thread::new(19, null());
    assert_eq!(thread.id, 19);
    assert_eq!(thread.bucket, 4);
    assert_eq!(thread.bucket_size(), 16);
    assert_eq!(thread.index, 4);
}
