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
use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::mem;
use std::mem::{ManuallyDrop, transmute};
use std::ops::Deref;
use std::ptr::{NonNull, null};
use rustc_hash::{FxHasher, FxHashMap};
use crate::mutex::Mutex;

// FIXME: do the mutexes actually experience low contention or should a mutex implementation that expects more contention be chosen instead?

/// Thread ID manager which allocates thread IDs. It attempts to aggressively
/// reuse thread IDs where possible to avoid cases where a ThreadLocal grows
/// indefinitely when it is used by many short-lived threads.
pub(crate) struct ThreadIdManager {
    free_from: usize,
    free_list: BinaryHeap<Reverse<usize>>,
}

impl ThreadIdManager {
    const fn new() -> Self {
        Self {
            free_from: 0,
            free_list: unsafe { transmute::<Vec<Reverse<usize>>, BinaryHeap<Reverse<usize>>>(Vec::new()) }, // FIXME: this is unsound, use const constructor, once its available!
        }
    }

    pub(crate) fn alloc(&mut self) -> usize {
        if let Some(id) = self.free_list.pop() {
            // println!("alloced tid: {}", id.0);
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
            if id == 0 {
                SHARED_IDS[0].set(alloc_shared(1));
            }

            let bucket = POINTER_WIDTH as usize - id.leading_zeros() as usize + 1;
            let bucket_size = 1 << bucket;
            SHARED_IDS[bucket].set(alloc_shared(bucket_size));
        }

        // println!("alloced tid: {}", id);
        id
    }

    pub(crate) fn free(&mut self, id: usize) {
        /*if self.free_list.iter().find(|x| x.0 == id).is_some() {
            panic!("double freed tid!");
        }
        if self.free_from <= id {
            panic!("freed tid although tid was never handed out {} max {} glob {:?} local {:?}", id, self.free_from, global_tid_manager(), self as *const ThreadIdManager);
        }*/

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

static THREAD_ID_MANAGER: Mutex<ThreadIdManager> = Mutex::new(ThreadIdManager::new());

pub(crate) static SHARED_IDS: [PtrCell<AtomicUsize>; BUCKETS] = {
    unsafe { transmute([null::<AtomicUsize>(); BUCKETS]) }
};

#[inline]
pub(crate) unsafe fn shared_id_ptr(id: usize) -> *const AtomicUsize {
    let (bucket, _, index) = id_into_parts(id);
    SHARED_IDS[bucket].get().offset(index as isize)
}

#[derive(Clone)]
#[repr(transparent)]
pub(crate) struct PtrCell<T>(Cell<*const T>);

impl<T> PtrCell<T> {

    #[inline]
    pub(crate) fn new(val: *const T) -> Self {
        Self(Cell::new(val))
    }

    #[inline]
    pub(crate) fn set(&self, val: *const T) {
        self.0.set(val);
    }

    #[inline]
    pub(crate) fn get(&self) -> *const T {
        self.0.get()
    }

}

unsafe impl<T: Send> Send for PtrCell<T> {}
unsafe impl<T: Sync> Sync for PtrCell<T> {}

#[repr(transparent)]
pub(crate) struct SendSyncPtr<T>(pub *const T);

unsafe impl<T: Send> Send for SendSyncPtr<T> {}
unsafe impl<T: Sync> Sync for SendSyncPtr<T> {}

impl<T> PartialEq<Self> for SendSyncPtr<T> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T> Eq for SendSyncPtr<T> {}

impl<T> Hash for SendSyncPtr<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.0 as usize);
    }
}

/// Data which is unique to the current thread while it is running.
/// A thread ID may be reused after a thread exits.
#[derive(Copy, Clone)]
pub(crate) struct Thread {
    // FIXME: should we rather just store the id and recompute bucket and index every time we need them?
    /// The bucket this thread's local storage will be in.
    pub(crate) bucket: usize,
    /// The index into the bucket this thread's local storage is in.
    pub(crate) index: usize,
    pub(crate) free_list: *const FreeList,
}

impl Thread {
    fn new(id: usize, free_list: *const FreeList) -> Self {
        let (bucket, _, index) = id_into_parts(id);

        Self {
            bucket,
            index,
            free_list,
        }
    }

    /// The size of the bucket this thread's local storage will be in.
    #[inline(always)]
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
pub(crate) fn free_id(id: usize) {
    THREAD_ID_MANAGER.lock().free(id);
}

pub(crate) struct FreeList {
    id: usize,
    pub(crate) dropping: AtomicBool,
    pub(crate) free_list: Mutex<FxHashMap<SendSyncPtr<Entry<()>>, EntryData>>,
}

impl FreeList {
    fn new(id: usize) -> Self {
        Self {
            id,
            dropping: Default::default(),
            free_list: Mutex::new(FxHashMap::default()),
        }
    }

    fn cleanup(&self) {
        self.dropping.store(true, Ordering::Release);
        let free_list = self.free_list.lock();
        // println!("alloced shared counter!");
        let outstanding_shared = unsafe { shared_id_ptr(self.id) };
        let mut outstanding = 0;
        for entry in free_list.iter() {
            // sum up all the "failed" cleanups
            if unsafe { !entry.1.cleanup(entry.0.0) } {
                outstanding += 1;
            }
        }

        if outstanding > 0 {
            // store the number of outstanding references
            unsafe { outstanding_shared.as_ref().unwrap_unchecked() }.store(outstanding, Ordering::Release);
        } else {
            // perform the actual cleanup of the id

            // Release the thread ID. Any further accesses to the thread ID
            // will go through get_slow which will either panic or
            // initialize a new ThreadGuard.
            THREAD_ID_MANAGER.lock().free(self.id);
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
static mut FREE_LIST: Option<ManuallyDrop<FreeList>> = None;
thread_local! { static THREAD_GUARD: ThreadGuard = const { ThreadGuard }; }

// Guard to ensure the thread ID is released on thread exit.
struct ThreadGuard;

impl Drop for ThreadGuard {
    fn drop(&mut self) {
        unsafe {
            let reference = FREE_LIST.as_ref().unwrap_unchecked().deref();
            // first clean up all entries in the freelist
            reference.cleanup();
            // ... then clean up the freelist itself.
            (reference as *const FreeList).read();
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
    let tid = THREAD_ID_MANAGER.lock().alloc();
    unsafe {
        FREE_LIST = Some(ManuallyDrop::new(FreeList::new(tid)));
    }
    let new = Thread::new(tid, unsafe {
        FREE_LIST.as_ref().unwrap_unchecked().deref() as *const FreeList
    });
    unsafe {
        THREAD = Some(new);
    }
    THREAD_GUARD.with(|_| {});
    new
}

/*#[test]
fn test_thread() {
    use std::ptr::null;
    let thread = Thread::new(0, null());
    assert_eq!(thread.bucket, 0);
    assert_eq!(thread.bucket_size(), 1);
    assert_eq!(thread.index, 0);

    let thread = Thread::new(1, null());
    assert_eq!(thread.bucket, 1);
    assert_eq!(thread.bucket_size(), 2);
    assert_eq!(thread.index, 0);

    let thread = Thread::new(2, null());
    assert_eq!(thread.bucket, 1);
    assert_eq!(thread.bucket_size(), 2);
    assert_eq!(thread.index, 1);

    let thread = Thread::new(3, null());
    assert_eq!(thread.bucket, 2);
    assert_eq!(thread.bucket_size(), 4);
    assert_eq!(thread.index, 0);

    let thread = Thread::new(19, null());
    assert_eq!(thread.bucket, 4);
    assert_eq!(thread.bucket_size(), 16);
    assert_eq!(thread.index, 4);
}*/
