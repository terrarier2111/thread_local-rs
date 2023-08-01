// Copyright 2017 Amanieu d'Antras
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::mutex::Mutex;
use crate::{Entry, BUCKETS, POINTER_WIDTH};
use rustc_hash::FxHashMap;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::cell::{Cell, UnsafeCell};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};
use std::mem::{transmute, ManuallyDrop};
use std::ops::{Deref, DerefMut};
use std::ptr::null;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
            free_list: unsafe {
                transmute::<Vec<Reverse<usize>>, BinaryHeap<Reverse<usize>>>(Vec::new())
            }, // FIXME: this is unsound, use const constructor, once its available!
        }
    }

    pub(crate) fn alloc(&mut self) -> usize {
        if let Some(id) = self.free_list.pop() {
            return id.0;
        }

        // `free_from` can't overflow as each thread takes up at least 2 bytes of memory and
        // thus we can't even have `usize::MAX / 2 + 1` threads.

        let id = self.free_from;
        self.free_from += 1;

        if (id + 1).is_power_of_two() {
            let (bucket, bucket_size, _) = id_into_parts(id);

            SHARED_IDS[bucket].set(alloc_shared(bucket_size));
        }

        id
    }

    pub(crate) fn free(&mut self, id: usize) {
        self.free_list.push(Reverse(id));
    }
}

impl Drop for ThreadIdManager {
    fn drop(&mut self) {
        let buckets = POINTER_WIDTH as usize
            - self.free_from.next_power_of_two().leading_zeros() as usize
            + 1;
        for bucket in 0..buckets {
            let ptr = SHARED_IDS[bucket].get().cast_mut();
            unsafe {
                dealloc(
                    ptr.cast(),
                    Layout::array::<AtomicUsize>(1 << bucket).unwrap_unchecked(),
                );
            }
        }
    }
}

fn alloc_shared(size: usize) -> *const AtomicUsize {
    let ret = unsafe { alloc_zeroed(Layout::array::<AtomicUsize>(size).unwrap()) };
    if ret.is_null() {
        panic!("There was an error allocating shared counters!");
    }
    ret.cast::<AtomicUsize>().cast_const()
}

static THREAD_ID_MANAGER: Mutex<ThreadIdManager> = Mutex::new_empty(ThreadIdManager::new());

pub(crate) static SHARED_IDS: [PtrCell<AtomicUsize>; BUCKETS] =
    { unsafe { transmute([null::<AtomicUsize>(); BUCKETS]) } };

#[inline]
pub(crate) unsafe fn shared_id_ptr(id: usize) -> *const AtomicUsize {
    let (bucket, _, index) = id_into_parts(id);
    SHARED_IDS[bucket].get().add(index)
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

#[derive(Copy, Clone)]
struct ThreadWrapper {
    self_ptr: *const Thread,
    thread: Thread,
}

impl ThreadWrapper {
    
    #[inline]
    fn new(id: usize, free_list: *const FreeList) -> Self {
        Self {
            self_ptr: unsafe { (THREAD.get() as usize + memoffset::offset_of!(ThreadWrapper, thread)) as *const Thread },
            thread: Thread::new(id, free_list),
        }
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
        let outstanding_shared = unsafe { shared_id_ptr(self.id) };
        let mut outstanding = 0;
        for entry in free_list.iter() {
            // sum up all the "failed" cleanups
            if unsafe { !entry.1.cleanup(entry.0 .0) } {
                outstanding += 1;
            }
        }

        if outstanding > 0 {
            // store the number of outstanding references
            unsafe { outstanding_shared.as_ref().unwrap_unchecked() }
                .store(outstanding, Ordering::Release);
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
static THREAD: UnsafeCell<ThreadWrapper> = UnsafeCell::new(ThreadWrapper {
    self_ptr: null(),
    thread: Thread {
        bucket: 0,
        index: 0,
        free_list: null(),
    },
});
#[thread_local]
static FREE_LIST: UnsafeCell<Option<ManuallyDrop<FreeList>>> = UnsafeCell::new(None);
thread_local! { static THREAD_GUARD: ThreadGuard = const { ThreadGuard }; }

// Guard to ensure the thread ID is released on thread exit.
struct ThreadGuard;

impl Drop for ThreadGuard {
    fn drop(&mut self) {
        unsafe {
            let ptr = FREE_LIST.get();
            {
                let reference = ptr.as_ref().unwrap_unchecked().as_ref().unwrap_unchecked().deref();
                // first clean up all entries in the freelist
                reference.cleanup();
            }
            // ... then clean up the freelist itself.
            let ptr = ptr.as_mut().unwrap_unchecked().as_mut().unwrap_unchecked().deref_mut() as *mut FreeList;
            ptr.drop_in_place();
        }
    }
}

/// Returns a thread ID for the current thread, allocating one if needed.
#[inline]
pub(crate) fn get() -> Thread {
    let ptr = unsafe { THREAD.get().as_ref().unwrap_unchecked().self_ptr };
    if !ptr.is_null() {
        unsafe { ptr.read() }
    } else {
        get_slow()
    }
}

/// Out-of-line slow path for allocating a thread ID.
#[cold]
fn get_slow() -> Thread {
    let tid = THREAD_ID_MANAGER.lock().alloc();
    unsafe {
        *FREE_LIST.get() = Some(ManuallyDrop::new(FreeList::new(tid)));
    }
    let new = ThreadWrapper::new(tid, unsafe {
        FREE_LIST.get().as_ref().unwrap_unchecked().as_ref().unwrap_unchecked().deref() as *const FreeList
    });
    unsafe {
        *THREAD.get() = new;
    }
    THREAD_GUARD.with(|_| {});
    new.thread
}

#[test]
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
}
