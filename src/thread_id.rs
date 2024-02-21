// Copyright 2017 Amanieu d'Antras
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{Entry, BUCKETS, POINTER_WIDTH};
use rustc_hash::{FxHashMap, FxHasher};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::cell::{Cell, UnsafeCell};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::mem::{transmute, ManuallyDrop};
use std::ops::{Deref, DerefMut};
use std::ptr::null;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use crate::mutex::{fillable, simple};

/// Thread ID manager which allocates thread IDs. It attempts to aggressively
/// reuse thread IDs where possible to avoid cases where a ThreadLocal grows
/// indefinitely when it is used by many short-lived threads.
pub(crate) struct ThreadIdManager {
    free_from: AtomicUsize,
    free_list: fillable::Mutex<BinaryHeap<Reverse<usize>>>,
}

impl ThreadIdManager {
    const fn new() -> Self {
        Self {
            free_from: AtomicUsize::new(0),
            free_list: fillable::Mutex::new_empty(BinaryHeap::new()),
        }
    }

    pub(crate) fn alloc(&self) -> usize {
        // FIXME: this isn't smart! turn this implicit cmp_xchg into an atomic load!
        if let Some(Some(mut guard)) = self.free_list.try_lock_full() {
            if let Some(id) = guard.pop() {
                return id.0;
            } else {
                guard.empty();
            }
        }

        // `free_from` can't overflow as each thread takes up at least 2 bytes of memory and
        // thus we can't even have `usize::MAX / 2 + 1` threads.
        let id = self.free_from.fetch_add(1, Ordering::Relaxed);

        if (id + 1).is_power_of_two() {
            let (bucket, bucket_size, _) = id_into_parts(id);

            SHARED_IDS[bucket].set(alloc_shared(bucket_size));
        }

        id
    }

    pub(crate) fn free(&self, id: usize) {
        self.free_list.lock().push(Reverse(id));
    }
}

impl Drop for ThreadIdManager {
    fn drop(&mut self) {
        let buckets = POINTER_WIDTH as usize
            - self.free_from.get_mut().next_power_of_two().leading_zeros() as usize
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

static THREAD_ID_MANAGER: ThreadIdManager = ThreadIdManager::new();

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

struct LocalData {
    thread: ThreadWrapper,
    free_list: ManuallyDrop<FreeList>,
}

impl LocalData {

    #[inline]
    fn new(id: usize) -> Self {
        let base_addr = THREAD.get() as usize;
        Self {
            thread: ThreadWrapper {
                self_ptr: unsafe { (base_addr + memoffset::offset_of!(ThreadWrapper, thread)) as *const Thread },
                thread: Thread::new(id),
            },
            free_list: ManuallyDrop::new(FreeList::new(id)),
        }
    }

}

#[derive(Clone, Copy)]
struct ThreadWrapper {
    self_ptr: *const Thread,
    thread: Thread,
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
}

impl Thread {
    fn new(id: usize) -> Self {
        let (bucket, _, index) = id_into_parts(id);

        Self {
            bucket,
            index,
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
    THREAD_ID_MANAGER.free(id);
}

const HASHER: BuildHasherDefault<FxHasher> = unsafe { transmute(()) }; // FIXME: this is unsound, use proper const constructor once its available!

pub(crate) struct FreeList {
    id: Cell<usize>,
    pub(crate) dropping: AtomicBool,
    pub(crate) free_list: simple::Mutex<FxHashMap<SendSyncPtr<Entry<()>>, EntryData>>,
}

impl FreeList {
    const fn new(id: usize) -> Self {
        Self {
            id: Cell::new(id),
            dropping: AtomicBool::new(false),
            free_list: simple::Mutex::new(FxHashMap::with_hasher(HASHER)),
        }
    }

    fn cleanup(&self) {
        self.dropping.store(true, Ordering::Release);
        let free_list = self.free_list.lock();
        let outstanding_shared = unsafe { shared_id_ptr(self.id.get()) };
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
            THREAD_ID_MANAGER.free(self.id.get());
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
static THREAD: UnsafeCell<LocalData> = UnsafeCell::new(LocalData {
    thread: ThreadWrapper {
        self_ptr: null(),
        thread: Thread {
            bucket: 0,
            index: 0,
        },
    },
    free_list: ManuallyDrop::new(FreeList::new(0)), // use 0 as a decoy
});
thread_local! { static THREAD_GUARD: ThreadGuard = const { ThreadGuard }; }

// Guard to ensure the thread ID is released on thread exit.
struct ThreadGuard;

impl Drop for ThreadGuard {
    fn drop(&mut self) {
        {
            // first clean up all entries in the freelist
            unsafe { (&*THREAD.get()).free_list.cleanup(); }
        }
        // ... then clean up the freelist itself.
        unsafe { (&(&*THREAD.get()).free_list as *const ManuallyDrop<FreeList>).cast::<FreeList>().read(); }
    }
}

/// Returns a thread ID for the current thread, allocating one if needed.
#[inline]
pub(crate) fn get() -> (Thread, *const FreeList) {
    let ptr = unsafe { *THREAD.get().cast::<ThreadWrapper>().byte_add(memoffset::offset_of!(LocalData, thread)) };
    if !ptr.self_ptr.is_null() {
        (unsafe { ptr.self_ptr.read() }, unsafe { ptr.self_ptr.byte_offset((memoffset::offset_of!(LocalData, free_list) as isize) - (memoffset::offset_of!(LocalData, thread) as isize) - (memoffset::offset_of!(ThreadWrapper, thread) as isize)).cast::<FreeList>() })
    } else {
        get_slow()
    }
}

/// Out-of-line slow path for allocating a thread ID.
#[cold]
fn get_slow() -> (Thread, *const FreeList) {
    let tid = THREAD_ID_MANAGER.alloc();
    let new = LocalData::new(tid);
    let ret = new.thread.thread.clone();
    let self_ptr = new.thread.self_ptr;
    unsafe {
        *THREAD.get() = new;
    }
    THREAD_GUARD.with(|_| {});
    (ret, unsafe { self_ptr.byte_offset((memoffset::offset_of!(LocalData, free_list) as isize) - (memoffset::offset_of!(LocalData, thread) as isize) - (memoffset::offset_of!(ThreadWrapper, thread) as isize)).cast::<FreeList>() })
}

#[test]
fn test_thread() {
    let thread = Thread::new(0);
    assert_eq!(thread.bucket, 0);
    assert_eq!(thread.bucket_size(), 1);
    assert_eq!(thread.index, 0);

    let thread = Thread::new(1);
    assert_eq!(thread.bucket, 1);
    assert_eq!(thread.bucket_size(), 2);
    assert_eq!(thread.index, 0);

    let thread = Thread::new(2);
    assert_eq!(thread.bucket, 1);
    assert_eq!(thread.bucket_size(), 2);
    assert_eq!(thread.index, 1);

    let thread = Thread::new(3);
    assert_eq!(thread.bucket, 2);
    assert_eq!(thread.bucket_size(), 4);
    assert_eq!(thread.index, 0);

    let thread = Thread::new(19);
    assert_eq!(thread.bucket, 4);
    assert_eq!(thread.bucket_size(), 16);
    assert_eq!(thread.index, 4);
}
