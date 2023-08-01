pub mod fillable {
    use crossbeam_utils::Backoff;
    use std::cell::UnsafeCell;
    use std::mem;
    use std::ops::{Deref, DerefMut};
    use std::sync::atomic::{AtomicU8, Ordering};

    /// A mutex optimized for little contention.
    pub(crate) struct Mutex<T> {
        guard: AtomicU8,
        val: UnsafeCell<T>,
    }

    const GUARD_UNLOCKED_FULL: u8 = 0;
    const GUARD_UNLOCKED_EMPTY: u8 = 1;
    const GUARD_LOCKED: u8 = 2;

    impl<T> Mutex<T> {
        pub const fn new_full(val: T) -> Self {
            Self {
                guard: AtomicU8::new(GUARD_UNLOCKED_FULL),
                val: UnsafeCell::new(val),
            }
        }

        pub const fn new_empty(val: T) -> Self {
            Self {
                guard: AtomicU8::new(GUARD_UNLOCKED_EMPTY),
                val: UnsafeCell::new(val),
            }
        }

        pub fn is_full(&self) -> bool {
            self.guard.load(Ordering::Acquire) == GUARD_UNLOCKED_FULL
        }

        pub fn is_maybe_full(&self) -> bool {
            self.guard.load(Ordering::Acquire) != GUARD_UNLOCKED_EMPTY
        }

        pub fn is_empty(&self) -> bool {
            self.guard.load(Ordering::Acquire) == GUARD_UNLOCKED_EMPTY
        }

        pub fn is_maybe_empty(&self) -> bool {
            self.guard.load(Ordering::Acquire) != GUARD_UNLOCKED_FULL
        }

        pub fn lock(&self) -> MutexGuardDyn<T> {
            match self.guard
                .compare_exchange(GUARD_UNLOCKED_FULL, GUARD_LOCKED, Ordering::Acquire, Ordering::Relaxed) {
                Ok(_) => return MutexGuardDyn(self, true),
                Err(err) => {
                    if err == GUARD_UNLOCKED_EMPTY {
                        if self.guard
                            .compare_exchange(GUARD_UNLOCKED_EMPTY, GUARD_LOCKED, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                            return MutexGuardDyn(self, false);
                        }
                    }
                }
            }

            let backoff = Backoff::new();
            loop {
                match self.guard.compare_exchange(GUARD_UNLOCKED_FULL, GUARD_LOCKED, Ordering::Acquire, Ordering::Acquire) {
                    Ok(_) => {
                        return MutexGuardDyn(self, true);
                    }
                    Err(err) => {
                        if err == GUARD_UNLOCKED_EMPTY {
                            if self.guard
                                .compare_exchange(GUARD_UNLOCKED_EMPTY, GUARD_LOCKED, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                                return MutexGuardDyn(self, false);
                            }
                        }
                        backoff.snooze();
                    }
                }
            }
        }

        pub fn lock_full(&self) -> Option<MutexGuard<T, true>> {
            match self.guard
                .compare_exchange(GUARD_UNLOCKED_FULL, GUARD_LOCKED, Ordering::Acquire, Ordering::Relaxed) {
                Ok(_) => return Some(MutexGuard(self)),
                Err(err) => {
                    if err == GUARD_UNLOCKED_EMPTY {
                        return None;
                    }
                }
            }

            let backoff = Backoff::new();
            loop {
                match self.guard.compare_exchange(GUARD_UNLOCKED_FULL, GUARD_LOCKED, Ordering::Acquire, Ordering::Acquire) {
                    Ok(_) => {
                        return Some(MutexGuard(self));
                    }
                    Err(err) => {
                        if err == GUARD_UNLOCKED_EMPTY {
                            return None;
                        }
                        backoff.snooze();
                    }
                }
            }
        }

        pub fn lock_empty(&self) -> Option<MutexGuard<T, false>> {
            match self.guard
                .compare_exchange(GUARD_UNLOCKED_EMPTY, GUARD_LOCKED, Ordering::Acquire, Ordering::Relaxed) {
                Ok(_) => return Some(MutexGuard(self)),
                Err(err) => {
                    if err == GUARD_UNLOCKED_FULL {
                        return None;
                    }
                }
            }

            let backoff = Backoff::new();
            loop {
                match self.guard.compare_exchange(GUARD_UNLOCKED_EMPTY, GUARD_LOCKED, Ordering::Acquire, Ordering::Acquire) {
                    Ok(_) => {
                        return Some(MutexGuard(self));
                    }
                    Err(err) => {
                        if err == GUARD_UNLOCKED_FULL {
                            return None;
                        }
                        backoff.snooze();
                    }
                }
            }
        }

        #[inline]
        pub fn try_lock(&self) -> Option<MutexGuardDyn<T>> {
            match self.guard
                .compare_exchange(GUARD_UNLOCKED_FULL, GUARD_LOCKED, Ordering::Acquire, Ordering::Relaxed) {
                Ok(_) => Some(MutexGuardDyn(self, true)),
                Err(err) => {
                    if err == GUARD_UNLOCKED_EMPTY {
                        if self.guard
                            .compare_exchange(GUARD_UNLOCKED_EMPTY, GUARD_LOCKED, Ordering::Acquire, Ordering::Relaxed).is_err() {
                            return None;
                        }
                        return Some(MutexGuardDyn(self, false));
                    }
                    None
                }
            }
        }

        #[inline]
        pub fn try_lock_full(&self) -> Option<Option<MutexGuard<T, true>>> {
            let val = self.guard.load(Ordering::Relaxed);
            if val == GUARD_UNLOCKED_EMPTY {
                return Some(None);
            }
            if val == GUARD_LOCKED {
                return None;
            }
            match self.guard
                .compare_exchange(GUARD_UNLOCKED_FULL, GUARD_LOCKED, Ordering::Acquire, Ordering::Relaxed) {
                Ok(_) => Some(Some(MutexGuard(self))),
                Err(err) => {
                    if err == GUARD_UNLOCKED_EMPTY {
                        return Some(None);
                    }
                    None
                }
            }
        }

        #[inline]
        pub fn try_lock_empty(&self) -> Option<Option<MutexGuard<T, false>>> {
            let val = self.guard.load(Ordering::Relaxed);
            if val == GUARD_UNLOCKED_FULL {
                return Some(None);
            }
            if val == GUARD_LOCKED {
                return None;
            }
            match self.guard
                .compare_exchange(GUARD_UNLOCKED_EMPTY, GUARD_LOCKED, Ordering::Acquire, Ordering::Relaxed) {
                Ok(_) => Some(Some(MutexGuard(self))),
                Err(err) => {
                    if err == GUARD_UNLOCKED_FULL {
                        return Some(None);
                    }
                    None
                }
            }
        }
    }

    unsafe impl<T: Send> Send for Mutex<T> {}

    unsafe impl<T: Sync> Sync for Mutex<T> {}

    pub(crate) struct MutexGuard<'a, T, const FULL: bool>(&'a Mutex<T>);

    impl<'a, T, const FULL: bool> Deref for MutexGuard<'a, T, FULL> {
        type Target = T;

        #[inline(always)]
        fn deref(&self) -> &'a Self::Target {
            unsafe { &*self.0.val.get() }
        }
    }

    impl<'a, T, const FULL: bool> DerefMut for MutexGuard<'a, T, FULL> {
        #[inline(always)]
        fn deref_mut(&mut self) -> &'a mut Self::Target {
            unsafe { &mut *self.0.val.get() }
        }
    }

    impl<'a, T> MutexGuard<'a, T, true> {
        #[inline(always)]
        pub fn is_full(&self) -> bool {
            true
        }

        #[inline(always)]
        pub fn is_empty(&self) -> bool {
            false
        }

        #[inline]
        pub fn empty(self) -> MutexGuard<'a, T, false> {
            let ret = MutexGuard(self.0);
            mem::forget(self);
            ret
        }
    }

    impl<'a, T> MutexGuard<'a, T, false> {
        #[inline(always)]
        pub fn is_full(&self) -> bool {
            false
        }

        #[inline(always)]
        pub fn is_empty(&self) -> bool {
            true
        }

        #[inline]
        pub fn fill(self) -> MutexGuard<'a, T, true> {
            let ret = MutexGuard(self.0);
            mem::forget(self);
            ret
        }
    }

    impl<T, const FULL: bool> Drop for MutexGuard<'_, T, FULL> {
        fn drop(&mut self) {
            let guard = if FULL {
                GUARD_UNLOCKED_FULL
            } else {
                GUARD_UNLOCKED_EMPTY
            };
            self.0.guard.store(guard, Ordering::Release);
        }
    }

    pub(crate) struct MutexGuardDyn<'a, T>(&'a Mutex<T>, bool);

    impl<'a, T> MutexGuardDyn<'a, T> {
        #[inline(always)]
        pub fn is_full(&self) -> bool {
            self.1
        }

        #[inline(always)]
        pub fn is_empty(&self) -> bool {
            !self.1
        }
    }

    impl<'a, T> Deref for MutexGuardDyn<'a, T> {
        type Target = T;

        #[inline(always)]
        fn deref(&self) -> &'a Self::Target {
            unsafe { &*self.0.val.get() }
        }
    }

    impl<'a, T> DerefMut for MutexGuardDyn<'a, T> {
        #[inline(always)]
        fn deref_mut(&mut self) -> &'a mut Self::Target {
            unsafe { &mut *self.0.val.get() }
        }
    }

    impl<'a, T> MutexGuardDyn<'a, T> {
        #[inline]
        pub fn empty(self) -> MutexGuard<'a, T, false> {
            let ret = MutexGuard(self.0);
            mem::forget(self);
            ret
        }

        #[inline]
        pub fn fill(self) -> MutexGuard<'a, T, true> {
            let ret = MutexGuard(self.0);
            mem::forget(self);
            ret
        }
    }

    impl<T> Drop for MutexGuardDyn<'_, T> {
        fn drop(&mut self) {
            let guard = if self.1 {
                GUARD_UNLOCKED_FULL
            } else {
                GUARD_UNLOCKED_EMPTY
            };
            self.0.guard.store(guard, Ordering::Release);
        }
    }
}

pub mod simple {
    use crossbeam_utils::Backoff;
    use std::cell::UnsafeCell;
    use std::intrinsics::{likely, unlikely};
    use std::ops::{Deref, DerefMut};
    use std::sync::atomic::{AtomicBool, Ordering};

    /// A mutex optimized for little contention.
    pub(crate) struct Mutex<T> {
        guard: AtomicBool,
        data: UnsafeCell<T>,
    }

    impl<T> Mutex<T> {
        pub const fn new(val: T) -> Self {
            Self {
                guard: AtomicBool::new(false),
                data: UnsafeCell::new(val),
            }
        }

        pub fn lock(&self) -> MutexGuard<'_, T> {
            if likely(
                self.guard
                    .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok(),
            ) {
                return MutexGuard(self);
            }

            let backoff = Backoff::new();
            while self
                .guard
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                backoff.snooze();
            }
            MutexGuard(self)
        }

        #[inline]
        pub fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
            if unlikely(
                self.guard
                    .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                    .is_err(),
            ) {
                return None;
            }
            Some(MutexGuard(self))
        }
    }

    unsafe impl<T: Send> Send for Mutex<T> {}
    unsafe impl<T: Sync> Sync for Mutex<T> {}

    pub(crate) struct MutexGuard<'a, T>(&'a Mutex<T>);

    impl<'a, T> Deref for MutexGuard<'a, T> {
        type Target = T;

        #[inline(always)]
        fn deref(&self) -> &Self::Target {
            unsafe { &*self.0.data.get() }
        }
    }

    impl<'a, T> DerefMut for MutexGuard<'a, T> {
        #[inline(always)]
        fn deref_mut(&mut self) -> &mut Self::Target {
            unsafe { &mut *self.0.data.get() }
        }
    }

    impl<'a, T> Drop for MutexGuard<'a, T> {
        #[inline]
        fn drop(&mut self) {
            self.0.guard.store(false, Ordering::Release);
        }
    }
}
