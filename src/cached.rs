#![allow(deprecated)]

use super::{IntoIter, IterMut, ThreadLocal};
use std::fmt;
use std::panic::UnwindSafe;
use std::usize;
use crate::Metadata;

/// Wrapper around [`ThreadLocal`].
///
/// This used to add a fast path for a single thread, however that has been
/// obsoleted by performance improvements to [`ThreadLocal`] itself.
#[deprecated(since = "1.1.0", note = "Use `ThreadLocal` instead")]
pub struct CachedThreadLocal<T: Send, M: Metadata = ()> {
    inner: ThreadLocal<T, M>,
}

impl<T: Send, M: Metadata> Default for CachedThreadLocal<T, M> {
    fn default() -> CachedThreadLocal<T, M> {
        CachedThreadLocal::new()
    }
}

impl<T: Send, M: Metadata> CachedThreadLocal<T, M> {
    /// Creates a new empty `CachedThreadLocal`.
    #[inline]
    pub fn new() -> CachedThreadLocal<T, M> {
        CachedThreadLocal {
            inner: ThreadLocal::new(),
        }
    }

    /// Returns the element for the current thread, if it exists.
    #[inline]
    pub fn get(&self) -> Option<&T> {
        self.inner.get()
    }

    /// Returns the element for the current thread, or creates it if it doesn't
    /// exist.
    #[inline]
    pub fn get_or<F>(&self, create: F) -> &T
    where
        F: FnOnce() -> T,
    {
        self.inner.get_or(create)
    }

    /// Returns the element for the current thread, or creates it if it doesn't
    /// exist. If `create` fails, that error is returned and no element is
    /// added.
    #[inline]
    pub fn get_or_try<F, E>(&self, create: F) -> Result<&T, E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        self.inner.get_or_try(create)
    }

    /// Returns a mutable iterator over the local values of all threads.
    ///
    /// Since this call borrows the `ThreadLocal` mutably, this operation can
    /// be done safely---the mutable borrow statically guarantees no other
    /// threads are currently accessing their associated values.
    #[inline]
    pub fn iter_mut(&mut self) -> CachedIterMut<T, M> {
        CachedIterMut {
            inner: self.inner.iter_mut(),
        }
    }

    /// Removes all thread-specific values from the `ThreadLocal`, effectively
    /// reseting it to its original state.
    ///
    /// Since this call borrows the `ThreadLocal` mutably, this operation can
    /// be done safely---the mutable borrow statically guarantees no other
    /// threads are currently accessing their associated values.
    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl<T: Send, M: Metadata> IntoIterator for CachedThreadLocal<T, M> {
    type Item = T;
    type IntoIter = CachedIntoIter<T, M>;

    fn into_iter(self) -> CachedIntoIter<T, M> {
        CachedIntoIter {
            inner: self.inner.into_iter(),
        }
    }
}

impl<'a, T: Send + 'a, M: Metadata> IntoIterator for &'a mut CachedThreadLocal<T, M> {
    type Item = &'a mut T;
    type IntoIter = CachedIterMut<'a, T, M>;

    fn into_iter(self) -> CachedIterMut<'a, T, M> {
        self.iter_mut()
    }
}

impl<T: Send + Default, M: Metadata> CachedThreadLocal<T, M> {
    /// Returns the element for the current thread, or creates a default one if
    /// it doesn't exist.
    pub fn get_or_default(&self) -> &T {
        self.get_or(T::default)
    }
}

impl<T: Send + fmt::Debug, M: Metadata> fmt::Debug for CachedThreadLocal<T, M> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ThreadLocal {{ local_data: {:?} }}", self.get())
    }
}

impl<T: Send + UnwindSafe, M: Metadata> UnwindSafe for CachedThreadLocal<T, M> {}

/// Mutable iterator over the contents of a `CachedThreadLocal`.
#[deprecated(since = "1.1.0", note = "Use `IterMut` instead")]
pub struct CachedIterMut<'a, T: Send + 'a, M: Metadata = ()> {
    inner: IterMut<'a, T, M>,
}

impl<'a, T: Send + 'a, M: Metadata> Iterator for CachedIterMut<'a, T, M> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, T: Send + 'a, M: Metadata> ExactSizeIterator for CachedIterMut<'a, T, M> {}

/// An iterator that moves out of a `CachedThreadLocal`.
#[deprecated(since = "1.1.0", note = "Use `IntoIter` instead")]
pub struct CachedIntoIter<T: Send, M: Metadata = ()> {
    inner: IntoIter<T, M>,
}

impl<T: Send, M: Metadata> Iterator for CachedIntoIter<T, M> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T: Send, M: Metadata> ExactSizeIterator for CachedIntoIter<T, M> {}
