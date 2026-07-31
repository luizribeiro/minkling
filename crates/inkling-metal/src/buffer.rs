//! Memory a kernel reads and writes, and the CPU reads and writes with it.

use std::marker::PhantomData;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::device::{Device, MetalError};

/// One allocation the CPU and the GPU both address.
///
/// Apple silicon has no discrete VRAM: the two processors sit on the same
/// physical memory, and `StorageModeShared` is what says so. It is not a
/// compromise that trades speed for convenience — there is no copy it is
/// avoiding. `StorageModePrivate` would place the same bytes where only the GPU
/// can reach them, which buys a driver free to relayout the allocation at the
/// price of a blit in each direction; `StorageModeManaged` keeps a CPU copy and
/// a GPU copy in step and does not exist on this hardware at all.
///
/// The cost that remains is coherency, not transfer: what the GPU wrote is
/// visible to the CPU once the command buffer completes, and every dispatch
/// here waits for that before returning.
const STORAGE: MTLResourceOptions = MTLResourceOptions::StorageModeShared;

/// A type a [`Buffer`] can hold.
///
/// # Safety
///
/// Every bit pattern must be a valid `Self`, and `Self` must carry no padding.
/// A buffer's bytes are whatever the GPU last wrote there, and reading them
/// back reinterprets them as `Self` without looking.
pub unsafe trait Element: Copy {}

// SAFETY: all three are plain fixed-width numbers with no invalid bit pattern
// and no padding. f32 admits NaN, which is a value and not a trap.
unsafe impl Element for f32 {}
unsafe impl Element for u32 {}
unsafe impl Element for u8 {}

/// `len` elements of `T` in shared storage.
#[derive(Debug)]
pub struct Buffer<T> {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    element: PhantomData<T>,
}

impl Device {
    /// `len` elements, zeroed — which `newBufferWithLength:` guarantees.
    ///
    /// This is the allocation that costs nothing to fill: write through
    /// [`Buffer::as_mut_slice`] and the values land where the kernel will read
    /// them, with no staging buffer in between and no copy at all.
    pub fn zeroed<T: Element>(&self, len: usize) -> Result<Buffer<T>, MetalError> {
        // Checked because a product that wrapped would allocate fewer bytes
        // than `len` elements, and `as_slice` would then hand out a slice
        // running off the end of the allocation.
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(MetalError::Overflow {
                len,
                size: size_of::<T>(),
            })?;
        let raw = self
            .raw()
            .newBufferWithLength_options(bytes, STORAGE)
            .ok_or(MetalError::Allocation { bytes })?;
        Ok(Buffer {
            raw,
            len,
            element: PhantomData,
        })
    }

    /// [`Device::zeroed`] filled from a slice, for values that already exist
    /// somewhere else. That copy is the caller's, not Metal's.
    pub fn buffer<T: Element>(&self, values: &[T]) -> Result<Buffer<T>, MetalError> {
        let mut buffer = self.zeroed(values.len())?;
        buffer.as_mut_slice().copy_from_slice(values);
        Ok(buffer)
    }
}

#[expect(
    clippy::len_without_is_empty,
    reason = "the device refuses a zero-byte allocation, so no buffer is empty to ask about"
)]
impl<T: Element> Buffer<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    /// What the buffer holds now, read in place.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `contents` is the start of an allocation of `len` elements —
        // page-aligned, so aligned for any `T` — and `Element` says every byte
        // pattern in it reads back as a `T`. No dispatch can be writing it: a
        // kernel reaches a buffer only through `arg`, which borrows exclusively
        // for as long as the binding lives, and `Device::run` does not return
        // until the GPU is done with it. A dispatch that stopped waiting would
        // invalidate this, which is why `run` is synchronous by decision rather
        // than by omission.
        unsafe { std::slice::from_raw_parts(self.raw.contents().as_ptr().cast(), self.len) }
    }

    /// The same memory, to write into.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as [`Buffer::as_slice`], and the exclusive borrow of `self`
        // is what makes the returned slice the only reference to those bytes.
        unsafe { std::slice::from_raw_parts_mut(self.raw.contents().as_ptr().cast(), self.len) }
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }

    /// The buffer bound to one of a kernel's argument slots.
    ///
    /// Exclusive, because a kernel may write through any slot it is handed and
    /// nothing in the source string says which. The borrow is what keeps a CPU
    /// slice of the same bytes from outliving the call that writes them, and
    /// what keeps one buffer from being bound to two slots that write.
    pub fn arg(&mut self) -> Arg<'_> {
        Arg(&self.raw)
    }
}

/// A buffer bound to a kernel argument slot, from [`Buffer::arg`].
#[derive(Debug)]
pub struct Arg<'a>(&'a ProtocolObject<dyn MTLBuffer>);

impl Arg<'_> {
    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use crate::device::MetalError;
    use crate::testing::device;

    #[test]
    fn a_buffer_round_trips_its_values() {
        let Some(device) = device() else { return };
        let values: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5 - 3.0).collect();

        let buffer = device.buffer(&values).expect("the buffer allocates");

        assert_eq!(buffer.len(), values.len());
        assert_eq!(buffer.to_vec(), values);
    }

    /// The values a checkpoint actually holds: packed codes and scale bytes
    /// beside the floats they decode to.
    #[test]
    fn buffers_hold_the_integer_widths_too() {
        let Some(device) = device() else { return };

        let codes = device.buffer(&[0xffff_ffffu32, 0, 0x89ab_cdef]).unwrap();
        let scales = device.buffer(&[0x00u8, 0x7f, 0xff]).unwrap();

        assert_eq!(codes.to_vec(), [0xffff_ffff, 0, 0x89ab_cdef]);
        assert_eq!(scales.to_vec(), [0x00, 0x7f, 0xff]);
    }

    #[test]
    fn a_zeroed_buffer_starts_at_zero_and_takes_writes() {
        let Some(device) = device() else { return };

        let mut buffer = device.zeroed::<f32>(64).expect("the buffer allocates");
        assert_eq!(buffer.to_vec(), vec![0.0; 64]);

        buffer.as_mut_slice()[17] = 1.5;
        assert_eq!(buffer.as_slice()[17], 1.5);
    }

    /// An allocation past `maxBufferLength` has to fail as an error rather than
    /// as a null pointer walked into: the M2 weights are close enough to that
    /// ceiling that a tiling bug will hit it.
    #[test]
    fn an_impossible_allocation_is_an_error() {
        let Some(device) = device() else { return };

        let err = device
            .zeroed::<u8>(device.max_buffer_bytes() + 1)
            .expect_err("an over-large buffer is refused");

        assert!(err.to_string().contains("would not allocate"), "{err}");
    }

    /// Where the refusal comes from is the device, not a length check here, and
    /// it is what makes `Buffer::len` never zero. A driver that started
    /// honouring this would leave `as_slice` handing out an empty slice over a
    /// pointer nothing owns.
    #[test]
    fn a_zero_length_buffer_is_refused() {
        let Some(device) = device() else { return };

        let err = device
            .zeroed::<f32>(0)
            .expect_err("a buffer of nothing is refused");

        assert!(matches!(err, MetalError::Allocation { bytes: 0 }), "{err}");
    }

    /// Unreachable through any real weight, and the point is that it stays a
    /// refusal rather than a wrap: a wrapped length would allocate less than
    /// `len` elements and `as_slice` would run off the end of it.
    #[test]
    fn a_length_that_overflows_its_byte_count_is_refused() {
        let Some(device) = device() else { return };

        let err = device
            .zeroed::<f32>(usize::MAX / 2)
            .expect_err("the byte count overflows");

        assert!(matches!(err, MetalError::Overflow { size: 4, .. }), "{err}");
    }
}
