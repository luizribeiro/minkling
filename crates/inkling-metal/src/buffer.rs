//! Memory a kernel reads and writes, and the CPU reads and writes with it.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;

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

/// The granularity `newBufferWithBytesNoCopy:` takes its bounds in, which on
/// Apple silicon is 16 KiB.
///
/// Asked of the kernel rather than written down, because a wrap whose bounds are
/// not the page size Metal is using raises an Objective-C exception — which
/// unwinds through no Rust destructor and takes the process with it.
fn page() -> usize {
    // SAFETY: `sysconf` reads a constant the kernel published at exec, and
    // `_SC_PAGESIZE` is one every POSIX system defines rather than an optional
    // limit that can come back as -1.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// A type a [`Buffer`] can hold.
///
/// # Safety
///
/// Every bit pattern must be a valid `Self`, and `Self` must carry no padding.
/// A buffer's bytes are whatever the GPU last wrote there, and reading them
/// back reinterprets them as `Self` without looking.
///
/// `Self` must also have a size, which a zero-sized type would satisfy the
/// paragraph above while not having: [`Device::wrap`] divides a byte count by it
/// to say how many elements a range of pages holds.
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
        let buffer = Buffer::of(self.raw().newBufferWithLength_options(bytes, STORAGE), len)?;
        self.allocated();
        Ok(buffer)
    }

    /// [`Device::zeroed`] filled from a slice, for values that already exist
    /// somewhere else. That copy is the caller's, not Metal's.
    pub fn buffer<T: Element>(&self, values: &[T]) -> Result<Buffer<T>, MetalError> {
        let mut buffer = self.zeroed(values.len())?;
        buffer.as_mut_slice().copy_from_slice(values);
        Ok(buffer)
    }

    /// Bytes this process already holds, given to the GPU where they lie.
    ///
    /// **No copy at all, and no residency of its own.** Wrapping a gibibyte of
    /// a mapped checkpoint takes about 50 microseconds against the 130
    /// milliseconds copying it takes, and leaves the resident set where it was:
    /// what the GPU then reads through it are the file's own pages, faulted by
    /// the same demand paging the CPU path faults them by. That is what makes a
    /// bank of experts nobody routed to cost nothing to have wrapped.
    ///
    /// # Safety
    ///
    /// `bytes` must lie in pages that are mapped for their whole length and
    /// stay mapped for the [`Mapped`]'s life — a slice of a mapping or of a
    /// whole-page allocation, not of a `Vec`. What is wrapped is the *pages*
    /// `bytes` falls in, which reach out to a page boundary either side of it,
    /// and Metal reads that whole range as one.
    ///
    /// Nothing may write those pages while a dispatch bound to the buffer is
    /// running. **The borrow does not say this** and cannot: a `&[u8]` rules out
    /// writes through Rust references, and the memory is reachable by raw
    /// pointer and by any other process holding the same file. It is the
    /// assumption [`Checkpoint`](inkling_core::Checkpoint) already maps under —
    /// that a checkpoint is a read-only artefact for the life of the process —
    /// and this inherits it rather than strengthening it.
    pub unsafe fn wrap<'a, T: Element>(
        &self,
        bytes: &'a [u8],
    ) -> Result<Mapped<'a, T>, MetalError> {
        let page = page();
        let offset = bytes.as_ptr() as usize % page;
        if offset % size_of::<T>() != 0 {
            return Err(MetalError::Misaligned {
                offset,
                size: size_of::<T>(),
            });
        }

        let len = (offset + bytes.len()).div_ceil(page) * page;
        // SAFETY: stepping back to the start of the page `bytes` begins in. A
        // mapping starts on a page boundary, so a pointer within one cannot be
        // rounded down out of it — which is what this needs to stay inside the
        // allocation it was derived from.
        let base = unsafe { NonNull::from(bytes).cast::<c_void>().byte_sub(offset) };
        // SAFETY: `base` is page-aligned and `len` a whole number of pages,
        // which is what this call raises an Objective-C exception for the
        // absence of. The range is the caller's mapping, which the contract
        // above says outlives the buffer. The deallocator is `None`, which is
        // what says the memory is not Metal's to free.
        let raw = unsafe {
            self.raw()
                .newBufferWithBytesNoCopy_length_options_deallocator(base, len, STORAGE, None)
        };

        Ok(Mapped {
            buffer: Buffer::of(raw, len / size_of::<T>())?,
            offset: offset / size_of::<T>(),
            wrapped: PhantomData,
        })
    }
}

impl<T: Element> Buffer<T> {
    /// What the device answered, as an error if it answered nothing.
    ///
    /// `len` is in elements and the error is in bytes, because what a driver
    /// refuses is a size and what a caller asked for is a count.
    fn of(
        raw: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
        len: usize,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            raw: raw.ok_or(MetalError::Allocation {
                bytes: len * size_of::<T>(),
            })?,
            len,
            element: PhantomData,
        })
    }
}

/// Pages of this process's address space the GPU reads in place, from
/// [`Device::wrap`].
///
/// The wrap is of pages and a checkpoint's tensors are not page-aligned — the
/// shard header is not padded, so every tensor in this checkpoint starts one
/// byte past a word — so what is wrapped is the pages the tensor falls in and
/// [`Mapped::offset`] is where in them it starts.
///
/// Read-only, and that is not a convention: the pages are a read-only mapping,
/// so a kernel that wrote through this binding would take the whole process
/// down with a bus error. Nothing here hands out a mutable slice, and the one
/// thing it does hand out — an [`Arg`] — belongs in a slot the kernel declares
/// `device const`.
#[derive(Debug)]
pub struct Mapped<'a, T> {
    buffer: Buffer<T>,
    offset: usize,
    wrapped: PhantomData<&'a [u8]>,
}

#[expect(
    clippy::len_without_is_empty,
    reason = "a wrap is a whole number of pages and the device refuses a zero-byte one, so no \
              wrap is empty to ask about"
)]
impl<T: Element> Mapped<'_, T> {
    /// Where the wrapped bytes start, in elements of `T` from the buffer's own
    /// start.
    ///
    /// A kernel is handed this and adds it, rather than being bound a buffer at
    /// an offset: what Metal requires of a binding offset varies by device, and
    /// what an added index requires is only that the elements line up — which
    /// [`Device::wrap`] refuses the wrap for the absence of.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// How many elements of `T` the wrapped pages hold, which is the tensor
    /// rounded out to its page bounds rather than the tensor.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn arg(&mut self) -> Arg<'_> {
        self.buffer.arg()
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
        Arg::Bound(&self.raw)
    }
}

/// Bytes a kernel reads, however they got where they are: a copy the device
/// made, or the checkpoint's own pages wrapped where they lie.
///
/// One type rather than two paths through every caller, because the difference
/// between them is exactly one number — where in the binding the tensor starts,
/// which a copy puts at zero and a wrap puts wherever the page boundary fell.
/// A kernel is handed that number either way and cannot tell which it got.
#[derive(Debug)]
pub enum Bytes<'a> {
    Copied(Buffer<u8>),
    Mapped(Mapped<'a, u8>),
}

impl Bytes<'_> {
    /// Where the tensor starts, in bytes from the binding's own start.
    pub fn offset(&self) -> usize {
        match self {
            Self::Copied(_) => 0,
            Self::Mapped(mapped) => mapped.offset(),
        }
    }

    pub fn arg(&mut self) -> Arg<'_> {
        match self {
            Self::Copied(buffer) => buffer.arg(),
            Self::Mapped(mapped) => mapped.arg(),
        }
    }
}

/// Where a dispatch's `[rows, groups * width]` output goes: the `[groups,
/// stride, width]` region of `out` whose rows start at `base`.
///
/// **The transpose is the write's own indexing.** What a projection or a
/// convolution produces is group-major within a row — the heads of a token side
/// by side — and what the attention step reads is row-major within a group, over
/// a span with room for more rows than the sequence has put in it. A kernel
/// handed these three numbers addresses all of that as it writes, so the
/// [`split_heads`](inkling_core::split_heads) between a projection and the step
/// is not a pass over a tensor and the append into the span is not a copy.
///
/// A dispatch with nothing to scatter passes `groups: 1`, `stride: rows` and
/// `base: 0`, which is the identity: the rows land where they were computed.
#[derive(Debug)]
pub struct Landing<'a> {
    pub out: &'a mut Buffer<f32>,
    /// How many groups of equal width each row divides into.
    pub groups: usize,
    /// Rows each group has room for in `out`, which is the stride between two
    /// groups.
    pub stride: usize,
    /// Where in those rows this call's own start.
    pub base: usize,
}

impl Landing<'_> {
    /// That `rows` rows of `groups` groups of `width` fit where this says they
    /// go.
    ///
    /// Checked here rather than by each kernel that writes one, because what
    /// these three numbers have to agree with is the buffer they index — and a
    /// GPU write past a buffer's end is memory somebody else owns rather than a
    /// fault. `width` is the caller's because only it knows what a row of its
    /// own is.
    pub fn fits(&self, rows: usize, width: usize) {
        assert!(self.groups > 0, "a row has groups");
        assert!(
            self.base + rows <= self.stride,
            "{rows} rows at {} past a landing of {}",
            self.base,
            self.stride
        );
        assert_eq!(
            self.out.len(),
            self.groups * self.stride * width,
            "the landing against the shape it is written under"
        );
    }
}

/// What fills one of a kernel's argument slots: an allocation the dispatch
/// reads, or bytes copied into the command buffer as it is encoded.
///
/// The second is what a dispatch's shape is. A shape is a dozen `uint`s that
/// describe one call and are read once, so what an allocation would buy it is a
/// lifetime it has no use for at a price a step pays 869 times.
#[derive(Debug)]
pub enum Arg<'a> {
    /// An allocation, from [`Buffer::arg`]. The command buffer retains it, so it
    /// outlives the encoding whatever the caller does.
    Bound(&'a ProtocolObject<dyn MTLBuffer>),
    /// Bytes `setBytes:length:atIndex:` copies where the GPU will read them,
    /// from [`Inline::arg`]. **Copied as the dispatch is encoded**, which is
    /// what makes them safe to overwrite immediately afterwards and what lets
    /// two calls of different shapes share a command buffer.
    Inline(&'a [u8]),
}

/// The most bytes `setBytes:length:atIndex:` takes, which every Apple GPU
/// family states as 4 KiB.
const INLINE_BYTES: usize = 4096;

/// Values a dispatch reads and no dispatch writes, put where it can read them:
/// in the command buffer when they fit, and in an allocation when they do not.
///
/// **One type, not a threshold repeated at every call site.** A shape is always
/// a few dozen bytes, but an expert list is one `uint` a row — six of them on a
/// decode step and 4614 on a 769-token prefill — so the same argument is inline
/// at one call and an allocation at the next, and nothing above here wants to
/// know which.
#[derive(Debug)]
pub enum Inline<'a, T> {
    Bytes(&'a [T]),
    Buffered(Buffer<T>),
}

impl Device {
    /// `values` where a dispatch can read them, without an allocation if they
    /// are small enough to travel in the command buffer.
    ///
    /// For arguments a kernel declares `constant` or `device const` and nothing
    /// else: what this hands out is a *copy* taken at encode time, so a slot the
    /// kernel writes through would be writing bytes nobody reads back.
    pub fn inline<'v, T: Element>(&self, values: &'v [T]) -> Result<Inline<'v, T>, MetalError> {
        match (1..=INLINE_BYTES).contains(&size_of_val(values)) {
            true => Ok(Inline::Bytes(values)),
            // Both the values too wide for the command buffer and the empty
            // slice, which the device refuses to allocate for — the refusal
            // this has always answered a dispatch over nothing with.
            false => self.buffer(values).map(Inline::Buffered),
        }
    }
}

impl<T: Element> Inline<'_, T> {
    /// The values in one of a kernel's argument slots.
    ///
    /// Exclusive for the reason [`Buffer::arg`] is, and only for the reason
    /// [`Buffer::arg`] is: the allocated arm is a buffer like any other. The
    /// inline arm needs nothing of the borrow.
    pub fn arg(&mut self) -> Arg<'_> {
        match self {
            Self::Bytes(values) => Arg::Inline(
                // SAFETY: `Element` says `T` carries no padding, so every byte
                // of the slice is initialised, and `u8` is aligned for any
                // address. The bytes are read for the length of the borrow and
                // copied out of it by `setBytes:`.
                unsafe {
                    std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(*values))
                },
            ),
            Self::Buffered(buffer) => buffer.arg(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{INLINE_BYTES, Inline, page};
    use crate::device::MetalError;
    use crate::kernel::Grid;
    use crate::testing::{SAXPY, SAXPY_ENTRY, device};

    /// Pages of this process's own, standing in for a checkpoint's mapping: the
    /// one thing [`Device::wrap`] needs that a `Vec` cannot give is that the
    /// bytes either side of the slice, out to the page bounds, are mapped.
    struct Pages {
        raw: *mut libc::c_void,
        len: usize,
    }

    impl Pages {
        fn new(pages: usize) -> Self {
            let len = pages * page();
            // SAFETY: an anonymous private mapping of a whole number of pages,
            // which is the shape `mmap` is documented to return page-aligned.
            let raw = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            assert_ne!(raw, libc::MAP_FAILED, "the anonymous mapping is made");
            Self { raw, len }
        }

        /// `len` bytes of the mapping, `at` bytes in.
        fn at(&self, at: usize, len: usize) -> &[u8] {
            // SAFETY: `at + len` is inside the mapping, which stays mapped for
            // as long as `self` — which the borrow says.
            unsafe { std::slice::from_raw_parts(self.raw.cast::<u8>().add(at), len) }
        }

        /// `values` written into the mapping `at` bytes in, through the mapping
        /// rather than through anything Metal knows about.
        fn write(&self, at: usize, values: &[f32]) {
            assert!(at + size_of_val(values) <= self.len, "inside the mapping");
            // SAFETY: the mapping is writable, `values` is elsewhere, and the
            // range written is inside it by the assertion above.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr().cast::<u8>(),
                    self.raw.cast::<u8>().add(at),
                    size_of_val(values),
                )
            };
        }
    }

    impl Drop for Pages {
        fn drop(&mut self) {
            // SAFETY: the mapping this made, unmapped once.
            unsafe { libc::munmap(self.raw, self.len) };
        }
    }

    #[test]
    fn the_page_size_is_one_a_wrap_can_be_rounded_to() {
        let page = page();
        assert!(page.is_power_of_two() && page >= 4096, "{page}");
    }

    /// The whole claim: a wrap copies nothing, so what the GPU reads through it
    /// is whatever the pages hold *now* rather than what they held when it was
    /// made.
    ///
    /// Stated by writing through the mapping after the wrap and reading back
    /// through the kernel, which is the one assertion a copy cannot pass. The
    /// write lands before any dispatch is encoded, which is what
    /// [`Device::wrap`]'s contract asks — a write racing a running kernel is
    /// the thing it forbids, not a write between them.
    ///
    /// A checkpoint's pages are never written at all. But nothing else tells a
    /// wrap from a copy so sharply, and 137 GB of expert banks is exactly the
    /// size at which the two have to be told apart by something.
    #[test]
    fn a_wrap_reads_the_pages_rather_than_a_copy_of_them() {
        let Some(device) = device() else { return };
        let kernel = device
            .compile(crate::testing::SAXPY, crate::testing::SAXPY_ENTRY)
            .expect("saxpy compiles");

        let pages = Pages::new(4);
        let len = 1024;
        // Straddling a page boundary, so that the wrap has to round out in both
        // directions rather than only up.
        let at = page() - 8;
        // SAFETY: a slice of the anonymous mapping above, which outlives the
        // wrap and which nothing else holds.
        let mut wrapped = unsafe { device.wrap::<f32>(pages.at(at, len * size_of::<f32>())) }
            .expect("the pages wrap");
        let (offset, wrapped_len) = (wrapped.offset(), wrapped.len());
        assert_eq!(
            offset,
            at / size_of::<f32>(),
            "where in its pages it starts"
        );
        assert_eq!(
            wrapped_len,
            2 * page() / size_of::<f32>(),
            "the two pages a slice across a boundary falls in"
        );

        // Written *after* the wrap, through the mapping and not through Metal.
        let written: Vec<f32> = (0..len).map(|i| i as f32 * 0.25 - 3.0).collect();
        pages.write(at, &written);

        let mut alpha = device.buffer(&[2.0f32]).unwrap();
        let mut count = device.buffer(&[wrapped_len as u32]).unwrap();
        let mut zeros = device.zeroed::<f32>(wrapped_len).unwrap();
        let mut out = device.zeroed::<f32>(wrapped_len).unwrap();
        let args = [
            alpha.arg(),
            count.arg(),
            wrapped.arg(),
            zeros.arg(),
            out.arg(),
        ];
        device
            .run(&kernel, &args, crate::kernel::Grid::new(wrapped_len, 64))
            .expect("the dispatch completes");

        let got = &out.as_slice()[offset..][..len];
        assert_eq!(
            got,
            written.iter().map(|x| 2.0 * x).collect::<Vec<f32>>(),
            "the kernel read a copy taken at wrap time"
        );
    }

    /// The checkpoint's own misalignment, which is why a wrapped weight is read
    /// a byte at a time. Its shard headers are not padded, so every tensor in it
    /// starts one byte past a word — and a wrap that promised `u32` elements
    /// over those bytes would hand the kernel a pointer it cannot dereference.
    #[test]
    fn a_wrap_whose_elements_do_not_line_up_is_refused() {
        let Some(device) = device() else { return };
        let pages = Pages::new(2);
        let odd = pages.at(1, 64);

        // SAFETY: a slice of the mapping above, as in the case before it.
        let err = unsafe { device.wrap::<u32>(odd) }.expect_err("one byte past a word");
        assert!(
            matches!(err, MetalError::Misaligned { offset: 1, size: 4 }),
            "{err}"
        );
        // SAFETY: as above.
        assert!(
            unsafe { device.wrap::<u8>(odd) }.is_ok(),
            "bytes always line up"
        );
    }

    /// `alpha * x + y` over `x` handed to the kernel through [`Device::inline`],
    /// with everything else bound the way it always was.
    ///
    /// The two lengths are what the case is about: `x` is the one argument here
    /// wide enough to fall either side of the threshold, so the same call is a
    /// `setBytes:` at one length and an allocation at the other, and both have
    /// to be the same arithmetic.
    fn saxpy_over(len: usize) -> (Vec<f32>, bool) {
        let device = device().expect("the caller checked");
        let kernel = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");
        let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.125 - 7.0).collect();
        let y: Vec<f32> = (0..len).map(|i| 3.0 - i as f32 * 0.0625).collect();

        let alpha = [2.5f32];
        let count = [len as u32];
        let mut alpha = device.inline(&alpha).expect("a scalar");
        let mut count = device.inline(&count).expect("a count");
        let mut rows = device.inline(&x).expect("the rows");
        let inlined = matches!(rows, Inline::Bytes(_));
        let mut y = device.buffer(&y).expect("the addend uploads");
        let mut out = device.zeroed::<f32>(len).expect("the output allocates");
        device
            .run(
                &kernel,
                &[alpha.arg(), count.arg(), rows.arg(), y.arg(), out.arg()],
                Grid::new(len, 64),
            )
            .expect("the dispatch completes");

        (out.to_vec(), inlined)
    }

    /// The whole of what an inline argument has to be: the same answer the same
    /// values in an allocation give.
    /// The widest call that still travels in the command buffer, which is the
    /// side of the threshold an inclusive range is the difference between.
    #[test]
    fn a_dispatch_reads_inline_values_the_way_it_reads_an_allocation() {
        let Some(_) = device() else { return };
        let len = INLINE_BYTES / size_of::<f32>();

        let (got, inlined) = saxpy_over(len);

        assert!(inlined, "{len} floats travel in the command buffer");
        let want: Vec<f32> = (0..len)
            .map(|i| 2.5 * (i as f32 * 0.125 - 7.0) + (3.0 - i as f32 * 0.0625))
            .collect();
        assert_eq!(got, want);
    }

    /// And one value more is an allocation instead, read whole rather than
    /// truncated to what would have fitted.
    ///
    /// One more rather than twice as many, because what the two cases either
    /// side of the threshold settle is where it falls. Not a boundary anything
    /// contrives: an expert list is one `uint` a row, so a decode step's is six
    /// and a 769-token prefill's is 4614.
    #[test]
    fn one_value_past_the_command_buffer_is_allocated_instead() {
        let Some(_) = device() else { return };
        let len = INLINE_BYTES / size_of::<f32>() + 1;

        let (got, inlined) = saxpy_over(len);

        assert!(!inlined, "{len} floats are past the inline threshold");
        assert_eq!(got.len(), len);
        assert_eq!(
            got[len - 1],
            2.5 * ((len - 1) as f32 * 0.125 - 7.0) + (3.0 - (len - 1) as f32 * 0.0625),
            "the last value the allocation carried"
        );
    }

    /// **What separates an inline argument from a resident one written in
    /// place**, and the reason a shape is the first and not the second: the
    /// bytes are copied as the dispatch is encoded, so two dispatches of one
    /// command buffer can be encoded from storage that no longer holds either
    /// value by the time the GPU runs them.
    ///
    /// A resident shape buffer would have the second dispatch's write reach the
    /// first, which is exactly the pair a layer's projections are — and the
    /// pair a batching scheduler will make of two sequences of different
    /// heights.
    #[test]
    fn two_dispatches_of_one_command_buffer_keep_the_inline_values_each_was_encoded_with() {
        let Some(device) = device() else { return };
        let kernel = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");
        const LEN: usize = 256;
        let values: Vec<f32> = (0..LEN).map(|i| i as f32 * 0.5 - 3.0).collect();
        let mut x = device.buffer(&values).expect("the rows upload");
        let mut y = device.zeroed::<f32>(LEN).expect("the addend allocates");
        let mut first = device.zeroed::<f32>(LEN).expect("an output allocates");
        let mut second = device.zeroed::<f32>(LEN).expect("an output allocates");

        let mut batch = device.batch().expect("a command buffer opens");
        for (scalar, out) in [(2.0f32, &mut first), (10.0f32, &mut second)] {
            // Both scalars live only to the end of this iteration, which is
            // before the batch is submitted and long before it completes.
            let alpha = [scalar];
            let count = [LEN as u32];
            let mut alpha = device.inline(&alpha).expect("a scalar");
            let mut count = device.inline(&count).expect("a count");
            batch
                .add(
                    &kernel,
                    &[alpha.arg(), count.arg(), x.arg(), y.arg(), out.arg()],
                    Grid::new(LEN, 64),
                )
                .expect("the dispatch encodes");
        }
        batch.wait().expect("the batch completes");

        let scaled = |by: f32| values.iter().map(|v| by * v).collect::<Vec<f32>>();
        assert_eq!(first.to_vec(), scaled(2.0));
        assert_eq!(
            second.to_vec(),
            scaled(10.0),
            "the second overwrote the first"
        );
    }

    /// A dispatch over no values stays the refusal it has always been rather
    /// than becoming a binding of nothing, which the kernel would read as a
    /// pointer it may not follow.
    #[test]
    fn inlining_no_values_is_refused_the_way_allocating_none_is() {
        let Some(device) = device() else { return };

        let err = device
            .inline::<f32>(&[])
            .expect_err("a dispatch over nothing is refused");

        assert!(matches!(err, MetalError::Allocation { bytes: 0 }), "{err}");
    }

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
