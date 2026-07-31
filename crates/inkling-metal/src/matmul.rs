//! `out = x @ wᵀ` against a weight that stays MXFP4-packed.
//!
//! This is the primitive the engine rests on. `lm_head`, every projection and —
//! with a gather — every expert are the same operation over the same format, so
//! what this module gets right applies everywhere and what it gets wrong applies
//! everywhere too.
//!
//! **The codes are decoded in registers, inside the multiply loop.** A kernel
//! that dequantised into a buffer and multiplied that would be the CPU path with
//! a GPU under it, and would have given away the whole point before it started:
//! every token touches all of every projection, so a decode step moves 41 GB of
//! float32 where the packed bytes are 5 GB. Here a thread loads one packed word,
//! turns its eight nibbles into eight floats it holds in registers, multiplies
//! them against eight inputs, and never writes a decoded weight anywhere.
//!
//! **One simdgroup per output element.** Lane `l` walks its weight row from word
//! `l` in strides of the simdgroup width and the group sums what the lanes held,
//! so the 32 lanes of one reduction read 32 consecutive words — which is what
//! makes a matmul this memory-bound run at the bandwidth rather than at a
//! thirty-second of it. The cost is that a call with several rows of `x` reads
//! the weight once per row; decode is one row, and a prefill shape that wants
//! the weight read once is a tiling commit of its own.
//!
//! **`0x00` scale bytes decode to zero here.** The scale is
//! `as_type<float>(byte << 23)`, which is exact for `0x01..=0xfe` and gives 0
//! where [`inkling_core::quant`] gives `2^-127`. That is the divergence that
//! module licenses: it surveys all 458 quantised tensors and has never found a
//! group where the two readings decode to different weights, because `0x00`
//! appears only against all-zero codes. Taking the shift is a branch removed
//! from the inner loop, and the CPU path stays pinned to MLX.

use std::cell::RefCell;

use inkling_core::ops::Projection;
use inkling_core::quant::{BITS, ELEMENTS, GROUP_SIZE};
use inkling_core::weights::Packed;

use crate::buffer::Buffer;
use crate::device::{Device, MetalError};
use crate::kernel::{Grid, Kernel};

const ENTRY: &str = "packed_matmul";

/// Threads one threadgroup of a dispatch holds.
///
/// A multiple of every simdgroup width Metal reports, which is what lets the
/// kernel take its output element from `thread_position_in_grid` divided by
/// `threads_per_simdgroup` and get the same answer as `thread_index_in_simdgroup`
/// gives for the lane.
const THREADS_PER_GROUP: usize = 256;

/// Codes packed into one `u32`, which is where the `[out, in/8]` shape of a
/// packed weight comes from.
const CODES_PER_WORD: usize = u32::BITS as usize / BITS;

/// Packed words one scale byte covers.
const WORDS_PER_GROUP: usize = GROUP_SIZE / CODES_PER_WORD;

/// Where an f32's exponent field starts, above its 23 stored mantissa bits.
const EXPONENT_SHIFT: u32 = f32::MANTISSA_DIGITS - 1;

#[derive(Debug, thiserror::Error)]
pub enum MatmulError {
    #[error(transparent)]
    Metal(#[from] MetalError),

    #[error("an input width of {0} is not whole groups of {GROUP_SIZE} codes")]
    PartialGroup(usize),

    #[error("{rows} rows of {in_dim} codes are {expected} packed bytes, got {got}")]
    WrongCodeLen {
        rows: usize,
        in_dim: usize,
        expected: usize,
        got: usize,
    },

    #[error("{rows} rows of {in_dim} codes need {expected} scale bytes, got {got}")]
    WrongScaleLen {
        rows: usize,
        in_dim: usize,
        expected: usize,
        got: usize,
    },
}

/// The compiled kernel, which every packed projection on a device shares.
///
/// Compilation is per source string rather than per weight, and the source does
/// not mention a shape, so one of these serves the whole model.
#[derive(Debug)]
pub struct PackedMatmul {
    kernel: Kernel,
}

impl PackedMatmul {
    pub fn new(device: &Device) -> Result<Self, MetalError> {
        Self::from_source(device, &source())
    }

    /// [`PackedMatmul::new`] out of a source string of the caller's own, which
    /// is how a test puts a deliberately wrong kernel through the same plumbing
    /// as the right one and measures the difference.
    pub(crate) fn from_source(device: &Device, source: &str) -> Result<Self, MetalError> {
        Ok(Self {
            kernel: device.compile(source, ENTRY)?,
        })
    }
}

/// One projection's weights, resident on the device exactly as the checkpoint
/// stores them.
///
/// Uploaded once and multiplied against many times, which is the shape the model
/// wants: `lm_head` is 411 MB of codes, and a decode step that re-uploaded them
/// would be moving more bytes than the multiply reads.
#[derive(Debug)]
pub struct PackedProjection<'a> {
    device: &'a Device,
    matmul: &'a PackedMatmul,
    in_dim: usize,
    out_dim: usize,
    resident: RefCell<Resident>,
}

/// What a projection holds on the device between calls: the two tensors the
/// checkpoint pairs, and the shape the kernel reads its bounds out of.
#[derive(Debug)]
struct Resident {
    shape: Buffer<u32>,
    codes: Buffer<u8>,
    scales: Buffer<u8>,
}

/// The three scalars in the order the kernel's `Shape` struct declares them.
///
/// The kernel reads them as `uint`, so this is where a shape too large to
/// describe has to stop. Unreachable through any real weight — four billion rows
/// of anything is decades past the 348 GiB one allocation can hold — and a
/// truncation would not fail: it would dispatch a grid for the wrong shape over
/// buffers of the right one.
fn shape(rows: usize, in_dim: usize, out_dim: usize) -> [u32; 3] {
    [rows, in_dim, out_dim].map(|extent| {
        u32::try_from(extent).unwrap_or_else(|_| panic!("{extent} is wider than a kernel's uint"))
    })
}

impl<'a> PackedProjection<'a> {
    /// Upload an `[out_dim, in_dim]` MXFP4 weight, as the two tensors the
    /// checkpoint pairs: `codes` is the bytes of the `U32` weight and `scales`
    /// the bytes of the `U8` one, both taken as they are mapped.
    ///
    /// The codes arrive as bytes and are bound to a `device const uint *`. That
    /// is a reinterpretation safetensors already made — the packed words are
    /// little-endian in the file and this machine is little-endian, so the bytes
    /// *are* the words — and taking `&[u32]` instead would mean transcoding
    /// `lm_head`'s 411 MB on the way in without changing a bit of it.
    pub fn upload(
        device: &'a Device,
        matmul: &'a PackedMatmul,
        in_dim: usize,
        out_dim: usize,
        codes: &[u8],
        scales: &[u8],
    ) -> Result<Self, MatmulError> {
        if in_dim == 0 || in_dim % GROUP_SIZE != 0 {
            return Err(MatmulError::PartialGroup(in_dim));
        }

        let expected = out_dim * in_dim * BITS / u8::BITS as usize;
        if codes.len() != expected {
            return Err(MatmulError::WrongCodeLen {
                rows: out_dim,
                in_dim,
                expected,
                got: codes.len(),
            });
        }
        let expected = out_dim * in_dim / GROUP_SIZE;
        if scales.len() != expected {
            return Err(MatmulError::WrongScaleLen {
                rows: out_dim,
                in_dim,
                expected,
                got: scales.len(),
            });
        }

        Ok(Self {
            resident: RefCell::new(Resident {
                shape: device.buffer(&shape(0, in_dim, out_dim))?,
                codes: device.buffer(codes)?,
                scales: device.buffer(scales)?,
            }),
            device,
            matmul,
            in_dim,
            out_dim,
        })
    }

    /// [`PackedProjection::upload`] over a checkpoint's own tensor, cut to its
    /// first `out_dim` slices.
    ///
    /// The cut is what the head's truncation is: `lm_head` is `[201024, 4096]`
    /// and 200058 of those rows are vocabulary, so a projection built to the
    /// vocabulary leaves 966 rows of padding in the mapping rather than putting
    /// 2 MB of guaranteed zeros on the device and dispatching over them. A bank
    /// of experts is the other caller this shape has — one slice of a leading
    /// axis, offset rather than cut — and it is not this method.
    pub fn upload_packed(
        device: &'a Device,
        matmul: &'a PackedMatmul,
        packed: &Packed<'_>,
        out_dim: usize,
    ) -> Result<Self, MatmulError> {
        let (codes, scales) = packed.prefix(out_dim);
        Self::upload(device, matmul, packed.slice_len(), out_dim, codes, scales)
    }

    /// `[rows, in_dim]` in, `[rows, out_dim]` out.
    ///
    /// Fallible where [`Projection::forward`] is not, because a dispatch can
    /// fail in ways no arithmetic can: the watchdog kills a command buffer that
    /// runs too long, and a caller that wants to say so rather than die needs
    /// the error.
    pub fn multiply(&self, x: &[f32]) -> Result<Vec<f32>, MatmulError> {
        assert_eq!(
            x.len() % self.in_dim,
            0,
            "{} values are not whole rows of {}",
            x.len(),
            self.in_dim
        );
        let rows = x.len() / self.in_dim;
        if rows == 0 {
            return Ok(Vec::new());
        }

        let mut resident = self.resident.borrow_mut();
        let resident = &mut *resident;
        resident
            .shape
            .as_mut_slice()
            .copy_from_slice(&shape(rows, self.in_dim, self.out_dim));

        let mut x = self.device.buffer(x)?;
        let mut out = self.device.zeroed::<f32>(rows * self.out_dim)?;
        let args = [
            resident.shape.arg(),
            x.arg(),
            resident.codes.arg(),
            resident.scales.arg(),
            out.arg(),
        ];

        let elements = rows * self.out_dim;
        let kernel = &self.matmul.kernel;
        let grid = Grid::new(elements * kernel.simd_width(), THREADS_PER_GROUP);
        self.device.run(kernel, &args, grid)?;

        Ok(out.to_vec())
    }
}

/// The seam [`inkling_core::ops`] names, so that a caller holding a projection
/// does not know whether its weights were ever decoded.
///
/// Infallible where [`PackedProjection::multiply`] is not. The CPU side of the
/// seam cannot fail, and a `Result` on the trait would be one every caller of
/// every projection carries for a case only this one has; a dispatch that does
/// not complete is a panic here for the reason a missing tensor is one in
/// `inkling_core::weights`, that nothing above it can do anything about it.
impl Projection for PackedProjection<'_> {
    fn in_dim(&self) -> usize {
        self.in_dim
    }

    fn out_dim(&self) -> usize {
        self.out_dim
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        self.multiply(x)
            .unwrap_or_else(|err| panic!("the packed matmul did not run: {err}"))
    }
}

/// The kernel, with the format's own constants and element table written into
/// its prelude.
///
/// Generated rather than spelled out because [`inkling_core::quant`] is the
/// authority on every one of them — the nibble order, where the sign lives, the
/// eight magnitudes — and each is a fact read off MLX rather than off the OCP
/// specification. A second copy of that table living in a source string is a
/// copy that can drift from the checkpoint it decodes.
fn source() -> String {
    let elements: Vec<String> = ELEMENTS.iter().map(|value| format!("{value:?}f")).collect();
    format!(
        "\
#include <metal_stdlib>
using namespace metal;

constant uint BITS = {BITS};
constant uint CODE_MASK = {};
constant uint CODES_PER_WORD = {CODES_PER_WORD};
constant uint WORDS_PER_GROUP = {WORDS_PER_GROUP};
constant uint EXPONENT_SHIFT = {EXPONENT_SHIFT};
constant float ELEMENTS[] = {{ {} }};
{BODY}",
        (1u32 << BITS) - 1,
        elements.join(", "),
    )
}

/// Everything of the kernel that the format does not decide.
const BODY: &str = r#"
struct Shape {
    uint rows;
    uint in_dim;
    uint out_dim;
};

kernel void packed_matmul(
    constant Shape &shape [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const uint *codes [[buffer(2)]],
    device const uchar *scales [[buffer(3)]],
    device float *out [[buffer(4)]],
    uint position [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]
) {
    const uint element = position / width;
    if (element >= shape.rows * shape.out_dim) {
        return;
    }

    const uint row = element / shape.out_dim;
    const uint col = element % shape.out_dim;
    const uint words = shape.in_dim / CODES_PER_WORD;

    device const uint *packed = codes + (ulong)col * words;
    device const uchar *scale = scales + (ulong)col * (words / WORDS_PER_GROUP);
    device const float *values = x + (ulong)row * shape.in_dim;

    // The whole of the decode. A word's eight codes become eight floats that
    // live in registers for the eight multiplies below and are never stored.
    // The group scale is applied once to their sum rather than eight times to
    // its terms, which is the same value and one multiply instead of eight.
    float sum = 0.0f;
    for (uint word = lane; word < words; word += width) {
        const uint code = packed[word];
        device const float *v = values + word * CODES_PER_WORD;

        float dot = 0.0f;
        for (uint i = 0; i < CODES_PER_WORD; ++i) {
            dot += ELEMENTS[(code >> (BITS * i)) & CODE_MASK] * v[i];
        }
        sum += dot * as_type<float>(uint(scale[word / WORDS_PER_GROUP]) << EXPONENT_SHIFT);
    }

    sum = simd_sum(sum);
    if (lane == 0) {
        out[element] = sum;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use inkling_core::fixture::{self, deviation};
    use inkling_core::ops::DenseProjection;
    use inkling_core::quant::dequantize_blocks;
    use inkling_core::weights::PackedRows;

    use crate::testing::device;

    /// The reduction the checkpoint's projections are: `lm_head`, every
    /// attention projection and every expert reduce over 4096.
    const IN_DIM: usize = 4096;

    /// Output elements enough to span many threadgroups without filling the last
    /// one, and prime so that no divisor of the dispatch happens to line up with
    /// it.
    const OUT_DIM: usize = 257;

    /// How far a dispatch may land from the CPU's answer for the same weights.
    ///
    /// Decoding is exact on both sides — a table lookup times a power of two —
    /// and neither rounds anywhere else, so summation order is the whole of what
    /// separates them. **This bound is therefore not a bound on the kernel; it
    /// is a bound on the oracle.** The CPU adds 4096 products serially, whose
    /// drift grows like the square root of the reduction: 64 ulps at this
    /// length, and an f32 ulp is 6e-8, so 3.8e-6 of the tensor's peak is what a
    /// serial f32 loop is expected to give up. The kernel sums 128 a lane and
    /// reduces 32 lanes in a tree, which is the better-conditioned order by a
    /// factor of the same shape.
    ///
    /// Measured against an f64 accumulation of the same products, that is
    /// exactly what happens: the kernel drifts 1.4e-7 — under three ulps — where
    /// the CPU drifts 2.8e-6, and the 2.8e-6 they disagree by is the CPU's own
    /// error arriving whole. 6e-6 admits that with a factor of two in hand.
    ///
    /// Which is why the assertion beside it is the one with teeth: the kernel
    /// has to be *closer to exact* than the CPU, not merely inside a bound. A
    /// dispatch that decoded something wrongly would fail that while a widened
    /// tolerance would still let it through — and the weakest mutation this has
    /// to catch, a kernel reading each word's nibbles from the top down, lands
    /// at 8.1e-1, five decades above.
    const TOLERANCE: f32 = 6e-6;

    /// The eight E2M1 magnitudes, written out here rather than read off the
    /// table the kernel is built from, so that a case computed by hand is
    /// computed independently of what it checks.
    const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

    /// What a code under a scale byte stands for, from the format rather than
    /// from a decoder: bit 3 carries the sign, the three below it index the
    /// magnitudes, and the byte is an exponent biased by 127.
    fn element(code: u8, scale: u8) -> f32 {
        let magnitude = MAGNITUDES[usize::from(code & 7)];
        let signed = if code & 8 == 0 { magnitude } else { -magnitude };
        signed * f32::from_bits(u32::from(scale) << EXPONENT_SHIFT)
    }

    /// Codes packed the way the checkpoint packs them: eight to a little-endian
    /// word, code `i` of a word in bits `4i..4i+4`.
    fn pack(codes: &[u8]) -> Vec<u8> {
        codes
            .chunks_exact(CODES_PER_WORD)
            .flat_map(|word| {
                word.iter()
                    .enumerate()
                    .fold(0u32, |packed, (i, code)| {
                        packed | (u32::from(*code) << (BITS * i))
                    })
                    .to_le_bytes()
            })
            .collect()
    }

    /// A deterministic stand-in for trained weights. Any bit pattern does, so
    /// long as a rerun sees the same one.
    ///
    /// The low eight bits of a linear congruential state are the poorly mixed
    /// ones, so they are shifted off and what is left is 24 bits.
    struct Noise(u32);

    impl Noise {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            self.0 >> 8
        }

        /// A value spread over `-1.0..1.0`.
        ///
        /// The spread is load-bearing rather than cosmetic: against a *constant*
        /// input a row's dot product is its weights summed, which no permutation
        /// of codes within a word can move — so a kernel reading its nibbles
        /// backwards would agree to four digits and the mutation this file
        /// measures would measure nothing.
        fn signed(&mut self) -> f32 {
            self.next() as f32 / (1u32 << 23) as f32 - 1.0
        }
    }

    /// One multiply: an `[out_dim, in_dim]` weight held as one code per element
    /// beside one scale byte per group, and the rows of `x` to put through it.
    struct Case {
        in_dim: usize,
        out_dim: usize,
        codes: Vec<u8>,
        scales: Vec<u8>,
        x: Vec<f32>,
    }

    impl Case {
        /// Codes over the whole table and inputs of mixed sign, which is what
        /// makes the reduction cancel the way a trained one does and so what
        /// makes two summation orders part company at all.
        ///
        /// The scales are shaped after the checkpoint's rather than spread over
        /// the byte: `lm_head`'s span `0x74..=0x7e` across the tensor while the
        /// 128 groups *within* a row span a median of one byte and at most four.
        /// That structure is what sets how ill-conditioned the reduction is, and
        /// a synthetic weight whose groups spanned twenty-six powers of two
        /// would be measuring a serial f32 loop falling apart on a case no
        /// checkpoint contains. `0x00` is left out on purpose — it is the one
        /// reading the two sides deliberately disagree about, and
        /// [`a_zero_scale_byte_multiplies_to_zero_where_the_cpu_gives_two_to_the_minus_127`]
        /// is where that is stated.
        fn noisy(in_dim: usize, out_dim: usize, rows: usize) -> Self {
            let mut noise = Noise(0x1234_5678);
            let groups = in_dim / GROUP_SIZE;
            let mut scales = Vec::with_capacity(out_dim * groups);
            for _ in 0..out_dim {
                let row = 0x74 + (noise.next() % 11) as u8;
                scales.extend((0..groups).map(|_| row + (noise.next() % 5) as u8));
            }

            Self {
                codes: (0..out_dim * in_dim)
                    .map(|_| (noise.next() % 16) as u8)
                    .collect(),
                x: (0..rows * in_dim).map(|_| noise.signed()).collect(),
                scales,
                in_dim,
                out_dim,
            }
        }

        fn packed(&self) -> Vec<u8> {
            pack(&self.codes)
        }

        fn upload<'a>(&self, device: &'a Device, matmul: &'a PackedMatmul) -> PackedProjection<'a> {
            PackedProjection::upload(
                device,
                matmul,
                self.in_dim,
                self.out_dim,
                &self.packed(),
                &self.scales,
            )
            .expect("the case's shapes pair")
        }

        /// The same multiply through the decoder and the CPU projection this
        /// kernel exists to replace, which is the oracle for everything below.
        fn on_the_cpu(&self) -> Vec<f32> {
            self.on_the_cpu_with(&self.packed(), &self.scales)
        }

        /// [`Case::on_the_cpu`] over bytes of the caller's choosing, which is
        /// how a mutation is measured against the same machinery.
        fn on_the_cpu_with(&self, packed: &[u8], scales: &[u8]) -> Vec<f32> {
            let weight = dequantize_blocks(packed, scales).expect("the case decodes");
            DenseProjection::new(self.in_dim, &weight).forward(&self.x)
        }

        /// The same multiply summed in f64, which neither side does.
        ///
        /// Decoding is exact on both sides — a table lookup times a power of two
        /// — so the products are the same f32s either way and summation order is
        /// the only thing left to differ about. Accumulating those products with
        /// 29 bits of headroom settles which of the two orders is drifting,
        /// which is what turns a disagreement into either float noise or a bug.
        fn exactly(&self) -> Vec<f64> {
            let weight = dequantize_blocks(&self.packed(), &self.scales).expect("the case decodes");
            let mut out = Vec::new();
            for x in self.x.chunks_exact(self.in_dim) {
                out.extend(weight.chunks_exact(self.in_dim).map(|row| {
                    x.iter()
                        .zip(row)
                        .map(|(x, w)| f64::from(*x) * f64::from(*w))
                        .sum::<f64>()
                }));
            }
            out
        }
    }

    /// How far an answer lands from the exact one, as a fraction of the exact
    /// tensor's peak — [`deviation`]'s measure, against f64 rather than against
    /// the other f32 answer.
    fn drift(got: &[f32], exact: &[f64]) -> f64 {
        assert_eq!(got.len(), exact.len(), "length");
        let scale = exact.iter().fold(0.0f64, |peak, w| peak.max(w.abs()));
        got.iter().zip(exact).fold(0.0f64, |worst, (got, exact)| {
            worst.max((f64::from(*got) - exact).abs())
        }) / scale
    }

    /// Everything a dispatch needs, so that no test opens a device twice.
    fn matmul(device: &Device) -> PackedMatmul {
        PackedMatmul::new(device).expect("the packed matmul compiles")
    }

    /// The smallest claim there is, and the one every other test here assumes:
    /// that a code times its group's scale times an input is what lands in the
    /// output.
    ///
    /// Exact rather than bounded, and that is affordable rather than lucky: every
    /// magnitude is a dyadic of three significant bits, the inputs are small
    /// integers, and the scales are powers of two either side of one, so every
    /// product and every partial sum is representable and no ordering can move a
    /// bit. A tolerance here would only be hiding a plumbing mistake.
    ///
    /// The two rows carry the same codes in opposite order under different
    /// scales, so a kernel that read one row's scale for both, or that indexed
    /// the codes from the wrong end, produces the other row's answer rather than
    /// a near miss.
    #[test]
    fn a_dispatch_multiplies_what_the_codes_and_their_scale_stand_for() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);

        let forwards: Vec<u8> = (0..GROUP_SIZE).map(|i| (i % 16) as u8).collect();
        let backwards: Vec<u8> = forwards.iter().rev().copied().collect();
        let case = Case {
            in_dim: GROUP_SIZE,
            out_dim: 2,
            codes: [forwards.clone(), backwards.clone()].concat(),
            scales: vec![0x7f, 0x80],
            x: (0..GROUP_SIZE).map(|i| i as f32 + 1.0).collect(),
        };

        let want: Vec<f32> = [(&forwards, 0x7f), (&backwards, 0x80)]
            .into_iter()
            .map(|(codes, scale)| {
                codes
                    .iter()
                    .zip(&case.x)
                    .map(|(code, x)| element(*code, scale) * x)
                    .sum()
            })
            .collect();
        assert_ne!(want[0], want[1], "two rows that agreed would prove nothing");

        let got = case
            .upload(&device, &matmul)
            .multiply(&case.x)
            .expect("the dispatch completes");
        assert_eq!(got, want);
    }

    /// The kernel against the CPU it replaces, over the reduction length and the
    /// spread of codes and scales the checkpoint actually holds.
    ///
    /// The dispatch is deliberately ragged: 257 outputs over three rows is 771
    /// simdgroups, which is neither a whole number of threadgroups nor a whole
    /// number of anything else, so the tail group runs lanes past the end of the
    /// work and the bounds check is what stops them writing.
    #[test]
    fn the_kernel_reproduces_the_cpu_over_synthetic_packed_weights() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let case = Case::noisy(IN_DIM, OUT_DIM, 3);

        let projection = case.upload(&device, &matmul);
        let width = matmul.kernel.simd_width();
        let elements = 3 * OUT_DIM;
        assert!(
            elements * width % THREADS_PER_GROUP != 0,
            "a dispatch that filled its last threadgroup would not exercise the bounds check"
        );

        let got = projection
            .multiply(&case.x)
            .expect("the dispatch completes");
        let on_the_cpu = case.on_the_cpu();
        let deviation = deviation(&got, &on_the_cpu);
        assert!(
            deviation > 0.0,
            "an exact match would mean the two are not summing independently"
        );

        // Which of the two is drifting, which is what says whether a
        // disagreement of this size is float noise or a bug. The kernel sums 128
        // products a lane and then reduces 32 lanes in a tree; the CPU sums 4096
        // serially. The tree is the better-conditioned order, so the kernel has
        // to be the *closer* of the two to the exact answer — a kernel that was
        // merely within the bound while sitting further out than a serial f32
        // loop would be one hiding a mistake inside a tolerance.
        let exact = case.exactly();
        let (mine, theirs) = (drift(&got, &exact), drift(&on_the_cpu, &exact));
        eprintln!(
            "synthetic weights: deviation {deviation:e}, drift from exact {mine:e} against the \
             CPU's {theirs:e}"
        );
        assert!(deviation <= TOLERANCE, "deviation {deviation:e}");
        assert!(mine < theirs, "{mine:e} against the CPU's {theirs:e}");
    }

    /// The nibble order, which is the one fact about the format a kernel can get
    /// backwards while still producing plausible weights of the right magnitude.
    ///
    /// Stated as a kernel rather than as a mutated input, because that is where
    /// the mistake would live: the same source, the same dispatch, the same
    /// bytes, and the eight codes of each word read from the top down instead of
    /// from the bottom up.
    #[test]
    fn reading_each_words_nibbles_from_the_top_down_is_a_different_answer() {
        let Some(device) = device() else { return };
        let case = Case::noisy(IN_DIM, OUT_DIM, 1);

        let reversed = source().replace("(code >> (BITS * i))", "(code >> (32u - BITS * (i + 1)))");
        assert_ne!(reversed, source(), "the mutation changed nothing");
        let mutant = PackedMatmul::from_source(&device, &reversed).expect("the mutant compiles");

        let got = case
            .upload(&device, &mutant)
            .multiply(&case.x)
            .expect("the dispatch completes");
        let deviation = deviation(&got, &case.on_the_cpu());
        eprintln!("nibbles read the other way round: deviation {deviation:e}");
        assert!(deviation > TOLERANCE, "deviation {deviation:e}");
    }

    /// A group's scale is its own, and a kernel that read one per weight row —
    /// or that took them an index out — would agree with everything above on a
    /// weight whose groups happened to share a scale. Exchanging two adjacent
    /// scale bytes has to move the answer.
    #[test]
    fn each_group_multiplies_under_its_own_scale() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let case = Case::noisy(IN_DIM, OUT_DIM, 1);

        // Inside one weight row, not across two of them: the scales are laid out
        // a row at a time, and a swap that straddled the boundary would be
        // exchanging two rows' scales rather than two groups' — which is a
        // different mistake, and one the row indexing already answers for.
        let groups = IN_DIM / GROUP_SIZE;
        let mut swapped = case.scales.clone();
        let boundary = (0..swapped.len() - 1)
            .find(|i| i % groups != groups - 1 && swapped[*i] != swapped[i + 1])
            .expect("some two adjacent groups of one row differ in scale");
        swapped.swap(boundary, boundary + 1);

        let got = case
            .upload(&device, &matmul)
            .multiply(&case.x)
            .expect("the dispatch completes");
        let deviation = deviation(&got, &case.on_the_cpu_with(&case.packed(), &swapped));
        assert!(deviation > TOLERANCE, "deviation {deviation:e}");
    }

    /// The licensed divergence, stated where it can be seen rather than left in
    /// a comment. `inkling_core::quant` reads `0x00` as `2^-127`, which is
    /// MLX's own reading and what the CPU path is pinned to; this kernel shifts
    /// the byte into the exponent field and so reads it as zero.
    ///
    /// Both are safe on this checkpoint because `0x00` appears only against
    /// all-zero codes, where the readings agree. A group with nonzero codes
    /// under it is what tells them apart, and it takes a synthetic weight to
    /// build one.
    #[test]
    fn a_zero_scale_byte_multiplies_to_zero_where_the_cpu_gives_two_to_the_minus_127() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);

        let case = Case {
            in_dim: GROUP_SIZE,
            out_dim: 1,
            codes: vec![7; GROUP_SIZE],
            scales: vec![0x00],
            x: vec![1.0; GROUP_SIZE],
        };

        let got = case
            .upload(&device, &matmul)
            .multiply(&case.x)
            .expect("the dispatch completes");
        assert_eq!(got, [0.0]);

        // The same group under the CPU's reading: nonzero, and thirty decades
        // below any weight a checkpoint carries. That gap is the whole of what
        // the shift throws away.
        let on_the_cpu = case.on_the_cpu()[0];
        assert!(on_the_cpu > 0.0 && on_the_cpu < 1e-30, "{on_the_cpu:e}");
    }

    /// Rows of `x` are independent, and each gets its own row of the output at
    /// its own offset. A kernel that took the row index off the wrong axis would
    /// still fill the buffer.
    #[test]
    fn every_row_of_the_input_gets_its_own_row_of_the_output() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let case = Case::noisy(IN_DIM, OUT_DIM, 2);
        let projection = case.upload(&device, &matmul);

        let both = projection
            .multiply(&case.x)
            .expect("the dispatch completes");
        assert_eq!(both.len(), 2 * OUT_DIM);
        for (row, x) in case.x.chunks_exact(IN_DIM).enumerate() {
            let alone = projection.multiply(x).expect("the dispatch completes");
            assert_eq!(both[row * OUT_DIM..][..OUT_DIM], alone[..], "row {row}");
        }
    }

    /// A checkpoint's packed tensor onto the device, cut where the vocabulary
    /// ends — and answering what the CPU makes of the same rows.
    ///
    /// [`PackedRows`] is the oracle rather than a decoded tensor here on
    /// purpose: it is the projection the engine runs today, so what this states
    /// is that exchanging one for the other is a change of backend and not a
    /// change of answer.
    ///
    /// The rows past the cut are the assertion's other half. They decode to
    /// exactly 0.0 — [`inkling_core::head`] is where that matters — so a
    /// dispatch that quietly uploaded the whole tensor would still agree on the
    /// 32 rows it was asked about, and only the length says otherwise.
    #[test]
    fn a_checkpoints_packed_tensor_uploads_the_rows_it_was_cut_to() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let ckpt = fixture::open(fixture::MXFP4);
        let packed =
            Packed::open(&ckpt, fixture::VOCAB_PADDING).expect("the fixture holds the slice");

        let mut noise = Noise(0x0f1e_2d3c);
        let x: Vec<f32> = (0..2 * packed.slice_len())
            .map(|_| noise.signed())
            .collect();
        let got =
            PackedProjection::upload_packed(&device, &matmul, &packed, fixture::VOCAB_PADDING_ROWS)
                .expect("the cut tensor uploads")
                .multiply(&x)
                .expect("the dispatch completes");
        assert_eq!(
            got.len(),
            2 * fixture::VOCAB_PADDING_ROWS,
            "the padding rows were uploaded"
        );

        let want = PackedRows::new(packed, fixture::VOCAB_PADDING_ROWS).forward(&x);
        let deviation = deviation(&got, &want);
        eprintln!(
            "{} rows of {}: deviation {deviation:e}",
            fixture::VOCAB_PADDING_ROWS,
            fixture::VOCAB_PADDING
        );
        assert!(deviation <= TOLERANCE, "deviation {deviation:e}");
    }

    /// A batch that happens to be empty is the caller's business rather than an
    /// error, and it cannot become a dispatch: the device refuses a zero-length
    /// buffer, so an output of nothing has to be answered without allocating one.
    #[test]
    fn no_rows_of_input_produce_no_output() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let case = Case::noisy(GROUP_SIZE, 2, 0);

        assert_eq!(
            case.upload(&device, &matmul)
                .multiply(&[])
                .expect("an empty multiply completes"),
            Vec::<f32>::new()
        );
    }

    /// A weight paired with another tensor's scales is the mistake the shapes
    /// exist to catch, and it has to be caught on the way in: the kernel takes
    /// its bounds from the shape it was told and would read off the end of
    /// whichever buffer was short.
    #[test]
    fn a_weight_and_scales_that_do_not_pair_are_refused() {
        let Some(device) = device() else { return };
        let matmul = matmul(&device);
        let case = Case::noisy(GROUP_SIZE, 4, 1);
        let upload = |in_dim, out_dim, codes: &[u8], scales: &[u8]| {
            PackedProjection::upload(&device, &matmul, in_dim, out_dim, codes, scales)
                .expect_err("the shapes do not pair")
        };

        let (packed, scales) = (case.packed(), case.scales.clone());
        assert!(
            matches!(
                upload(GROUP_SIZE, 4, &packed[..packed.len() - 4], &scales),
                MatmulError::WrongCodeLen { expected: 64, .. }
            ),
            "short codes"
        );
        assert!(
            matches!(
                upload(GROUP_SIZE, 4, &packed, &scales[..3]),
                MatmulError::WrongScaleLen {
                    expected: 4,
                    got: 3,
                    ..
                }
            ),
            "short scales"
        );
        assert!(
            matches!(
                upload(GROUP_SIZE / 2, 8, &packed, &scales),
                MatmulError::PartialGroup(16)
            ),
            "a width that is not whole groups"
        );
    }
}
