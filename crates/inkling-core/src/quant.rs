//! MXFP4 dequantisation, pinned to MLX.
//!
//! Every projection in an Inkling MXFP4 checkpoint is stored as a `U32` tensor
//! of 4-bit element codes paired with a `U8` tensor of block scales, one scale
//! per 32 codes. What follows — the nibble order, where the sign lives, the
//! element table, and what the extreme scale bytes mean — was read off MLX's
//! own dequantiser rather than off the OCP specification, because two of the
//! answers differ from it. `reference/fixtures/mxfp4_dequant.safetensors` holds
//! the slices that establish each claim, and the tests below reproduce them
//! bit for bit.

use crate::checkpoint::{Dtype, TensorView};

/// Codes sharing one scale byte.
pub const GROUP_SIZE: usize = 32;
pub const BITS: usize = 4;

const CODES_PER_WORD: usize = u32::BITS as usize / BITS;
const WORDS_PER_GROUP: usize = GROUP_SIZE / CODES_PER_WORD;
const WORD_BYTES: usize = size_of::<u32>();
const CODE_MASK: u32 = (1 << BITS) - 1;
/// Where an f32's exponent field starts, above its 23 stored mantissa bits.
const EXPONENT_SHIFT: u32 = f32::MANTISSA_DIGITS - 1;

/// The element table. A code is E2M1 — sign in bit 3, a two-bit exponent and a
/// one-bit mantissa below it — which enumerates to these eight magnitudes and
/// their negations.
///
/// Written out rather than computed because the sign of zero has to survive:
/// code 8 is `-0.0`, MLX carries that sign through the scale multiply, and a
/// table that spelled it `0.0` would differ from MLX on exactly those values
/// while looking right everywhere else.
const ELEMENTS: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// `2^-127`, which f32 holds only as the subnormal `2^22 * 2^-149`.
const SMALLEST_BLOCK_SCALE: f32 = f32::from_bits(1 << (EXPONENT_SHIFT - 1));

#[derive(Debug, thiserror::Error)]
pub enum QuantError {
    #[error("{what} is {got:?}, not {expected:?}")]
    WrongDtype {
        what: &'static str,
        expected: Dtype,
        got: Dtype,
    },

    #[error("{0} packed bytes are not a whole number of 32-bit words")]
    PartialWord(usize),

    #[error("{words} packed words hold {} codes, not a whole number of {GROUP_SIZE}-code groups", .words * CODES_PER_WORD)]
    PartialGroup { words: usize },

    #[error("{groups} groups of codes need {groups} scale bytes, got {scales}")]
    ScaleCountMismatch { groups: usize, scales: usize },

    #[error("weights shaped {weights:?} do not pair with scales shaped {scales:?}")]
    ShapeMismatch {
        weights: Vec<usize>,
        scales: Vec<usize>,
    },
}

/// A decoded tensor: the logical shape the packed one stood for, and its
/// values in row-major order.
#[derive(Debug, Clone)]
pub struct Dequantized {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

/// Decode a `U32` weight tensor and its `U8` scales into f32.
pub fn dequantize(
    weights: &TensorView<'_>,
    scales: &TensorView<'_>,
) -> Result<Dequantized, QuantError> {
    expect_dtype("packed weights", Dtype::U32, weights.dtype())?;
    expect_dtype("block scales", Dtype::U8, scales.dtype())?;
    Ok(Dequantized {
        shape: logical_shape(weights.shape(), scales.shape())?,
        values: dequantize_blocks(weights.data(), scales.data())?,
    })
}

/// Decode packed codes straight from their bytes, one scale byte per group.
///
/// The packed words are little-endian as safetensors wrote them, and within a
/// word the codes run from the least-significant nibble up: code `i` of a word
/// occupies bits `4i..4i+4`. That order was established by handing MLX words
/// with a single nonzero nibble and seeing which output slot moved.
pub fn dequantize_blocks(packed: &[u8], scales: &[u8]) -> Result<Vec<f32>, QuantError> {
    if packed.len() % WORD_BYTES != 0 {
        return Err(QuantError::PartialWord(packed.len()));
    }
    let words = packed.len() / WORD_BYTES;
    if words % WORDS_PER_GROUP != 0 {
        return Err(QuantError::PartialGroup { words });
    }
    let groups = words / WORDS_PER_GROUP;
    if scales.len() != groups {
        return Err(QuantError::ScaleCountMismatch {
            groups,
            scales: scales.len(),
        });
    }

    let mut values = Vec::with_capacity(words * CODES_PER_WORD);
    let blocks = packed.chunks_exact(WORDS_PER_GROUP * WORD_BYTES);
    for (block, &byte) in blocks.zip(scales) {
        let scale = block_scale(byte);
        for word in block.chunks_exact(WORD_BYTES) {
            let word = u32::from_le_bytes(word.try_into().expect("chunked into words"));
            for shift in (0..u32::BITS).step_by(BITS) {
                values.push(ELEMENTS[((word >> shift) & CODE_MASK) as usize] * scale);
            }
        }
    }
    Ok(values)
}

/// The multiplier a scale byte stands for.
///
/// The byte is an E8M0 exponent biased by 127, so it means `2^(byte - 127)`,
/// which for `0x01..=0xfe` is exactly the f32 whose exponent field is the byte
/// itself. The two ends are where reading the OCP specification would mislead:
///
/// - `0x00` is `2^-127`, below the smallest normal f32, so it has to be written
///   as a subnormal instead of as an exponent field. Metal's kernel does shift
///   the byte into the exponent field and so makes this scale zero; the CPU
///   kernel does not. The checkpoint pairs `0x00` only with all-zero codes — in
///   `lm_head`'s rows past the unpadded vocabulary — so both readings decode it
///   to zero there, and this follows the exact one.
/// - `0xff` is `2^128`, which overflows to infinity. OCP reserves the byte for
///   NaN; MLX makes it infinity, and a NaN only where it meets the zero code.
///   No scale byte in the checkpoint exceeds `0x82`.
fn block_scale(byte: u8) -> f32 {
    if byte == 0 {
        return SMALLEST_BLOCK_SCALE;
    }
    f32::from_bits(u32::from(byte) << EXPONENT_SHIFT)
}

fn expect_dtype(what: &'static str, expected: Dtype, got: Dtype) -> Result<(), QuantError> {
    if got == expected {
        return Ok(());
    }
    Err(QuantError::WrongDtype {
        what,
        expected,
        got,
    })
}

/// The shape the packed tensor stands for: its last axis holds eight codes per
/// word, and the scales carry one byte per group over that same axis.
fn logical_shape(weights: &[usize], scales: &[usize]) -> Result<Vec<usize>, QuantError> {
    let mismatch = || QuantError::ShapeMismatch {
        weights: weights.to_vec(),
        scales: scales.to_vec(),
    };
    let (&packed_width, rows) = weights.split_last().ok_or_else(mismatch)?;
    let (&groups, scale_rows) = scales.split_last().ok_or_else(mismatch)?;

    let width = packed_width * CODES_PER_WORD;
    if rows != scale_rows || groups * GROUP_SIZE != width {
        return Err(mismatch());
    }
    Ok([rows, &[width]].concat())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::checkpoint::Checkpoint;

    /// Packed slices and MLX's dequantisation of them, from
    /// `just dump-quant-fixture`.
    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../reference/fixtures/mxfp4_dequant.safetensors"
    );

    /// One synthetic group per scale byte, each holding all 16 codes twice, so
    /// value `32 * byte + code` is that code under that scale.
    const GRID: &str = "code_grid";
    const SLICES: [&str; 4] = ["routed_expert", "dense_ffn", "vocab_padding", GRID];

    fn fixture() -> Checkpoint {
        Checkpoint::open(Path::new(FIXTURE)).expect("fixture opens")
    }

    fn decode(ckpt: &Checkpoint, slice: &str) -> Dequantized {
        let weights = ckpt.tensor(&format!("{slice}.weight")).expect("weight");
        let scales = ckpt.tensor(&format!("{slice}.scales")).expect("scales");
        dequantize(&weights, &scales).expect("slice decodes")
    }

    fn expected(ckpt: &Checkpoint, slice: &str) -> (Vec<usize>, Vec<f32>) {
        let view = ckpt
            .tensor(&format!("{slice}.dequantized"))
            .expect("dequantized");
        assert_eq!(view.dtype(), Dtype::F32);
        let values = view
            .data()
            .chunks_exact(size_of::<f32>())
            .map(|b| f32::from_le_bytes(b.try_into().expect("chunked into floats")))
            .collect();
        (view.shape().to_vec(), values)
    }

    /// Compared as bit patterns, not as floats: dequantisation is a table
    /// lookup times a power of two, so anything short of equality is a bug —
    /// and the grid's `0xff` scale decodes to NaN, which no `==` would match.
    fn assert_identical(slice: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{slice}: length");
        for (i, (got, want)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{slice}[{i}]: {got:e} is not {want:e}"
            );
        }
    }

    fn grid_value(values: &[f32], scale_byte: usize, code: usize) -> f32 {
        values[scale_byte * GROUP_SIZE + code]
    }

    #[test]
    fn every_slice_reproduces_mlx_bit_for_bit() {
        let ckpt = fixture();
        for slice in SLICES {
            let got = decode(&ckpt, slice);
            let (shape, want) = expected(&ckpt, slice);
            assert_eq!(got.shape, shape, "{slice}: shape");
            assert_identical(slice, &got.values, &want);
        }
    }

    #[test]
    fn the_zero_codes_keep_their_signs() {
        let values = decode(&fixture(), GRID).values;
        let unscaled = 0x7f;
        assert_eq!(grid_value(&values, unscaled, 0).to_bits(), 0.0f32.to_bits());
        assert_eq!(
            grid_value(&values, unscaled, 8).to_bits(),
            (-0.0f32).to_bits()
        );
    }

    #[test]
    fn the_extreme_scale_bytes_follow_mlx_not_the_specification() {
        let values = decode(&fixture(), GRID).values;

        // 0x00 is 2^-127, not zero: code 2 is 1.0, so it decodes to the scale.
        assert_eq!(grid_value(&values, 0x00, 2), SMALLEST_BLOCK_SCALE);
        assert_eq!(grid_value(&values, 0x00, 1), SMALLEST_BLOCK_SCALE / 2.0);
        assert_eq!(grid_value(&values, 0x00, 0).to_bits(), 0.0f32.to_bits());

        // 0xff is 2^128, which overflows; only the zero code turns it into NaN.
        assert_eq!(grid_value(&values, 0xff, 2), f32::INFINITY);
        assert_eq!(grid_value(&values, 0xff, 10), f32::NEG_INFINITY);
        assert!(grid_value(&values, 0xff, 0).is_nan());
    }

    #[test]
    fn consecutive_groups_step_the_scale_by_one_power_of_two() {
        let values = decode(&fixture(), GRID).values;
        for byte in 0x01..0xfe {
            let (below, above) = (
                grid_value(&values, byte, 2),
                grid_value(&values, byte + 1, 2),
            );
            assert_eq!(above, below * 2.0, "scale byte {byte:#04x}");
        }
    }

    /// A decoder that read one scale per row, or that took the scales an index
    /// out, would still agree with the fixture on a slice whose groups happen
    /// to share a scale. Swapping two adjacent scale bytes has to move both of
    /// their groups and nothing else.
    #[test]
    fn each_group_decodes_under_its_own_scale() {
        let ckpt = fixture();
        let packed = ckpt.tensor("dense_ffn.weight").expect("weight");
        let mut scales = ckpt
            .tensor("dense_ffn.scales")
            .expect("scales")
            .data()
            .to_vec();
        let boundary = scales
            .windows(2)
            .position(|pair| pair[0] != pair[1])
            .expect("some two adjacent groups differ in scale");
        scales.swap(boundary, boundary + 1);

        let swapped = dequantize_blocks(packed.data(), &scales).expect("decodes");
        let (_, want) = expected(&ckpt, "dense_ffn");
        let touched = boundary * GROUP_SIZE..(boundary + 2) * GROUP_SIZE;
        for (i, (got, want)) in swapped.iter().zip(&want).enumerate() {
            // A code that decodes to zero decodes to zero under either scale.
            let expected_to_move = touched.contains(&i) && *want != 0.0;
            assert_eq!(
                got.to_bits() != want.to_bits(),
                expected_to_move,
                "value {i}: {got:e} against {want:e}"
            );
        }
    }

    #[test]
    fn packed_bytes_short_of_a_word_are_refused() {
        let err = dequantize_blocks(&[0; WORDS_PER_GROUP * WORD_BYTES - 1], &[0x7f]).unwrap_err();
        assert!(matches!(err, QuantError::PartialWord(15)), "got {err:?}");
    }

    #[test]
    fn a_length_that_is_not_whole_groups_is_refused() {
        let err = dequantize_blocks(&[0; 5 * WORD_BYTES], &[0x7f; 2]).unwrap_err();
        assert!(
            matches!(err, QuantError::PartialGroup { words: 5 }),
            "got {err:?}"
        );
        assert!(err.to_string().contains("40 codes"));
    }

    #[test]
    fn a_scale_per_group_is_required() {
        let err = dequantize_blocks(&[0; 8 * WORD_BYTES], &[0x7f]).unwrap_err();
        assert!(
            matches!(
                err,
                QuantError::ScaleCountMismatch {
                    groups: 2,
                    scales: 1
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn scales_offered_as_weights_are_refused() {
        let ckpt = fixture();
        let scales = ckpt.tensor("dense_ffn.scales").expect("scales");
        let err = dequantize(&scales, &scales).unwrap_err();
        assert!(
            matches!(err, QuantError::WrongDtype { got: Dtype::U8, .. }),
            "got {err:?}"
        );
        assert!(err.to_string().contains("packed weights"));
    }

    #[test]
    fn weights_paired_with_another_tensors_scales_are_refused() {
        let ckpt = fixture();
        let weights = ckpt.tensor("routed_expert.weight").expect("weight");
        let scales = ckpt.tensor("dense_ffn.scales").expect("scales");

        let err = dequantize(&weights, &scales).unwrap_err();
        assert!(
            matches!(err, QuantError::ShapeMismatch { .. }),
            "got {err:?}"
        );
        assert!(err.to_string().contains("[2, 32, 512]"));
    }

    /// The pair above disagrees on its leading axes too, so it cannot show that
    /// the group count is checked at all. These agree on everything but that:
    /// 512 words are 4096 codes and so need 128 scale bytes, not 64.
    #[test]
    fn scales_too_few_for_the_packed_width_are_refused() {
        let err = logical_shape(&[64, 512], &[64, 64]).unwrap_err();
        assert!(
            matches!(err, QuantError::ShapeMismatch { .. }),
            "got {err:?}"
        );
    }
}
