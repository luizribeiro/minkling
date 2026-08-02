use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::{SafeTensorError, SafeTensors, tensor::Metadata};
use serde::Deserialize;

pub use safetensors::Dtype;

const INDEX_FILE: &str = "model.safetensors.index.json";
const SHARD_EXTENSION: &str = "safetensors";
const HEADER_LEN_FIELD: usize = size_of::<u64>();

#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("no checkpoint at {}", .0.display())]
    NotFound(PathBuf),

    #[error("{} holds neither {INDEX_FILE} nor any *.{SHARD_EXTENSION}", .0.display())]
    NoTensorFiles(PathBuf),

    #[error("{} is not a readable shard index: {source}", .path.display())]
    MalformedIndex {
        path: PathBuf,
        source: serde_json::Error,
    },

    #[error("{} lists shard {}, which is not on disk", .index.display(), .shard.display())]
    MissingShard { index: PathBuf, shard: PathBuf },

    #[error("{} is not a readable safetensors file: {source}", .path.display())]
    MalformedShard {
        path: PathBuf,
        source: SafeTensorError,
    },

    #[error("{name} appears in both {} and {}", .first.display(), .second.display())]
    DuplicateTensor {
        name: String,
        first: PathBuf,
        second: PathBuf,
    },

    #[error("no tensor named {0} in this checkpoint")]
    NoSuchTensor(String),

    #[error("cannot read {}: {source}", .path.display())]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

/// A checkpoint opened for reading, either a directory of index-listed shards
/// or a single `*.safetensors` file.
///
/// Opening maps every shard and reads its header — a few KiB each — but no
/// tensor bytes. The slice handed out by [`Checkpoint::tensor`] faults its
/// pages in on first touch, so a 140 GB checkpoint costs nothing to open.
pub struct Checkpoint {
    shards: Vec<Shard>,
    owners: BTreeMap<String, usize>,
}

/// Bytes one bfloat16 value occupies.
pub const BF16_BYTES: usize = size_of::<u16>();

/// Where a bfloat16's bits sit in the float32 it widens to, which is the whole
/// of the format: an f32 with the low sixteen mantissa bits dropped.
///
/// Named rather than only used, because a kernel that widens the same bytes
/// where it multiplies needs these two facts too — and a second reading of the
/// format living in a source string is one that can drift from this one.
pub const BF16_SHIFT: u32 = 16;

/// The dtype, shape and undecoded bytes of one tensor.
///
/// The bytes are exactly as they sit in the file: MXFP4 blocks stay packed and
/// bf16 stays 16-bit. Decoding that needs more than these bytes — MXFP4, which
/// wants the block scales stored beside them — belongs to the layer that knows
/// what it wants; widening a float dtype needs nothing else and lives here.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    dtype: Dtype,
    shape: &'a [usize],
    data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn shape(&self) -> &'a [usize] {
        self.shape
    }

    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// A `BF16` or `F32` tensor's values, widened to f32. `None` for anything
    /// else, which for an Inkling checkpoint means a packed MXFP4 tensor —
    /// those decode through [`crate::quant`], which needs their scales too.
    ///
    /// Widening bfloat16 is exact and needs no table: it is an f32 with the low
    /// sixteen mantissa bits dropped, so putting them back is a shift.
    pub fn to_f32(&self) -> Option<Vec<f32>> {
        match self.dtype {
            Dtype::F32 => Some(
                self.data
                    .chunks_exact(size_of::<f32>())
                    .map(|b| f32::from_le_bytes(b.try_into().expect("chunked into floats")))
                    .collect(),
            ),
            Dtype::BF16 => Some(
                self.data
                    .chunks_exact(BF16_BYTES)
                    .map(|b| {
                        let bits = u16::from_le_bytes(b.try_into().expect("chunked into halves"));
                        f32::from_bits(u32::from(bits) << BF16_SHIFT)
                    })
                    .collect(),
            ),
            _ => None,
        }
    }
}

impl Checkpoint {
    /// Open a checkpoint directory or a single `*.safetensors` file.
    ///
    /// Tensor metadata comes from the shard headers rather than from the
    /// index's `weight_map`: the headers are what the bytes are addressed
    /// against, so a stale or truncated index cannot silently misplace a
    /// tensor.
    ///
    /// **Every `*.safetensors` file in the directory is mapped, listed or
    /// not.** The index says what a checkpoint's shards *are*; the directory
    /// says what it *has*, and a published Inkling checkpoint has more than its
    /// index lists — `mtp.safetensors` is 160 tensors the mxfp4 index never
    /// names. What the index is still read for is the one question the
    /// directory cannot answer: whether a shard it names is missing. A file
    /// restating a tensor another one holds is
    /// [`CheckpointError::DuplicateTensor`] either way.
    pub fn open(path: &Path) -> Result<Self, CheckpointError> {
        let mut shards: Vec<Shard> = Vec::new();
        let mut owners: BTreeMap<String, usize> = BTreeMap::new();

        for shard_path in shard_paths(path)? {
            let shard = Shard::open(&shard_path)?;
            let index = shards.len();
            for name in shard.metadata.offset_keys() {
                if let Some(&first) = owners.get(&name) {
                    return Err(CheckpointError::DuplicateTensor {
                        name,
                        first: shards[first].path.clone(),
                        second: shard.path,
                    });
                }
                owners.insert(name, index);
            }
            shards.push(shard);
        }

        Ok(Self { shards, owners })
    }

    /// Every tensor in the checkpoint, in name order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.owners.keys().map(String::as_str)
    }

    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, CheckpointError> {
        self.owners
            .get(name)
            .and_then(|&shard| self.shards[shard].view(name))
            .ok_or_else(|| CheckpointError::NoSuchTensor(name.to_owned()))
    }
}

impl std::fmt::Debug for Checkpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Checkpoint")
            .field("shards", &self.shards.len())
            .field("tensors", &self.owners.len())
            .finish()
    }
}

struct Shard {
    path: PathBuf,
    map: Mmap,
    data_start: usize,
    metadata: Metadata,
}

impl Shard {
    fn open(path: &Path) -> Result<Self, CheckpointError> {
        let io = io_error(path);
        let file = File::open(path).map_err(&io)?;
        // SAFETY: the mapping aliases the file's pages, so a concurrent writer
        // truncating or rewriting the checkpoint would be undefined behaviour.
        // Checkpoints are read-only artefacts for the lifetime of a process.
        let map = unsafe { Mmap::map(&file) }.map_err(&io)?;

        let (header_len, metadata) =
            SafeTensors::read_metadata(&map).map_err(|source| CheckpointError::MalformedShard {
                path: path.to_owned(),
                source,
            })?;

        Ok(Self {
            path: path.to_owned(),
            map,
            data_start: HEADER_LEN_FIELD + header_len,
            metadata,
        })
    }

    fn view(&self, name: &str) -> Option<TensorView<'_>> {
        let info = self.metadata.info(name)?;
        let (start, end) = info.data_offsets;
        Some(TensorView {
            dtype: info.dtype,
            shape: &info.shape,
            // `read_metadata` rejects offsets that overrun the file.
            data: &self.map[self.data_start + start..self.data_start + end],
        })
    }
}

#[derive(Deserialize)]
struct ShardIndex {
    weight_map: BTreeMap<String, String>,
}

fn io_error(path: &Path) -> impl Fn(std::io::Error) -> CheckpointError + '_ {
    move |source| CheckpointError::Io {
        path: path.to_owned(),
        source,
    }
}

fn shard_paths(path: &Path) -> Result<Vec<PathBuf>, CheckpointError> {
    if path.is_file() {
        return Ok(vec![path.to_owned()]);
    }
    if !path.is_dir() {
        return Err(CheckpointError::NotFound(path.to_owned()));
    }

    let shards = loose_shard_paths(path)?;
    let index = path.join(INDEX_FILE);
    if index.is_file() {
        indexed_shard_paths(&index, path)?;
    }

    if shards.is_empty() {
        return Err(CheckpointError::NoTensorFiles(path.to_owned()));
    }
    Ok(shards)
}

fn loose_shard_paths(dir: &Path) -> Result<Vec<PathBuf>, CheckpointError> {
    let entries = std::fs::read_dir(dir).map_err(io_error(dir))?;
    let mut loose: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == SHARD_EXTENSION))
        .collect();
    loose.sort();
    Ok(loose)
}

/// Every shard the index names, which is read for whether they are all there.
fn indexed_shard_paths(index: &Path, dir: &Path) -> Result<Vec<PathBuf>, CheckpointError> {
    let text = std::fs::read_to_string(index).map_err(io_error(index))?;
    let parsed: ShardIndex =
        serde_json::from_str(&text).map_err(|source| CheckpointError::MalformedIndex {
            path: index.to_owned(),
            source,
        })?;

    let mut names: Vec<&str> = parsed.weight_map.values().map(String::as_str).collect();
    names.sort_unstable();
    names.dedup();

    names
        .into_iter()
        .map(|name| {
            let shard = dir.join(name);
            if shard.is_file() {
                Ok(shard)
            } else {
                Err(CheckpointError::MissingShard {
                    index: index.to_owned(),
                    shard,
                })
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;
    use crate::fixture;

    /// The committed oracle dump: a single file, no index.
    const FIXTURE: &str = "layer_activations.safetensors";

    fn fixture() -> Checkpoint {
        fixture::open(FIXTURE)
    }

    struct Blob {
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl safetensors::View for &Blob {
        fn dtype(&self) -> Dtype {
            self.dtype
        }
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }
        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    fn write_shard(path: &Path, name: &str, byte: u8) {
        let blob = Blob {
            dtype: Dtype::U32,
            shape: vec![2, 2],
            data: vec![byte; 16],
        };
        safetensors::serialize_to_file([(name, &blob)], None, path).expect("shard is written");
    }

    /// Two shards, one tensor each, wired together by an index.
    fn sharded_checkpoint(dir: &Path) {
        write_shard(&dir.join("model-00001-of-00002.safetensors"), "first", 0xaa);
        write_shard(
            &dir.join("model-00002-of-00002.safetensors"),
            "second",
            0xbb,
        );
        std::fs::write(
            dir.join(INDEX_FILE),
            r#"{"metadata": {"total_size": 32}, "weight_map": {
                 "first":  "model-00001-of-00002.safetensors",
                 "second": "model-00002-of-00002.safetensors"
               }}"#,
        )
        .expect("index is written");
    }

    #[test]
    fn single_file_layout_needs_no_index() {
        let ckpt = fixture();
        assert_eq!(ckpt.num_shards(), 1);
        assert_eq!(ckpt.tensor_names().count(), 80);
    }

    #[test]
    fn views_carry_dtype_shape_and_undecoded_bytes() {
        let ckpt = fixture();

        let embed = ckpt.tensor("embed_out").expect("embed_out");
        assert_eq!(embed.dtype(), Dtype::F32);
        assert_eq!(embed.shape(), [1, 8, 4096]);
        assert_eq!(embed.data().len(), 8 * 4096 * 4);

        let ids = ckpt.tensor("input_ids").expect("input_ids");
        assert_eq!(ids.dtype(), Dtype::I32);
        assert_eq!(ids.shape(), [1, 8]);
        assert_eq!(ids.data().len(), 8 * 4);
    }

    /// Widening bfloat16 is a shift, and every way to get it wrong — the byte
    /// order, the shift's direction, half a nibble out — still produces finite
    /// numbers of a plausible size. The exponent and mantissa fields have to
    /// land where f32 keeps them, and the sign has to survive.
    #[test]
    fn bfloat16_widens_by_restoring_the_mantissa_bits_it_dropped() {
        // 1.0, -2.5, +0, -0, the smallest normal 2^-126, and 1 + 2^-7 — the
        // next value up, whose set bit is the last of bfloat16's seven.
        let halves: [u16; 6] = [0x3f80, 0xc020, 0x0000, 0x8000, 0x0080, 0x3f81];
        let data: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
        let view = TensorView {
            dtype: Dtype::BF16,
            shape: &[6],
            data: &data,
        };

        let got = view.to_f32().expect("bfloat16 widens");
        assert_eq!(
            got,
            [1.0, -2.5, 0.0, -0.0, f32::MIN_POSITIVE, 1.0 + 2f32.powi(-7)]
        );
        assert_eq!(got[3].to_bits(), (-0.0f32).to_bits(), "the sign of zero");
    }

    /// A packed MXFP4 tensor means nothing without the scales stored beside it,
    /// so it is refused here rather than read as raw bytes.
    #[test]
    fn a_packed_tensor_does_not_widen() {
        let ckpt = fixture::open("mxfp4_dequant.safetensors");
        let packed = ckpt.tensor("dense_ffn.weight").expect("a packed slice");
        assert_eq!(packed.dtype(), Dtype::U32);
        assert!(packed.to_f32().is_none());
    }

    #[test]
    fn absent_tensor_is_a_typed_error() {
        let err = fixture().tensor("layer0.nonesuch").unwrap_err();
        assert!(
            matches!(&err, CheckpointError::NoSuchTensor(name) if name == "layer0.nonesuch"),
            "got {err:?}"
        );
        assert!(err.to_string().contains("layer0.nonesuch"));
    }

    #[test]
    fn sharded_layout_resolves_tensors_to_their_own_shard() {
        let dir = tempfile::tempdir().expect("tempdir");
        sharded_checkpoint(dir.path());

        let ckpt = Checkpoint::open(dir.path()).expect("sharded checkpoint opens");
        assert_eq!(ckpt.num_shards(), 2);
        assert_eq!(ckpt.tensor_names().collect::<Vec<_>>(), ["first", "second"]);
        assert_eq!(ckpt.tensor("first").expect("first").data(), [0xaa; 16]);
        assert_eq!(ckpt.tensor("second").expect("second").data(), [0xbb; 16]);
    }

    #[test]
    fn loose_shards_are_discovered_without_an_index() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_shard(
            &dir.path().join("model-00002-of-00002.safetensors"),
            "second",
            0xbb,
        );
        write_shard(
            &dir.path().join("model-00001-of-00002.safetensors"),
            "first",
            0xaa,
        );
        std::fs::write(dir.path().join("config.json"), "{}").expect("non-shard file is written");

        let ckpt = Checkpoint::open(dir.path()).expect("loose checkpoint opens");
        assert_eq!(ckpt.num_shards(), 2);
        assert_eq!(ckpt.tensor_names().collect::<Vec<_>>(), ["first", "second"]);
        assert_eq!(ckpt.tensor("second").expect("second").data(), [0xbb; 16]);
    }

    /// The shape both published Inkling quantisations ship: an index that names
    /// the main stack's thirty-odd shards, and `mtp.safetensors` beside it
    /// holding 160 tensors it does not name. A loader that mapped the index's
    /// list would open such a checkpoint, run it, and find no MTP head in it.
    #[test]
    fn a_shard_the_index_does_not_list_is_still_the_checkpoints() {
        let dir = tempfile::tempdir().expect("tempdir");
        sharded_checkpoint(dir.path());
        write_shard(&dir.path().join("mtp.safetensors"), "head", 0xcc);

        let ckpt = Checkpoint::open(dir.path()).expect("checkpoint opens");
        assert_eq!(ckpt.num_shards(), 3);
        assert_eq!(
            ckpt.tensor_names().collect::<Vec<_>>(),
            ["first", "head", "second"]
        );
        assert_eq!(ckpt.tensor("head").expect("head").data(), [0xcc; 16]);
    }

    #[test]
    fn a_tensor_claimed_by_two_shards_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_shard(
            &dir.path().join("model-00001-of-00002.safetensors"),
            "first",
            0xaa,
        );
        write_shard(
            &dir.path().join("model-00002-of-00002.safetensors"),
            "first",
            0xbb,
        );

        let err = Checkpoint::open(dir.path()).unwrap_err();
        assert!(
            matches!(&err, CheckpointError::DuplicateTensor { name, .. } if name == "first"),
            "got {err:?}"
        );
        assert!(err.to_string().contains("model-00002-of-00002.safetensors"));
    }

    #[test]
    fn an_index_listing_no_shards_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join(INDEX_FILE), r#"{"weight_map": {}}"#)
            .expect("index is written");

        let err = Checkpoint::open(dir.path()).unwrap_err();
        assert!(
            matches!(err, CheckpointError::NoTensorFiles(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn shard_listed_but_absent_names_the_shard() {
        let dir = tempfile::tempdir().expect("tempdir");
        sharded_checkpoint(dir.path());
        std::fs::remove_file(dir.path().join("model-00002-of-00002.safetensors")).expect("removed");

        let err = Checkpoint::open(dir.path()).unwrap_err();
        assert!(
            matches!(err, CheckpointError::MissingShard { .. }),
            "got {err:?}"
        );
        assert!(err.to_string().contains("model-00002-of-00002.safetensors"));
    }

    #[test]
    fn malformed_index_names_the_index() {
        let dir = tempfile::tempdir().expect("tempdir");
        sharded_checkpoint(dir.path());
        std::fs::write(dir.path().join(INDEX_FILE), "{\"weight_map\": [")
            .expect("index is written");

        let err = Checkpoint::open(dir.path()).unwrap_err();
        assert!(
            matches!(err, CheckpointError::MalformedIndex { .. }),
            "got {err:?}"
        );
        assert!(err.to_string().contains(INDEX_FILE));
    }

    #[test]
    fn truncated_shard_names_the_shard() {
        let dir = tempfile::tempdir().expect("tempdir");
        let shard = dir.path().join("model.safetensors");
        write_shard(&shard, "first", 0xaa);
        let full = std::fs::read(&shard).expect("read back");
        std::fs::write(&shard, &full[..full.len() - 4]).expect("truncated");

        let err = Checkpoint::open(dir.path()).unwrap_err();
        assert!(
            matches!(err, CheckpointError::MalformedShard { .. }),
            "got {err:?}"
        );
        assert!(err.to_string().contains("model.safetensors"));
    }

    #[test]
    fn directory_without_tensors_is_distinct_from_a_missing_one() {
        let dir = tempfile::tempdir().expect("tempdir");
        let empty = Checkpoint::open(dir.path()).unwrap_err();
        assert!(
            matches!(empty, CheckpointError::NoTensorFiles(_)),
            "got {empty:?}"
        );

        let absent = Checkpoint::open(&dir.path().join("nowhere")).unwrap_err();
        assert!(
            matches!(absent, CheckpointError::NotFound(_)),
            "got {absent:?}"
        );
        assert!(absent.to_string().contains("nowhere"));
    }
}
