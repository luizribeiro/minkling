//! Source string to compute pipeline.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary};

use crate::device::{Device, MetalError};

/// One compiled entry point.
///
/// Compiling produces a whole library and then a pipeline for one function in
/// it. Only the pipeline is kept: the library exists to be searched, and a
/// second entry point out of the same source is a second [`Device::compile`].
#[derive(Debug)]
pub struct Kernel {
    entry: String,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Device {
    /// Compile `source` and take `entry` out of it.
    ///
    /// The three ways this fails are three different mistakes — source that
    /// does not compile, a name that is not in it, and a function that compiles
    /// but cannot be a compute pipeline — and they are three errors, because
    /// the first one carries the compiler's own diagnostic and the others have
    /// nothing to do with it.
    pub fn compile(&self, source: &str, entry: &str) -> Result<Kernel, MetalError> {
        let library = self
            .raw()
            .newLibraryWithSource_options_error(&NSString::from_str(source), None)
            .map_err(|err| MetalError::Compile(diagnostic(&err)))?;

        let function = library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| MetalError::NoSuchKernel(entry.to_owned()))?;

        let pipeline = self
            .raw()
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| MetalError::Pipeline {
                entry: entry.to_owned(),
                diagnostic: diagnostic(&err),
            })?;

        Ok(Kernel {
            entry: entry.to_owned(),
            pipeline,
        })
    }
}

impl Kernel {
    pub fn entry(&self) -> &str {
        &self.entry
    }

    /// The widest threadgroup this kernel can be dispatched in, which is a
    /// property of the compiled kernel and not of the device: register pressure
    /// lowers it below the device's 1024.
    pub fn max_threads_per_group(&self) -> usize {
        self.pipeline.maxTotalThreadsPerThreadgroup()
    }
}

/// What an `NSError` has to say, which for a compile failure is the compiler's
/// own output — file, line, column, caret and all.
fn diagnostic(err: &NSError) -> String {
    err.localizedDescription().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{SAXPY, SAXPY_ENTRY, device};

    #[test]
    fn a_kernel_reports_the_threadgroup_it_can_be_dispatched_in() {
        let Some(device) = device() else { return };

        let kernel = device.compile(SAXPY, SAXPY_ENTRY).expect("saxpy compiles");

        assert_eq!(kernel.entry(), SAXPY_ENTRY);
        assert!(kernel.max_threads_per_group() > 0);
    }

    /// A Metal compile error with the message swallowed is the worst thing this
    /// crate could do to whoever writes the next kernel, so the compiler's own
    /// text has to survive the trip through `NSError`.
    #[test]
    fn a_malformed_kernel_reports_the_compilers_own_diagnostic() {
        let Some(device) = device() else { return };
        let source = "#include <metal_stdlib>\nkernel void broken() { not_a_function(); }\n";

        let err = device
            .compile(source, "broken")
            .expect_err("the source does not compile");

        let message = err.to_string();
        assert!(message.contains("not_a_function"), "{message}");
        assert!(message.contains("program_source:2"), "{message}");
    }

    #[test]
    fn a_missing_entry_point_is_not_a_compile_error() {
        let Some(device) = device() else { return };

        let err = device
            .compile(SAXPY, "daxpy")
            .expect_err("the source has no daxpy");

        assert!(matches!(err, MetalError::NoSuchKernel(name) if name == "daxpy"));
    }

    /// Source can compile, hold the name, and still not be dispatchable: a
    /// vertex function is found by `newFunctionWithName` like any other and
    /// fails only at the pipeline. Three failures, and the caller is told which.
    #[test]
    fn a_function_that_is_not_a_kernel_fails_at_the_pipeline() {
        let Some(device) = device() else { return };
        let source = "#include <metal_stdlib>\nvertex float4 shade() { return float4(0); }\n";

        let err = device
            .compile(source, "shade")
            .expect_err("a vertex function makes no compute pipeline");

        assert!(
            matches!(err, MetalError::Pipeline { ref entry, .. } if entry == "shade"),
            "{err}"
        );
    }
}
