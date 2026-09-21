//! Native execution of a BF16 AOTInductor package exported by python/export_mcts.py.
use std::{ffi::{c_char, c_void, CStr, CString}, ptr::NonNull};
use crate::{game::SQUARES, mcts::Evaluate};

extern "C" {
    fn alpha_model_load(path: *const c_char, width: usize, cuda: bool) -> *mut c_void;
    fn alpha_model_free(model: *mut c_void);
    fn alpha_model_error() -> *const c_char;
    fn alpha_model_eval(model: *mut c_void, boards: *const u8, scores: *const i32,
                        priors: *mut f32, values: *mut f32) -> i32;
}

fn error() -> String {
    // The shim owns a thread-local, null-terminated error string.
    unsafe { CStr::from_ptr(alpha_model_error()).to_string_lossy().into_owned() }
}

pub struct CompiledModel {
    handle: NonNull<c_void>,
    width: usize,
}

impl CompiledModel {
    pub fn load(path: &str, width: usize, cuda: bool) -> Result<Self, String> {
        let metadata = std::fs::read_to_string(format!("{path}.meta")).map_err(|e| e.to_string())?;
        let device = if cuda { "cuda" } else { "cpu" };
        if metadata.trim() != format!("alpha-lines-mcts-v1 {width} {device}") {
            return Err("model package batch/device mismatch; export with the requested --batch and --device".into());
        }
        let path = CString::new(path).map_err(|e| e.to_string())?;
        // The C++ loader copies the path; the returned handle is owned until Drop.
        let handle = NonNull::new(unsafe { alpha_model_load(path.as_ptr(), width, cuda) })
            .ok_or_else(error)?;
        Ok(Self { handle, width })
    }
}

impl Evaluate for CompiledModel {
    fn evaluate(&mut self, positions: &[[u8; SQUARES]], scores: &[[i32; 2]],
                priors: &mut [f32], values: &mut [f32]) {
        assert_eq!(positions.len(), self.width);
        assert_eq!(scores.len(), self.width);
        assert_eq!(priors.len(), 2 * self.width * SQUARES);
        assert_eq!(values.len(), 2 * self.width);
        // Fixed-size Rust arrays are contiguous. All buffers stay alive for this synchronous
        // call, and the shim checks model output sizes before copying into the output slices.
        let status = unsafe { alpha_model_eval(self.handle.as_ptr(), positions.as_ptr().cast(),
            scores.as_ptr().cast(), priors.as_mut_ptr(), values.as_mut_ptr()) };
        assert_eq!(status, 0, "compiled inference failed: {}", error());
    }
}

impl Drop for CompiledModel {
    fn drop(&mut self) {
        unsafe { alpha_model_free(self.handle.as_ptr()); }
    }
}
