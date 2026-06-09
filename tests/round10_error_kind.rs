//! Round-10 typed-CUDA-error coverage.
//!
//! `NvError` has historically forced callers to substring-match
//! `message` to distinguish "no NVIDIA stack" from "unexpected driver
//! failure". Round 10 adds [`CudaErrorKind`], a typed view of the
//! underlying `CUresult` that callers can `match` on directly. These
//! tests exercise the mapping in isolation: every test runs on every
//! host (no GPU required) because [`CudaErrorKind::from_cu`] is a pure
//! lookup over the public CUDA driver-API codes.
//!
//! Gated to `cfg(target_os = "linux")` because the crate body is
//! Linux-only; the `registry` feature isn't needed for the typed-kind
//! mapping itself, but we keep the file under the same gate as the
//! other integration tests so the suite has a single platform story.

#![cfg(target_os = "linux")]

use oxideav_nvidia::CudaErrorKind;

/// Each named variant must round-trip through `from_cu` ↔ `as_code`.
/// `Other(c)` echoes the wrapped value unchanged.
#[test]
fn cuda_error_kind_round_trips() {
    let cases: &[(i32, CudaErrorKind)] = &[
        (0, CudaErrorKind::Success),
        (-1, CudaErrorKind::FrameworkLoad),
        (1, CudaErrorKind::InvalidValue),
        (2, CudaErrorKind::OutOfMemory),
        (3, CudaErrorKind::NotInitialized),
        (4, CudaErrorKind::Deinitialized),
        (100, CudaErrorKind::NoDevice),
        (101, CudaErrorKind::InvalidDevice),
        (201, CudaErrorKind::InvalidContext),
        (800, CudaErrorKind::NotPermitted),
        (801, CudaErrorKind::NotSupported),
        (999, CudaErrorKind::Unknown),
    ];
    for &(code, expected) in cases {
        let kind = CudaErrorKind::from_cu(code);
        assert_eq!(
            kind, expected,
            "from_cu({code}) -> {kind:?}, expected {expected:?}"
        );
        assert_eq!(
            kind.as_code(),
            code,
            "as_code(round-trip) -> {} != {}",
            kind.as_code(),
            code
        );
    }
}

/// Unknown / unnamed driver codes flow through as `Other(c)` with the
/// raw value preserved. Pick a few that the bridge does not name
/// explicitly: `200` (`CUDA_ERROR_INVALID_IMAGE`), `400`
/// (`CUDA_ERROR_INVALID_HANDLE`), `500` (`CUDA_ERROR_NOT_FOUND`).
#[test]
fn cuda_error_kind_other_preserves_raw_code() {
    for code in [200, 400, 500, 12345] {
        let kind = CudaErrorKind::from_cu(code);
        assert_eq!(kind, CudaErrorKind::Other(code));
        assert_eq!(kind.as_code(), code);
    }
}

/// `is_unavailable()` must light up only on the three "no NVIDIA stack
/// present" kinds — FrameworkLoad / NoDevice / NotInitialized — and
/// stay quiet for every other named variant.
#[test]
fn cuda_error_kind_is_unavailable_matches_documented_set() {
    let unavailable = &[
        CudaErrorKind::FrameworkLoad,
        CudaErrorKind::NoDevice,
        CudaErrorKind::NotInitialized,
    ];
    for k in unavailable {
        assert!(
            k.is_unavailable(),
            "{k:?} should be is_unavailable() but wasn't"
        );
    }
    let not_unavailable = &[
        CudaErrorKind::Success,
        CudaErrorKind::InvalidValue,
        CudaErrorKind::OutOfMemory,
        CudaErrorKind::Deinitialized,
        CudaErrorKind::InvalidDevice,
        CudaErrorKind::InvalidContext,
        CudaErrorKind::NotPermitted,
        CudaErrorKind::NotSupported,
        CudaErrorKind::Unknown,
        CudaErrorKind::Other(42),
    ];
    for k in not_unavailable {
        assert!(
            !k.is_unavailable(),
            "{k:?} should NOT be is_unavailable() but was"
        );
    }
}

/// On a host with no NVIDIA driver, `Cuda::init()` must produce an
/// `NvError` whose typed `kind()` lands on one of the unavailable
/// variants. Gated behind the `registry` feature because that's where
/// `Cuda` lives in the test surface; skipped cleanly on hosts that
/// happen to have a working CUDA driver.
#[cfg(feature = "registry")]
#[test]
fn cuda_init_error_kind_is_unavailable_on_no_gpu_hosts() {
    use oxideav_nvidia::Cuda;
    match Cuda::init() {
        Ok(_) => {
            eprintln!(
                "cuda_init_error_kind_is_unavailable_on_no_gpu_hosts: skipping — CUDA available"
            );
        }
        Err(e) => {
            let kind = e.kind();
            eprintln!("Cuda::init() error: code={} kind={:?}", e.code, kind);
            assert!(
                kind.is_unavailable(),
                "expected an `is_unavailable` kind on a no-NVIDIA host, got {kind:?} (code {})",
                e.code
            );
            // Backward-compat: the legacy substring helper must agree
            // with the typed accessor on the no-GPU host path.
            assert!(
                e.is_unavailable(),
                "NvError::is_unavailable() must remain true for the no-GPU error path"
            );
        }
    }
}
