//! Cached environment-variable feature flags.
//!
//! `std::env::var` takes a process-wide lock and allocates a String on every
//! call. Reading these flags once at first access via `LazyLock` removes that
//! cost from per-read hot paths (DP/chain dispatch).

use std::sync::LazyLock;

#[inline(always)]
fn flag(name: &'static str) -> bool {
    std::env::var(name).is_ok()
}

pub static FORCE_SCALAR: LazyLock<bool> = LazyLock::new(|| flag("RAMMAP_FORCE_SCALAR"));
#[cfg(target_arch = "x86_64")]
pub static FORCE_SSE: LazyLock<bool> = LazyLock::new(|| flag("RAMMAP_FORCE_SSE"));
#[cfg(target_arch = "x86_64")]
pub static FORCE_AVX2: LazyLock<bool> = LazyLock::new(|| flag("RAMMAP_FORCE_AVX2"));
#[cfg(target_arch = "x86_64")]
pub static FORCE_AVX512: LazyLock<bool> = LazyLock::new(|| flag("RAMMAP_FORCE_AVX512"));
// wasm always uses the wasm chain kernel, so gate it out there to avoid a dead-code warning
#[cfg(not(target_arch = "wasm32"))]
pub static FORCE_SCALAR_CHAIN: LazyLock<bool> = LazyLock::new(|| flag("RAMMAP_FORCE_SCALAR_CHAIN"));
pub static COMPARE_SCALAR: LazyLock<bool> = LazyLock::new(|| flag("RAMMAP_COMPARE_SCALAR"));
