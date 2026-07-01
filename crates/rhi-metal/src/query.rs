//! GPU timestamp queries for per-pass profiling (Phase 9 M1).
//!
//! A [`MetalQueryHeap`] wraps an `MTLCounterSampleBuffer` of `count` timestamp
//! slots (storage mode `Shared`, counter set = the device's timestamp counter
//! set). The render graph samples a timestamp at each pass boundary
//! ([`crate::command::MetalCommandBuffer::write_timestamp`], which records the
//! sample at the *start-of-encoder* stage boundary of a tiny empty compute pass —
//! the only sampling point Apple-family GPUs expose); the host reads the resolved
//! samples back ([`Self::read`]) once the frame's fence has signalled.
//! `MTLCounterResultTimestamp::timestamp` is already in **GPU nanoseconds**, so
//! [`Self::period_ns`] returns `1.0` (tick == ns) to match the ns-per-tick
//! contract the Vulkan/D3D12 backends expose.
//!
//! Hardware / OS that lacks a stage-boundary timestamp counter set degrades
//! gracefully: the heap has no sample buffer, sampling is a no-op, and
//! [`Self::read`] returns zero ticks (the old stub behaviour), so the backend
//! still runs — the profiler just shows 0 ms as before.

use std::mem;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLCommonCounterSetTimestamp, MTLCounterResultTimestamp, MTLCounterSampleBuffer,
    MTLCounterSampleBufferDescriptor, MTLCounterSamplingPoint, MTLCounterSet, MTLDevice,
    MTLStorageMode,
};

use crate::Result;
use crate::device::DeviceShared;

/// `MTLCounterDontSample` (`(NSUInteger)-1`): a sample-index sentinel meaning
/// "omit this boundary's sample". Not exported by objc2-metal, so defined here.
pub(crate) const COUNTER_DONT_SAMPLE: usize = usize::MAX;

/// `MTLCounterErrorValue` (`~0ULL`): resolved value for a slot whose sample
/// failed; treated as 0 so it never yields a nonsense delta.
const COUNTER_ERROR_VALUE: u64 = u64::MAX;

/// A timestamp query heap of `count` slots, backed by an `MTLCounterSampleBuffer`.
pub struct MetalQueryHeap {
    count: u32,
    /// The counter sample buffer, or `None` when the device supports no
    /// stage-boundary timestamp counter set (heap then reports zero ticks).
    sample_buffer: Option<Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>>,
}

impl MetalQueryHeap {
    pub(crate) fn new(shared: &DeviceShared, count: u32) -> Result<Self> {
        let sample_buffer = create_sample_buffer(&shared.device, count);
        Ok(Self {
            count,
            sample_buffer,
        })
    }

    /// The backing sample buffer, if timestamp sampling is supported. Used by the
    /// command buffer's `write_timestamp` to sample into slot `index`.
    pub(crate) fn sample_buffer(&self) -> Option<&ProtocolObject<dyn MTLCounterSampleBuffer>> {
        self.sample_buffer.as_deref()
    }

    /// Number of timestamp slots.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Nanoseconds per tick. Metal resolves timestamps directly in GPU
    /// nanoseconds, so one tick == one nanosecond (matches the ns-per-tick unit
    /// the Vulkan/D3D12 backends return).
    pub fn period_ns(&self) -> f32 {
        if self.sample_buffer.is_some() {
            1.0
        } else {
            0.0
        }
    }

    /// Read all `count` resolved timestamp ticks (GPU nanoseconds). Call only
    /// after the submission that sampled them has completed (e.g. after the frame
    /// fence). Returns zeros when the device lacks timestamp counters or a resolve
    /// fails, so the caller's `ticks[i+1]-ticks[i]` math yields 0 ms rather than
    /// garbage.
    pub fn read(&self) -> Vec<u64> {
        let mut out = vec![0u64; self.count as usize];
        let Some(sb) = self.sample_buffer.as_deref() else {
            return out;
        };
        // Resolve the full range into an NSData of packed `MTLCounterResultTimestamp`
        // (one u64 each). `resolveCounterRange` may only be called on a Shared buffer
        // (guaranteed at creation). Slots that failed to sample come back as
        // `MTLCounterErrorValue` (u64::MAX); treat those as 0 so a bad slot doesn't
        // produce a nonsense delta.
        let range = NSRange::new(0, self.count as usize);
        let data = unsafe { sb.resolveCounterRange(range) };
        let Some(data) = data else {
            return out;
        };
        let byte_len = data.len();
        let stride = mem::size_of::<MTLCounterResultTimestamp>();
        let n = (byte_len / stride).min(self.count as usize);
        // NSData bytes are contiguous, valid for `byte_len` bytes, and correctly
        // aligned for u64. Copy out each sample's `timestamp` field.
        let bytes = data.to_vec();
        for (i, slot) in out.iter_mut().take(n).enumerate() {
            let off = i * stride;
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[off..off + 8]);
            let t = u64::from_ne_bytes(buf);
            *slot = if t == COUNTER_ERROR_VALUE { 0 } else { t };
        }
        out
    }
}

/// Build an `MTLCounterSampleBuffer` of `count` timestamp slots, or `None` when
/// the device exposes no timestamp counter set / no supported sampling point.
/// Never panics — unsupported hardware falls back to the zero-tick heap.
///
/// Apple-family GPUs support only **stage-boundary** sampling (not blit/dispatch
/// mid-encoder sampling), so [`crate::command::MetalCommandBuffer::write_timestamp`]
/// records each boundary as the *start-of-encoder* of a tiny empty compute pass
/// (see there). Discrete GPUs that report stage-boundary support work the same
/// way. Devices reporting no stage-boundary support get the zero-tick fallback.
fn create_sample_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    count: u32,
) -> Option<Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>> {
    // We sample at a compute pass's start-of-encoder stage boundary, so the device
    // must support stage-boundary counter sampling.
    if !device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary) {
        return None;
    }
    // Find the device's timestamp counter set by matching the common timestamp name.
    let sets = device.counterSets()?;
    let want = unsafe { MTLCommonCounterSetTimestamp };
    let mut timestamp_set: Option<Retained<ProtocolObject<dyn MTLCounterSet>>> = None;
    for set in sets.iter() {
        if set.name().isEqualToString(want) {
            timestamp_set = Some(set);
            break;
        }
    }
    let timestamp_set = timestamp_set?;

    let desc = MTLCounterSampleBufferDescriptor::new();
    desc.setCounterSet(Some(&timestamp_set));
    desc.setStorageMode(MTLStorageMode::Shared);
    // SAFETY: sampleCount is a plain property set; count is the requested slot count.
    unsafe { desc.setSampleCount(count as usize) };

    // `newCounterSampleBufferWithDescriptor:error:` returns Err on unsupported
    // configs; degrade to the zero-tick heap rather than propagating.
    device
        .newCounterSampleBufferWithDescriptor_error(&desc)
        .ok()
}
