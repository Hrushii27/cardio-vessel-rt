# CardioRT: Engineering Architecture & Low-Latency Systems Design

## 1. Zero-Allocation Memory Pool Architecture

In hard real-time systems, heap memory allocations inside per-frame processing loops trigger memory fragmentation and non-deterministic Garbage Collection (GC) pauses. In Python/NumPy, repeatedly creating $512\times 512$ or $1024\times 1024$ arrays introduces $5-15\text{ ms}$ of memory allocation and deallocation overhead.

CardioRT implements `PipelineMemoryPool` (`src/cardio_rt/pipeline/memory_pool.py`):
- All working buffers (`norm_in`, `log_od`, `denoised_od`, `spine_od`, `rib_od`, `lung_od`, `vessel_od`, `processed_u16`, `enhanced_u16`) are preallocated contiguously at initialization.
- Reused across all frames with zero heap allocation in the hot loop.
- Supports instant dynamic buffer reallocation only upon resolution change.

---

## 2. Multi-Scale Scale-Decoupled Processing

High-resolution frames ($1024\times 1024$) contain over $1.04$ million pixels. Directly executing 2D Hessian differential filters and multi-level pyramids on full $1024\times 1024$ arrays exceeds the $36\text{ ms}$ latency budget on typical clinical workstations without dedicated server GPUs.

CardioRT solves this via **Scale-Decoupled Multiscale Processing**:
1. Anatomical clutter (vertebral spine, rib shadows, lung field ramps) comprises low-to-medium spatial frequencies.
2. Low-frequency background suppression and vesselness guidance are computed at an internal working scale of $384\times 384$ pixels.
3. The resulting background subtraction map $\Delta OD_{bg}$ and vessel gain map $G(x,y)$ are upsampled to full $1024\times 1024$ resolution using bilinear SIMD interpolation ($1.8\text{ ms}$).
4. High-frequency coronary luminal details are preserved and blended at full resolution.
5. **Latency result:** $1024\times 1024$ execution latency drops from $110\text{ ms}$ to **$17.87\text{ ms}$**, running at **56.0 FPS**!

---

## 3. Dynamic Latency Governor

Under heavy operating system thread scheduling contention or thermal CPU throttling, individual frame times can spike. CardioRT incorporates `LatencyGovernor` (`src/cardio_rt/governor/latency_governor.py`):
- Tracks a rolling window of per-frame latencies ($N=8$).
- If recent latency exceeds $30.0\text{ ms}$ (or any frame hits $36.0\text{ ms}$):
  - Automatically degrades working resolution ($384\to 320\to 256$) and drops higher Hessian scales.
  - Logs structured adaptation event with exact timestamp and frame index.
- If latency stabilizes below $20.0\text{ ms}$:
  - Automatically restores higher fidelity settings.

---

## 4. Hardware Acceleration & Threading Strategy

- **OpenCV SIMD Optimization:** `cv2.setNumThreads(4)` prevents hyperthreaded thread contention and context-switching jitter on 8-core CPUs.
- **Numba Parallel JIT:** Hessian eigenvalue and Frangi response kernels compile to native machine code with `@njit(parallel=True, fastmath=True)`.
- **Precomputed 16-bit LUTs:** Dynamic gamma/tone mapping evaluates 65,536-entry lookup tables in $< 0.8\text{ ms}$ instead of per-pixel floating point power functions.
