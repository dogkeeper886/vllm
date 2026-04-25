# CUTLASS arch-trait system — sm_37 port reconnaissance

**Story:** [#19][issue-19] (Phase 0 of epic [#12][epic])
**CUTLASS commit pinned for citations:** `7a9fe055cb69ab2de605a0cf7dbb33f27833f7f3` (NVIDIA/cutlass `main`, 2026-04-24).
**Methodology:** every claim cites CUTLASS source by `file:line`. No claim without a citation. Reasoning from upstream comments alone is not sufficient evidence — the K80 fork itself proves vLLM's "K80 cannot run" comment was wrong, so that pattern is the rule for this whole epic.

[issue-19]: https://github.com/dogkeeper886/vllm/issues/19
[epic]: https://github.com/dogkeeper886/vllm/issues/12

---

## 1. Scope

This doc answers Story 0.1's questions for CUTLASS:

1. Where do `cutlass::arch::Sm*` tag classes live?
2. How does GEMM dispatch consume the `ArchTag`?
3. How does CUTLASS pick SIMT vs tensor-op?
4. What does the `Sm50` path actually require from hardware?
5. What CUDA features does the `Sm50` path use that sm_37 might lack?
6. What would adding an `Sm37` tag touch?
7. How does the build system handle the SM list?

These answers feed into Phase 1 stories [#25][s11]–[#30][s16] (CUTLASS sm_37 enablement). They do **not** by themselves justify GO on Phase 1 — that's Story 0.6's call after all Phase 0 reconnaissance is in.

[s11]: https://github.com/dogkeeper886/vllm/issues/25
[s16]: https://github.com/dogkeeper886/vllm/issues/30

## 2. Files / locations summary

The whole system is small. Two files do most of the work; everything else hangs off them.

| Component | File | Citation |
|---|---|---|
| **Arch tag definitions** | `include/cutlass/arch/arch.h` | lines 68–114 |
| **Main GEMM kernel template** | `include/cutlass/gemm/kernel/default_gemm.h` | line 104 (template param), line 831 (SIMT gate) |
| **Threadblock MMA dispatcher** | `include/cutlass/gemm/threadblock/default_mma.h` | lines 48–51, 65–110, 227–231, 431–435 |
| **MMA core (SIMT)** | `include/cutlass/gemm/threadblock/default_mma_core_simt.h` | (whole file) |
| **MMA core (Sm70 tensor-op)** | `include/cutlass/gemm/threadblock/default_mma_core_sm70.h` | (whole file) |
| **MMA core (Sm80 tensor-op)** | `include/cutlass/gemm/threadblock/default_mma_core_sm80.h` | (whole file) |
| **Warp-level SIMT MMA** | `include/cutlass/gemm/warp/mma_simt.h` | line 114 |
| **Sm50 thread-MMA scalar ops** | `include/cutlass/arch/mma_sm50.h` | lines 61–143 |
| **Sm70 HMMA tensor-core ops** | `include/cutlass/arch/mma_sm70.h` | (whole file) |
| **Sm80 HMMA / IMMA / TF32 ops** | `include/cutlass/arch/mma_sm80.h` | (whole file) |
| **Sm90 GMMA / TMA ops** | `include/cutlass/arch/mma_sm90.h` | (whole file) |
| **Universal-adapter dispatch gate** | `include/cutlass/gemm/device/gemm_universal_adapter.h` | lines 388, 401–403 |
| **Build / CMake** | `CMakeLists.txt` | lines 86–90 (CUDA min), 609–662 (arch handling) |

There is **no** Python autogen script in CUTLASS comparable to XFormers' `generate_kernels.py`. Per-arch instantiation is by hand-authored test files and C++ template specialization.

## 3. The arch tag classes

Every `Sm*` tag is a single-member struct with no behaviour beyond declaring its compute-capability number. Definitions in `include/cutlass/arch/arch.h:68–114`:

```cpp
// arch.h:68-70
struct Sm50 { static int const kMinComputeCapability = 50; };

// arch.h:71-73
struct Sm60 { static int const kMinComputeCapability = 60; };

// arch.h:74-76
struct Sm61 { static int const kMinComputeCapability = 61; };

// arch.h:77-79
struct Sm70 { static int const kMinComputeCapability = 70; };

// arch.h:80-82
struct Sm72 { static int const kMinComputeCapability = 72; };

// arch.h:83-85
struct Sm75 { static int const kMinComputeCapability = 75; };

// arch.h:86-88
struct Sm80 { static int const kMinComputeCapability = 80; };

// arch.h:89-91
struct Sm86 { static int const kMinComputeCapability = 86; };

// arch.h:92-94
struct Sm89 { static int const kMinComputeCapability = 89; };

// arch.h:95-97
struct Sm90 { static int const kMinComputeCapability = 90; };

// arch.h:100-102
struct Sm100 { static int const kMinComputeCapability = 100; };

// arch.h:104-106
struct Sm101 { static int const kMinComputeCapability = 101; };

// arch.h:108-110
struct Sm120 { static int const kMinComputeCapability = 120; };

// arch.h:112-114
struct Sm103 { static int const kMinComputeCapability = 103; };
```

**No common base class. No virtual interface. No required methods.** They are pure type tags — used as template parameters for partial specialization further down the chain.

This is unambiguous: **adding `struct Sm37 { static int const kMinComputeCapability = 37; };` is a one-line edit.** No interface to satisfy. The complexity (if any) is downstream.

## 4. GEMM dispatch — how `ArchTag` flows to a concrete kernel

### 4.1 Template parameter entry point

`include/cutlass/gemm/kernel/default_gemm.h:104` declares `typename ArchTag` as a template parameter on `DefaultGemm<>`.

### 4.2 The SIMT gate

The single most important line for our purposes is `default_gemm.h:831`:

```cpp
typename platform::enable_if< ! platform::is_same<ArchTag, arch::Sm80>::value >::type
```

This `enable_if` selects the SIMT specialization of `DefaultGemm` for **every arch tag that is not `Sm80`**. So Sm50, Sm60, Sm61, Sm70, Sm72, Sm75, Sm86, Sm89, Sm90, Sm100, Sm101, Sm103, Sm120 — and a hypothetical Sm37 — all match this gate by default.

In other words: **the default GEMM dispatch routes any non-Sm80 arch through the SIMT path.** A new Sm37 tag would be picked up by this gate automatically with no further changes to `default_gemm.h`.

This is counterintuitive — you would expect tensor-op-capable arches like Sm70 or Sm90 to take a tensor-op path here. The actual story is that tensor-op kernels are *additional* specializations (registered via `OpClassTensorOp` rather than `OpClassSimt`) that override the SIMT default when explicitly requested. The default for an arch tag alone is SIMT.

### 4.3 Threadblock MMA selection

Inside the SIMT specialization (`default_gemm.h:838–859`), `DefaultMma<..., OpClassSimt, Sm50, ...>` is referenced — note the **hard-coded `Sm50`**, not `ArchTag`. This is where the chain stops being arch-parameterized.

`include/cutlass/gemm/threadblock/default_mma.h:48–51` conditionally includes the MMA core headers:
- `default_mma_core_simt.h` (SIMT, all archs)
- `default_mma_core_sm70.h` (Sm70 / Sm75 tensor-op)
- `default_mma_core_sm80.h` (Sm80+ tensor-op)

Specializations at `default_mma.h:227–231` (tensor-op) and `default_mma.h:431–435` (SIMT) pick which header's templates apply.

### 4.4 Warp-level MMA

`include/cutlass/gemm/warp/mma_simt.h:114` hard-codes `ArchTag = arch::Sm50` on the warp-level SIMT MMA. **This is a label, not a hardware contract.** The warp MMA's actual operations are scalar FMA (see §5).

### 4.5 The chain, summarized

```
DefaultGemm<ArchTag = Sm{anything except 80}>
  → default_gemm.h:831 SIMT gate (matches)
  → DefaultMma<..., OpClassSimt, Sm50, ...>     [arch param hard-coded to Sm50]
  → DefaultMmaCore<..., OpClassSimt, ...>
  → default_mma_core_simt.h
  → MmaSimt warp MMA                            [mma_simt.h:114, ArchTag = Sm50 again]
  → arch::Mma<GemmShape<1,1,1>, ...>            [mma_sm50.h, scalar FMA]
```

So once the dispatch enters the SIMT path, **`ArchTag` stops mattering**. The actual kernel uses scalar FP32 FMA, which sm_37 supports natively. A new Sm37 tag would walk this same chain and produce the same scalar FMA code.

## 5. SIMT vs tensor-op gating — when does CUTLASS reach for tensor cores?

Tensor-op selection is by **explicit specialization on `OpClass`**, not by `ArchTag` alone. The dispatch is:

- `OpClassSimt` ⇒ SIMT path (works for any arch)
- `OpClassTensorOp` ⇒ requires arch ≥ Sm70, picks the tightest matching `mma_core_sm{70,80}.h`
- `OpClassWmmaTensorOp` ⇒ Sm70+ via WMMA API
- (newer) `OpClassTfBlocked`, etc. ⇒ Sm90+

The `gemm_universal_adapter.h` device-level dispatch gates on `kMinComputeCapability` directly:

```cpp
// gemm_universal_adapter.h:388
if constexpr (GemmKernel::ArchTag::kMinComputeCapability >= 90)
```

```cpp
// gemm_universal_adapter.h:401-403  (Sm100/101/103 dynamic-cluster checks)
```

For an `Sm37` tag (`kMinComputeCapability = 37`), every `>= 80`, `>= 90`, `>= 100` gate evaluates false. The kernel automatically takes the legacy launch path. No code edits needed to **avoid** newer-arch features — they exclude themselves correctly when the new tag has a low number.

## 6. What does the Sm50 SIMT path actually use from hardware?

This is the load-bearing question. Adding `Sm37` is trivial *if* the SIMT path's actual instruction stream is sm_37-compatible. Reading the code:

### 6.1 Scalar FMA only — confirmed

`include/cutlass/arch/mma_sm50.h:61–143` defines the thread-level MMA operators for the Sm50 SIMT path. They are scalar 1×1×1 multiply-accumulate operations: a single `float`, `double`, or integer FMA per thread. **No inline PTX; no tensor-core intrinsics; no `mma.*` instructions.** The compiler emits ordinary `fma.f32` instructions, which Kepler sm_37 has supported since CUDA 7.

### 6.2 No `cp.async` in the Sm50 path

`cp.async` (the asynchronous global → shared memory copy instruction, sm_80+) appears only in multistage pipelines and tensor-op kernels:

- `include/cutlass/gemm/threadblock/mma_with_reduction_multistage.h:86, 126, 240, 243, 275, 278`

These are not reachable from the SIMT default path. The base SIMT MMA uses ordinary global / shared loads with explicit `__syncthreads()`. **Sm37 is safe here.**

### 6.3 No tensor-core intrinsics in the Sm50 path

`mma.*` and `wmma.*` PTX instructions appear in `mma_sm70.h`, `mma_sm75.h`, `mma_sm80.h`, `mma_sm90.h` — none in `mma_sm50.h` and none reachable through the SIMT chain.

### 6.4 ⚠️ `__shfl_sync` does appear in CUTLASS — but in utility code, not the SIMT MMA

This is the one risk to flag for the port. `__shfl_sync` and friends are **sm_70+ only**. They appear in:

- `include/cutlass/gemm/kernel/grouped_problem_visitor.h:241, 248, 289, 296, 314` — `__shfl_sync` in grouped-GEMM scheduling
- `include/cutlass/gemm/warp/mma_mixed_input_tensor_op.h:174–177, 259–260` — `__shfl_up_sync`, `__shfl_down_sync` in mixed-input tensor-op (not reachable from SIMT)

The warp-level SIMT MMA itself (`mma_simt.h`, `mma_simt_policy.h`) does **not** use shuffles. So a vanilla SIMT GEMM is unaffected.

But: any kernel that uses **grouped problem visitors** (typically batched/grouped GEMM, used by some attention paths) will pull `grouped_problem_visitor.h` and hit `__shfl_sync`. If the K80 attention port reaches any grouped-GEMM code path, this needs guarding or replacing with the deprecated `__shfl` (which sm_37 supports) plus explicit warp synchronization.

**Action item for Story 0.2 (FlashAttention dependency map):** verify whether the FA kernels we want to port use grouped problem visitors. Same check applies for Story 2.x (XFormers cutlassF kernel — the *F* in `cutlassF` likely refers to "Forward attention," which may or may not use grouped GEMM).

### 6.5 No 96 KB+ shared memory requirement in the SIMT path

The Sm80 multistage path configures shared memory beyond 48 KB via `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`. The SIMT path uses only the default 48 KB available on every NVIDIA GPU since Fermi (sm_20). **Sm37 has 48 KB shared memory per block; safe.**

### 6.6 No `cooperative_groups` in the GEMM core

`cooperative_groups` (Sm60+ in basic form, Sm70+ in modern form) is not used by the SIMT GEMM dispatch. (It appears elsewhere in CUTLASS — collective kernels, certain reductions — but not on the path we care about.)

### Summary table

| Feature | Sm37 has it? | Used by Sm50 SIMT path? |
|---|---|---|
| Scalar FP32 / FP64 / INT FMA | ✅ | ✅ — the only thing the SIMT MMA uses |
| `__ldg` (sm_35+) | ✅ | (used in some loads, harmless) |
| 48 KB shared memory | ✅ | ✅ |
| `__shfl` (deprecated, pre-sync) | ✅ | not in core SIMT MMA |
| `__shfl_sync` (sm_70+) | ❌ | ❌ in core SIMT MMA; ⚠️ yes in grouped_problem_visitor.h |
| `cp.async` (sm_80+) | ❌ | ❌ |
| Tensor cores (`mma.*` PTX) | ❌ | ❌ |
| TMA (sm_90+) | ❌ | ❌ |
| 96 KB+ dynamic shared memory | ❌ | ❌ |
| Cooperative groups (modern) | ❌ | ❌ in core SIMT MMA |

The intersection of *required-by-Sm50-SIMT* and *missing-on-sm_37* is **empty**, with one footnote: any code path that drags in `grouped_problem_visitor.h` needs handling.

## 7. What adding `Sm37` would touch

Minimum-viable port (SIMT-only, no tensor-op aspirations):

| File | Change | Confidence |
|---|---|---|
| `include/cutlass/arch/arch.h` | Add `struct Sm37 { static int const kMinComputeCapability = 37; };` before `Sm50` | High — purely additive |
| `include/cutlass/gemm/kernel/default_gemm.h:831` | No change — the `!is_same<Sm80>` gate already includes Sm37 | High |
| `include/cutlass/gemm/threadblock/default_mma.h:48–51` | No change — SIMT core covers all archs | High |
| `include/cutlass/gemm/warp/mma_simt.h:114` | No change — the hard-coded Sm50 there is a type label, not a hardware contract | Medium — needs runtime verification on K80 |
| `test/unit/conv/...` and `test/unit/gemm/...` | Optional — add `_sm37.cu` test variants for coverage | Low priority |
| `CMakeLists.txt` | No change — build accepts arbitrary `CMAKE_CUDA_ARCHITECTURES` value (see §8) | High |

**Caveat for any kernel that reaches `grouped_problem_visitor.h`:** that file uses `__shfl_sync`. If we want grouped-GEMM on Sm37, we need a workaround — wrap the calls in arch-tagged macros, or fall back to the pre-sync `__shfl` for Sm37. This is one additional file change beyond the minimum, conditional on whether the attention kernels we want need grouped GEMM (decided in Story 0.2).

**No changes needed in:**

- `gemm_universal_adapter.h` — the `kMinComputeCapability >= 90` gates correctly skip Sm37 to the legacy path.
- Any tensor-op MMA core (`default_mma_core_sm70.h`, `_sm80.h`, `_sm90.h`) — Sm37 doesn't reach them.
- `mma_sm{70,75,80,86,89,90,100,103,120}.h` — same.

## 8. Build system

`CMakeLists.txt:86–90` requires CUDA Toolkit ≥ 11.3 (recommends 11.8). The K80 fork is on **CUDA 11.4**, which satisfies this minimum.

`CMakeLists.txt:609–662` handles `CMAKE_CUDA_ARCHITECTURES` generically. **There is no allow-list of acceptable SM values.** Adding `37` to the list passes through to `nvcc -gencode arch=compute_37,code=sm_37` without complaint.

NVCC has supported sm_37 codegen since CUDA 6.0 and continues to support it through CUDA 11.8. It was deprecated (warning) starting CUDA 11.0 and removed in CUDA 12.0 — so on our 11.4 toolchain, **sm_37 is a fully supported NVCC target with no special flags needed**.

CUTLASS is largely header-only. The build artifact for our purposes is just the propagated arch flag — there is nothing to gate at the `cutlass/` library level.

## 9. Open questions / unknowns

These are not blockers for Story 0.6 (consolidation), but they are things this doc cannot answer alone and should feed into adjacent stories:

1. **Does the warp MMA's hard-coded `ArchTag = arch::Sm50` (mma_simt.h:114) cause incorrect kernel selection for an Sm37-tagged GEMM at runtime?** Static analysis says no — the hard-coding is a type label internal to the SIMT chain. But until we compile and run a CUTLASS Sm37 GEMM on real K80 hardware (Phase 1 Story [#28][s14]), this remains a hypothesis.

2. **Do the FlashAttention / XFormers attention kernels reach `grouped_problem_visitor.h`?** Story [#20][s02] (FA MMA dependency map) needs to answer this. If yes, we have an additional file to patch.

3. **Are there any `__shfl_sync` or other sm_70+ intrinsics in the cutlassF attention kernels that XFormers ships?** Story 0.2 again — that recon pass will examine XFormers' attention code, not just FA's.

4. **Are there CUTLASS examples of pre-Maxwell forks?** Story [#23][s05] (prior art) covers this. If somebody has already done Sm37 in a fork, even a half-finished one, we save weeks.

[s02]: https://github.com/dogkeeper886/vllm/issues/20
[s05]: https://github.com/dogkeeper886/vllm/issues/23
[s14]: https://github.com/dogkeeper886/vllm/issues/28

## 10. Implications for Phase 1

Subject to Story 0.6's GO call after the rest of Phase 0 lands, the Phase 1 stories now look like this:

- **#25 (Add `Sm37` arch trait):** ~one-line edit in `include/cutlass/arch/arch.h`. **High confidence, near-zero scope.**
- **#26 (GEMM trait specialization):** **No new specializations needed** for the SIMT path — Sm37 is picked up by the existing `!is_same<Sm80>` gate. Story can probably close as "no work required" once this is empirically verified.
- **#27 (Verify SIMT-only):** static analysis already strongly suggests yes (§5). Confirmation comes from a runtime check in #28.
- **#28 (Compile minimal CUTLASS GEMM for sm_37 on K80):** the actual interesting story — this is where we discover whether the static analysis above survives contact with real hardware.
- **#29 (Numerical correctness vs cuBLAS):** standard.
- **#30 (Pin CUTLASS fork):** the Sm37 patch is so small we may not even need a fork — could potentially upstream a PR to NVIDIA/cutlass. Out of scope for this epic but worth noting.

The TBD effort estimates in epic #12 should drop significantly for Phase 1 based on this finding. **My current best estimate for Phase 1, pending #28's runtime verification: ~1 week of work, possibly less.** Bigger numbers were assumption-driven.

## 11. The bigger picture

Three lines from this report matter most:

- The arch tags are pure type structs with one integer member. **Adding Sm37 is a one-line change.**
- The default GEMM dispatch routes any non-Sm80 arch through SIMT — *including* a hypothetical Sm37 — with **no code edits to the dispatcher.**
- The SIMT path uses scalar FP32 FMA and nothing else from the per-arch instruction set. **Kepler can run scalar FP32 FMA.**

The wall between K80 and CUTLASS is much thinner than it looks from outside. The vLLM gate at `cuda.py:366` ("FlashAttention and xformers require sm_70+/sm_80+") is doubly wrong: XFormers' minimum is actually sm_50, and the underlying CUTLASS that XFormers depends on is structurally one struct away from supporting sm_37.

That doesn't mean the rest of the epic is easy — Phases 2 and 3 still need to deal with XFormers' and FA's own SM gates and any sm_70+ intrinsics they use directly (not through CUTLASS). But the long-pole assumption — "you'd be porting CUTLASS itself" — is much smaller than my [first take in PR review][assumption-correction] suggested.

[assumption-correction]: https://github.com/dogkeeper886/vllm/issues/12

---

**End of Story 0.1 deliverable.** Story 0.2 (FlashAttention MMA dependency map, [#20][s02]) is the recommended next pickup — its findings determine whether the optimistic picture above survives the FA-specific scrutiny.
