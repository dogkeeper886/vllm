# FlashAttention MMA / SM dependency map — sm_37 port reconnaissance

**Story:** [#20][issue-20] (Phase 0 of epic [#12][epic])
**flash-attention commit pinned for citations:** `ac6f2eb5413748c68192aa384a40a38d60ad6abd` (Dao-AILab/flash-attention `main`, 2026-04-23).
**Methodology:** every claim cites flash-attention source by `file:line`. No claim without a citation.

[issue-20]: https://github.com/dogkeeper886/vllm/issues/20
[epic]: https://github.com/dogkeeper886/vllm/issues/12

---

## 1. Scope

This doc answers Story 0.2's questions for FlashAttention:

1. Repo layout — where the kernels live.
2. Where are the SM gates?
3. What MMA / tensor-core intrinsics does FA use?
4. Is the FA algorithm separable from its tensor-core implementation?
5. Does FA reach CUTLASS's `grouped_problem_visitor.h` (the risk surfaced by Story [#19][s01])?
6. Is there any SIMT fallback?
7. Build / `setup.py` SM list.
8. CUDA 11.4 compatibility risk.
9. **Decision: Phase 3 port path (3.2a) or rewrite path (3.2b)?**

[s01]: https://github.com/dogkeeper886/vllm/issues/19

## 2. Repo layout

| Directory | Contents |
|---|---|
| `csrc/flash_attn/` | FA-2 implementation. All `.cu` files end with `_sm80` suffix (e.g., `flash_fwd_hdim32_fp16_sm80.cu`). `setup.py:306-380` enumerates them. |
| `hopper/` | FA-3 (Hopper-optimized). Has dual-arch headers: `flash_fwd_kernel_sm80.h` and `flash_fwd_kernel_sm90.h`. |
| `flash_attn/` | Pure-Python wrapper that dispatches to the compiled CUDA module via `flash_attn_2_cuda`. No reference attention here. |
| `csrc/cutlass/` | Vendored CUTLASS submodule. |
| `csrc/composable_kernel/` | AMD ROCm path (out of scope for our K80 work). |

Naming convention is unambiguous: the FA-2 kernels are explicitly `_sm80` only. There are no `_sm70`, `_sm75`, `_sm60`, or `_sm37` variants in the build.

## 3. SM gates inventory

### 3.1 Preprocessor gates

The defining gate. `csrc/flash_attn/src/flash_fwd_launch_template.h:17`:

```cpp
// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif
```

The non-`ARCH_SUPPORTS_FLASH` branch falls through to `FLASH_UNSUPPORTED_ARCH` at line 26:

```cpp
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");
```

So **anything below sm_80 hits a runtime fatal printf** in the FA-2 forward path. There is no compile-time `static_assert` that prevents the build, but the runtime message is unambiguous: FA-2 expects sm_80–sm_90.

A second gate at `csrc/flash_attn/src/kernel_traits.h:18`:

```cpp
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using Element = elem_type;
    static constexpr bool Has_cp_async = true;
#else
    using Element = cutlass::half_t;
    static constexpr bool Has_cp_async = false;
#endif
```

This gate selects whether `cp.async` (sm_80+ asynchronous copies) is used. The `else` path in current FA exists primarily for the SM75 fallback documented below.

### 3.2 Hopper static_assert gates

`hopper/flash_fwd_kernel_sm80.h:55`:

```cpp
static_assert(ArchTag::kMinComputeCapability >= 80);
```

`hopper/flash_fwd_kernel_sm90.h:68`:

```cpp
static_assert(ArchTag::kMinComputeCapability >= 90);
```

These are compile-time hard gates in FA-3. A build with `ArchTag::kMinComputeCapability = 37` would fail to compile.

### 3.3 Build-system gates (setup.py)

`setup.py:73`:

```python
def cuda_archs() -> str:
    return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;110;120").split(";")
```

The default arch list is `80;90;100;110;120`. **No arch below 80 is in the default build.** This is overridable by env var — adding `37` would cause `nvcc` to attempt sm_37 codegen, but the preprocessor gate at §3.1 means any sm_37-built kernel still falls through to the FATAL printf.

`setup.py:259`:

```python
if bare_metal_version < Version("11.7"):
    raise RuntimeError(
        "FlashAttention is only supported on CUDA 11.7 and above.  "
        "Note: make sure nvcc has a supported version by running nvcc -V."
    )
```

This is a **hard `RuntimeError` at pip-install time**, not a warning. Our K80 fork is pinned to CUDA 11.4, which is below this minimum. **Current FA cannot even build on our toolchain.**

## 4. MMA / tensor-core intrinsic inventory

### 4.1 The MMA atom selection

`csrc/flash_attn/src/kernel_traits.h:32–36`:

```cpp
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;
#else
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
#endif
```

Two paths exist:
- **sm_80+:** SM80 16×8×16 F32-acc tensor-core MMA, FP16 or BF16 input.
- **else (would include sm_70, sm_75, hypothetical sm_37):** SM75 16×8×8 F32-acc tensor-core MMA, FP16 only.

**Both atoms target tensor cores.** SM75 (Turing) introduced tensor cores; SM37 (Kepler) has none. The "else" branch is **not a SIMT FP32 fallback** — it is a Turing-specific tensor-core fallback. Compiling it for sm_37 would fail at link time (the SM75 MMA atom expands to PTX `mma.*` instructions that have no sm_37 form) or, worse, succeed and produce a kernel that throws a runtime "no kernel image available" error on K80.

### 4.2 No hand-written MMA PTX in FA-2

A grep across `csrc/flash_attn/` for inline `mma.*`/`wmma.*` PTX produces zero hits in FA-2 sources. **All MMA goes through CUTLASS's `MMA_Atom<...>` template abstraction.** This is good news for portability *if* a SIMT atom existed for the required shapes — see §6.

### 4.3 cp.async (sm_80+) usage

`csrc/flash_attn/src/kernel_traits.h:126–127`:

```cpp
using GmemCopyAtom = std::conditional_t<
    Has_cp_async, SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, AutoVectorizingCopy
>;
```

When `Has_cp_async` is false (the non-sm_80+ path), the global-memory copy atom falls back to `AutoVectorizingCopy`. This part of the codebase **does** have a non-sm_80 path that doesn't require new instructions.

`csrc/flash_attn/src/utils.h:289–290` uses inline PTX `cp.async.wait_group` gated on `CUTE_ARCH_CP_ASYNC_SM80_ENABLED` — only emitted on sm_80+ builds.

### 4.4 GMMA / TMA (Hopper-only)

FA-3 uses Hopper-specific instructions:
- `hopper/utils.h:270` references `GMMA::DescriptorIterator`.
- `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` (referenced from `hopper/flash_fwd_launch_template.h:22`) uses Tensor Memory Accelerator (TMA, sm_90 only).

These paths are unreachable from sm < 90. They do not constrain the K80 port, but they confirm FA-3 is permanently out of scope for this hardware.

## 5. Algorithm vs implementation separability

The FlashAttention algorithm — Q/K/V tiling into shared memory, online softmax with running statistics, register-resident output accumulation — is conceptually portable to any architecture with shared memory and FP32 FMA. The question is whether FA's *source* allows that algorithm to be reused without the tensor-core implementation.

### 5.1 Algorithm code is in `flash_fwd_kernel.h`

`csrc/flash_attn/src/flash_fwd_kernel.h:51` — `compute_attn_1rowblock()` contains:

- Shared-memory tile allocation (lines 160–167)
- Online softmax instance (line 285)
- Q/K/V global → shared loads (lines 138–158)
- Iterative masking + partial matmul accumulation loop (lines 302–375)

The algorithm itself is C++ template code with no hand-written PTX.

### 5.2 The matmul call is abstracted

`csrc/flash_attn/src/flash_fwd_kernel.h:319–322` invokes a `FLASH_NAMESPACE::gemm<...>()` helper, defined in `utils.h`. The helper takes a `tiled_mma` parameter — a CUTLASS `TiledMMA<MMA_Atom<...>, ...>` constructed from the atom selected in `kernel_traits.h:32–36` (§4.1).

So the chain is:

```
compute_attn_1rowblock        (algorithm — portable C++)
    └─ FLASH_NAMESPACE::gemm  (CUTLASS abstraction layer — portable)
        └─ tiled_mma          (CUTLASS TiledMMA — needs an Atom)
            └─ MMA_Atom<SM80_...>  ← arch-specific PTX expansion
```

In principle, swapping the Atom for a SIMT FP32 implementation would let the algorithm run unchanged. **In practice:**

1. The CUTLASS SIMT atoms surveyed in Story [#19][s01] (`mma_sm50.h:61–143`, the warp-level `MmaSimt`) are scalar 1×1×1 FMA — much smaller than the 16×8×16 or 16×8×8 shapes FA expects. Whether larger SIMT atoms exist in unexamined CuTe paths in CUTLASS is not yet established. Re-tiling the FA inner loop to use scalar SIMT is non-trivial — the register layout, shared-memory swizzle (`SmemLayoutAtomQ` with its `Swizzle<kSwizzle, 3, 3>{}` composition at `kernel_traits.h:79–86`, plus `SmemLayoutQ` and `SmemLayoutKV` at `kernel_traits.h:84` and `:88`), and bank-conflict avoidance are all designed for tensor-core access patterns.
2. The fallback Atom in §4.1 (SM75) does not buy us anything: it is still a tensor-core atom, just on Turing instead of Ampere.

So while the algorithm is structurally separable, **there is no drop-in replacement Atom that runs on sm_37 in the CUTLASS surface area surveyed so far**. Any port would require either (a) writing a SIMT Atom for CUTLASS that matches FA's expected shapes, or (b) rewriting the kernel against a different abstraction.

## 6. Does FA reach CUTLASS's `grouped_problem_visitor.h`?

**No.** Grep across `csrc/`, `hopper/`, and `flash_attn/` for `grouped_problem_visitor`, `GroupedProblemVisitor`, and `GroupedGemm` returns zero matches. FA computes one block per `(batch, head)` pair via `dim3 grid(num_m_block, params.b, params.h)` (typical pattern; see kernel launches in `csrc/flash_attn/src/flash_fwd_launch_template.h`). It uses per-block independent matmul, not grouped GEMM dispatch.

**This is good news.** The `__shfl_sync` risk surfaced in Story 0.1 (CUTLASS's `grouped_problem_visitor.h` lines 241/248/289/296/314) does not apply to FA. The CUTLASS code path FA actually traverses is the SIMT/tensor-op MMA dispatch we mapped in Story 0.1 — not the grouped-visitor path.

## 7. SIMT fallback — does FA have one?

**No SIMT FP32 fallback for CUDA hardware exists.** Several places to look, all empty for our purposes:

1. **No SIMT Atom in the kernel template.** The fallback in `kernel_traits.h:36` is the SM75 Turing Atom, not a SIMT FP32 path.
2. **No CPU / pure-Python reference attention** in `flash_attn/`. There is no `if not cuda_available: use_pytorch()` branch.
3. **No `simt` namespace, no `if __CUDA_ARCH__ < 700: ...`** branches in the kernel C++ code.
4. **No reference-quality PyTorch implementation** elsewhere in the tree. Tests under `tests/` use FA's own kernels as ground truth.

There is, however, a non-CUDA backend dispatch worth noting. `flash_attn/flash_attn_interface.py:21–24`:

```python
if USE_TRITON_ROCM:
    from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2 as flash_attn_gpu
else:
    import flash_attn_2_cuda as flash_attn_gpu
```

The `USE_TRITON_ROCM` path uses a Triton kernel from `aiter` for AMD ROCm hardware. **This is irrelevant for K80 (we are CUDA, not ROCm) and is not a SIMT path** — Triton compiles to PTX/HIP, not SIMT loops. But its existence is a structural hint for Story [#38][s32b]: a Triton-CUDA implementation could plausibly piggyback on the algorithm logic already factored out for the AMD path. Whether Triton can target sm_37 is an open question that Story 3.2b would need to answer before committing to that strategy.

The Turing-only [flash-attention-turing](https://github.com/ssiu/flash-attention-turing) fork referenced in the README is a separate project — confirms upstream's stance that pre-Ampere variants are out-of-tree work, not a maintained codepath.

## 8. Build / setup.py findings

| What | Where | Value |
|---|---|---|
| Default `TORCH_CUDA_ARCH_LIST` | `setup.py:73` | `"80;90;100;110;120"` |
| Hard CUDA minimum | `setup.py:259` | `RuntimeError` if CUDA < 11.7 |
| FA-2 kernel sources | `setup.py:306–380` | All filenames end `_sm80.cu`; no other arch |
| FA-3 (hopper) | `hopper/setup.py` | sm_80 + sm_90, requires CUDA 12.3+ for full feature set |

README claim ([root README.md] line ~30s): *"FlashAttention-2 with CUDA currently supports: 1. Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100)."*

Compare to the corresponding XFormers picture (Story 0.1 §11): XFormers' minimum is sm_50 with an explicit fp32 path. **FA's stance is much harder than XFormers' on this point.**

## 9. CUDA 11.4 compatibility risk

This is the single biggest finding from this story.

**Current FA cannot build on our pinned CUDA 11.4 toolchain.** The hard minimum at `setup.py:259` is 11.7. This is a `RuntimeError`, not a deprecation warning.

The K80 fork pinned to CUDA 11.4 because newer CUDA toolkits may have dropped Kepler support. Story [#21][s03] (CUDA 11.4 compatibility survey) is responsible for finding:

- Whether older FA versions had a lower CUDA minimum (likely yes — the 11.7 requirement was added at some commit; before that it may have been 11.0+).
- Whether any older FA version still has the SM gates we need to patch around.
- Whether bumping our toolkit to CUDA 11.7 is feasible without losing K80 driver compatibility.

Even if Story 0.3 finds an older FA version that supports CUDA 11.4, the SM-gate / tensor-core wall in §3 and §4 still applies — older FA still uses CUTLASS tensor-core MMA atoms. So this is "necessary but not sufficient": getting an older FA to build is a precondition for any port path; it does not by itself solve the kernel problem.

[s03]: https://github.com/dogkeeper886/vllm/issues/21

## 10. Recommendation for Phase 3 — port vs rewrite

**Recommendation: rewrite path (Story [#38][s32b], 3.2b). Confidence: high.**

[s32b]: https://github.com/dogkeeper886/vllm/issues/38

The reasoning, anchored in source:

1. **No code path exists for sm_37.** The preprocessor gate at `kernel_traits.h:18` requires `__CUDA_ARCH__ >= 800` for the SM80 path; the `else` branch at line 36 selects an SM75 tensor-core atom, which sm_37 cannot run. There is no third branch.

2. **MMA atoms are tensor-core templates.** Both atoms in `kernel_traits.h:32–36` (`SM80_16x8x16_F32F16F16F32_TN` and `SM75_16x8x8_F32F16F16F32_TN`) expand to PTX `mma.*` instructions that require tensor cores. K80 has none. CUTLASS's existing SIMT atom (`mma_sm50.h:61–143`, found in Story 0.1) is scalar 1×1×1 FMA — wrong shape for FA's tile structure.

3. **Build minimum CUDA 11.7 vs our 11.4.** Even before the kernel question, current FA fails to install. Older FA versions might be available (Story 0.3) but that's a separate question.

4. **Algorithm is conceptually portable, but no abstraction is left after you remove the tensor-core layer.** §5 shows the algorithm code (`flash_fwd_kernel.h:51`) is C++ template code, structurally separable. But it depends on a CUTLASS `TiledMMA` whose Atom must match a tensor-core shape. Rewriting the inner loop to use scalar SIMT FP32 means re-doing the register layout, the shared-memory swizzle (`kernel_traits.h:117–146`), and the bank-conflict avoidance. At that point you are not patching FA; you are reimplementing the algorithm against a different kernel skeleton.

5. **No SIMT fallback to fork from.** There is no FP32 / pure-PyTorch reference attention in this repo (§7). Any rewrite starts from the algorithm description (online softmax + tiling) and a sm_37-friendly kernel scaffold.

### What "rewrite" practically means

This is **Story 3.2b** (#38). Three candidate implementation strategies, in rough order of pragmatism:

- **Triton kernel.** Triton has historically supported older arches via PTX backends; if it can target sm_37 (open question — verify before committing), implementing FA's algorithm in Triton is order-of-magnitude less work than hand-written CUDA. Triton handles tile layout, register allocation, and scheduling automatically. The existence of an AMD Triton path in the FA tree (§7) suggests the algorithm has already been factored for non-CUDA-PTX backends.
- **Hand-written CUDA, FP32 SIMT.** Write a `flash_fwd_sm37.cu` that follows FA's algorithm but uses scalar `__fma_rn` and standard `__syncthreads()`. Tile sizes would be smaller because K80 has 48 KB shared memory (vs 96+ KB on Ampere), so per-block batch is reduced. The expected speedup on K80 is modest — K80 is memory-bound, not compute-bound, so the O(N) memory benefit (long-context unlock, story #54) is the real prize, not throughput.
- **Reuse FA's softmax helpers.** `csrc/flash_attn/src/softmax.h` (online softmax statistics) is mostly arch-agnostic and could be reused regardless of which strategy above is picked. Worth pulling into a rewrite even if everything else is from scratch.

**Effort estimate intentionally omitted.** Story 0.6 owns the consolidated effort estimate across all of Phase 0. This story alone is not enough evidence to commit to a number — it depends on which strategy is chosen, on Story 0.3's findings about CUDA toolkit pin options, on Story 0.5's prior art (does the Turing fork or another project already give us 80% of the work?), and on the user's resolution of the issue #2 conflict.

### What this means for issue #2 study plan

Story 3.2b explicitly conflicts with issue #2's "Explicitly skipping → Writing CUDA kernels from scratch — deep time sink, low ROI" line. **Story 0.6 (consolidation) needs to surface this conflict and ask the user whether the K80 attention-port goal justifies revisiting #2.** I am not making that call here.

## 11. Stability / forward risks

1. **CUTLASS upgrade churn.** FA uses CUTLASS atoms via templates. If we vendor an older FA version (Story 0.3), and that older FA pins an older CUTLASS, our Story 0.1 findings about the current CUTLASS arch system may not transfer cleanly. Each version pin needs verification.

2. **CUDA 11.4 deprecation pressure.** Same as Story 0.1 §9 risk 3 — CUTLASS already deprecated CUDA 11.3 and warns at 11.4. FA went further, hard-erroring at 11.6. The trajectory across the ecosystem points to 11.4 falling out of supported configurations entirely. Long-term, the K80 fork's value depends on freezing tooling at a known-good combination and accepting it as a museum piece. This affects the long-term maintainability of any rewrite landed.

3. **The Turing fork.** [flash-attention-turing](https://github.com/ssiu/flash-attention-turing) is referenced in FA's README as the pre-Ampere path. **Story [#23][s05] (prior art search) needs to examine it.** If the Turing fork has already done the work of cleaning up the SM80-only assumptions in the build and providing a fp16-on-Turing path, parts of that work might transplant — although Turing has tensor cores and K80 does not, so the kernel itself will not transfer.

[s05]: https://github.com/dogkeeper886/vllm/issues/23

## 12. Implications for Phase 3 stories

Subject to Story 0.6's GO call, this story's evidence supports the following Phase 3 routing — but Story 0.6 owns the final decisions:

- **#36 (Decide port vs rewrite):** Story 0.2 produces a citation-backed recommendation of "rewrite." Story 0.6 consolidates with the other Phase 0 outputs and either ratifies or revisits.
- **#37 (port path 3.2a):** Story 0.6 should evaluate closing as "not pursued" — there is no viable port path given FA's current source state. Not closing it unilaterally here.
- **#38 (rewrite path 3.2b):** The actual Phase 3 work if the recommendation is ratified. Conflicts with issue #2's "no CUDA kernels from scratch" line — Story 0.6 must surface this for explicit user resolution before any 3.2b work starts.
- **#39 (numerical correctness):** standard.
- **#40 (performance baseline):** standard. Note expectation: the win on K80 will be the O(N) memory benefit (long-context unlock, story #54), not raw throughput.

## 13. FlashAttention-only conclusions

This story examined FA in isolation. Conclusions are limited to FA:

- FA's kernels are tensor-core-only by construction. Both the sm_80 path and the sm_75 fallback compile to PTX `mma.*` instructions K80 cannot execute.
- FA's CUDA toolkit minimum (11.7) is incompatible with our pinned CUDA 11.4 even before the kernel question.
- The FA *algorithm* is conceptually separable from the implementation, but the *abstraction layer* between them (CUTLASS `TiledMMA`) does not have a usable SIMT Atom in scope.
- Phase 3 must be a rewrite, not a port. This conflicts with issue #2's "no CUDA kernels from scratch" rule, which Story 0.6 must surface.

The optimistic CUTLASS picture from Story 0.1 does not transfer to FlashAttention. **CUTLASS gives us scalar SIMT GEMM on sm_37; FA expects 16×8×16 tensor-core GEMM and there is no replacement Atom.**

XFormers (Story 2.x) remains the better-positioned port target — XFormers' fp32 path uses the SIMT GEMM that Story 0.1 found is portable to sm_37. Whether FA is worth the rewrite cost given XFormers may already deliver the long-context goal is a question for Story 0.6.

---

**End of Story 0.2 deliverable.** Story 0.3 (CUDA 11.4 compatibility survey, [#21][s03]) is the recommended next pickup — it answers whether an older FA version exists that builds on CUDA 11.4 (necessary but not sufficient for any port path), and the parallel question for XFormers and CUTLASS pin candidates.
