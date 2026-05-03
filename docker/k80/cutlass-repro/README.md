# CUTLASS sm_37 SGEMM reproducer (Story #28)

Minimal CUTLASS GEMM that dispatches through `cutlass::arch::Sm37` (the tag added by [PR #54][pr54], Story #25) and runs on a Tesla K80. Closes Phase 1 Story [#28][issue-28] of the K80 attention port epic [#12][epic].

[pr54]: https://github.com/dogkeeper886/vllm37/pull/54
[issue-28]: https://github.com/dogkeeper886/vllm37/issues/28
[epic]: https://github.com/dogkeeper886/vllm37/issues/12

## What this proves

1. **NVCC 11.4 accepts `arch::Sm37`** as a CUTLASS template parameter when the source tree has `sm37-trait.patch` applied.
2. **The compiled binary actually runs** on Tesla K80 hardware (driver R470 / sm_37 / Kepler GK210). Build success ≠ runtime success on this hardware ([Story 0.5 §3.7][story05] flagged this as the most actionable finding from prior art).
3. **Numerical output matches expected** — A=ones (256×256), B=ones (256×256) ⇒ each element of D should equal K=256.

[story05]: https://github.com/dogkeeper886/vllm37/blob/main/docs/port/prior-art.md

## Files

| File | Purpose |
|---|---|
| `sgemm_sm37.cu` | Story #28 — minimal reproducer using `cutlass::gemm::device::Gemm<..., arch::Sm37>` to compute a 256×256×256 SGEMM with deterministic ones-and-ones input, verifies output element-wise against expected `K`. |
| `sgemm_sm37_vs_cublas.cu` | Story #29 — same CUTLASS path vs cuBLAS reference. Five sizes (64²–1024²), randomized inputs, reports max abs / max rel / mean abs error per size. Tolerance: max relative error ≤ 1e-3 across all sizes. |
| `build.sh` | Pulls CUTLASS at `v4.0.0` (matches our `FetchContent` pin), applies `docker/k80/cutlass-patches/*.patch`, then compiles both binaries with NVCC targeting `sm_37`. Runs inside the K80 builder image. The cuBLAS comparison binary additionally links `-lcublas`. |
| `README.md` | This file. |

## Running it

### In CI (recommended)

The `k80-cutlass-repro` workflow does the full pipeline: builds the binary on the K80 self-hosted runner, executes it, and uploads the output as an artifact.

```bash
gh workflow run k80-cutlass-repro.yml --repo dogkeeper886/vllm37
```

The workflow runs on a single K80 die, no NCCL, no multi-die — equivalent risk to a TP=1 vLLM smoke test, so it's not gated by the weekend rule for hardware-risky operations.

### Manually inside the K80 builder image

If you want to iterate locally on the runner:

```bash
docker run --rm --runtime=nvidia --gpus all \
    -v "$(pwd)/docker/k80/cutlass-repro:/repro" \
    -v "$(pwd)/docker/k80/cutlass-patches:/patches" \
    vllm37-builder:latest \
    bash -c "cd /repro && PATCH_DIR=/patches ./build.sh && /tmp/cutlass-sm37-build/build/sgemm_sm37"
```

### What success looks like

**Story #28 — `sgemm_sm37`:**

```
=== Story #28 — CUTLASS sm_37 SGEMM reproducer ===
device       : Tesla K80 (cc 3.7)
problem      : 256 x 256 x 256 (M x N x K), fp32, SIMT
expected     : 256 per element
got D[0]     : 256
max abs err  : 0.000000e+00
tolerance    : 1.000000e-03
errors       : 0 / 65536
RESULT       : PASS
```

Verified bit-exact on K80 hardware via [run 24946197232][run-28].

**Story #29 — `sgemm_sm37_vs_cublas`:**

```
=== Story #29 — CUTLASS Sm37 vs cuBLAS SGEMM ===
device       : Tesla K80 (cc 3.7)

--- M=64 N=64 K=64 ---
  max abs error : <small>
  max rel error : <small>  (tolerance 1.000e-03)
  ...
--- M=1024 N=1024 K=1024 ---
  ...
=== Summary ===
passed: 5/5
RESULT       : PASS
```

Exit code 0 = pass. Non-zero = either CUDA runtime failure (exit 2), CUTLASS dispatch failure (exit 3), or numerical disagreement / tolerance violation (exit 1).

[run-28]: https://github.com/dogkeeper886/vllm37/actions/runs/24946197232

## What this does NOT do

- **Does not benchmark.** Numerical correctness only. Story 5.x covers performance.
- **Does not patch the vLLM build.** Current vLLM K80 build still uses `VLLM_BUILD_LEGACY_CUDA=ON` (`CMakeLists.txt:273`) to skip CUTLASS. This reproducer is standalone — it proves the kernel path works in isolation. Story [#30][story30] handles the build-graph integration.
- **Does not exercise the FA-style attention algorithm.** Just GEMM. Story 3.x will do attention-shaped work after Phase 1 closes.

[story30]: https://github.com/dogkeeper886/vllm37/issues/30

## Hardware safety notes

- TP=1, single-die, no NCCL — falls under the "always fine" category in the saved hardware-risky-ops timing rule.
- Compute-only workload: no power-cap experiments, no driver pokes, no `nvidia-smi -pl`.
- Total VRAM footprint: ~1 MB (3 × 256² × 4 bytes). Negligible on the K80's 11.45 GiB-per-die budget.

## Related

- [`docs/port/cutlass-arch-system.md`](../../../docs/port/cutlass-arch-system.md) — Story 0.1's CUTLASS dispatch analysis (why this works at all).
- [`docs/port/kepler-vs-maxwell.md`](../../../docs/port/kepler-vs-maxwell.md) — Story 0.4's hardware feature gap (why sm_37 has every primitive Sm50 SIMT needs).
- [`docker/k80/cutlass-patches/`](../cutlass-patches/) — the patches `build.sh` applies.
