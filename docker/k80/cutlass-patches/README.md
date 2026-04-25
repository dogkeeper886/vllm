# CUTLASS patches for sm_37 (Tesla K80, Kepler)

These are git-format patches applied to a CUTLASS source tree at build time, enabling Tesla K80 (sm_37 / Kepler) targets that upstream CUTLASS does not natively support.

The patches are small and additive — none of them remove or modify upstream functionality. They add tags, traits, or specializations that the upstream library would silently route around for sm < 50 hardware.

## Why patches and not a fork

The full set of changes needed to support sm_37 is small enough that maintaining a fork would cost more than applying patches at build time. Phase 1 of the [K80 attention port epic][epic] decided this approach matches the precedent set by [`fcheung2888/llama.cpp-kepler`][llama-cpp-kepler] (sm_30 patches for llama.cpp) and keeps the diff visible in this repository.

If patch volume grows to where rebasing is painful, Story #30 will revisit by vendoring a CUTLASS fork.

[epic]: https://github.com/dogkeeper886/vllm/issues/12
[llama-cpp-kepler]: https://github.com/fcheung2888/llama.cpp-kepler

## Target CUTLASS versions

- **v4.0.0** — the version pinned by `CMakeLists.txt:302` via `FetchContent`. Primary target.
- **HEAD** (currently `7a9fe055cb` per Story 0.1's reconnaissance pin) — checked for forward-compatibility so future CUTLASS bumps don't silently break us.

Each patch must apply cleanly to both. Tested via `git apply --check` against both refs.

## Patches

### `sm37-trait.patch` (Phase 1 Story #25)

Adds `struct Sm37 { static int const kMinComputeCapability = 37; };` to `include/cutlass/arch/arch.h`, immediately before `struct Sm50`.

**Why:** Story 0.1 mapped CUTLASS's arch dispatch and found:

- All `cutlass::arch::Sm*` tags are pure type structs with one integer member (`kMinComputeCapability`). No interface to satisfy.
- The default GEMM dispatch routes any non-`Sm80` arch through SIMT via `!is_same<ArchTag, Sm80>` at `include/cutlass/gemm/kernel/default_gemm.h:831` — so a new `Sm37` tag is picked up automatically with no dispatcher changes.
- The SIMT path uses scalar FP32 FMA only, which sm_37 supports natively (Story 0.4).

**What this single struct unlocks:**

- Compile-time dispatch of `cutlass::gemm::device::Gemm<..., arch::Sm37, ...>` to the existing SIMT GEMM path.
- Per-arch trait specializations downstream stories can add (`DefaultGemmType<Sm37, ...>`, `DefaultGemmConfiguration<Sm37, ...>`).
- A static-analysis foothold for verifying that no tensor-core-only kernel can be reached when `ArchTag = Sm37` (Phase 1 Story #27).

**What this patch does NOT do:**

- Does not by itself produce a working sm_37 kernel binary. That is Phase 1 Story #28 (compile minimal CUTLASS GEMM for sm_37 on K80).
- Does not patch any `mma_simt.h` warp-MMA hard-coding. Story 0.1 §4.4 confirmed the hard-coded `Sm50` label there is internal to the SIMT chain and does not constrain the exterior arch tag — verifiable when Story #28 runs.
- Does not register `Sm37` in CUTLASS's CMake architecture list, test scaffolding, or Python autogen scripts. Test files and CMake handling are tracked separately (and are not blocking until Story #28).

**Applies to:** CUTLASS v4.0.0 and HEAD (`7a9fe055cb`).

## Applying the patches

**Status:** the patches are not yet wired into the build pipeline. Story #28 will integrate them via either:

1. A `prepare_cutlass.sh` script run inside the K80 builder image, between `FetchContent_Declare(cutlass)` and the actual build, or
2. A CMake `execute_process(COMMAND patch ...)` call gated on `VLLM_K80_FORK=ON`.

For now, the patches live in this directory as inspectable source. Manually apply with:

```bash
cd /path/to/cutlass-source
git apply /path/to/vllm/docker/k80/cutlass-patches/sm37-trait.patch
```

A dry-run check:

```bash
git apply --check docker/k80/cutlass-patches/sm37-trait.patch
```

### Idempotency

**Patches in this directory are not idempotent.** A second `git apply` of the same patch against an already-patched tree fails with `error: patch does not apply`. This is expected git-apply behavior, but it matters for repeated builds (Docker image rebuilds, dev iteration, CI re-runs) where the build script may run against either a freshly-fetched CUTLASS or a previously-patched copy.

Recommended patterns for build-pipeline integration (Story #28 will pick one):

```bash
# Option A — explicit pre-check via grep marker
grep -q "struct Sm37" include/cutlass/arch/arch.h \
  || git apply /path/to/sm37-trait.patch

# Option B — 3-way merge handles already-applied (also more robust against
# upstream additions near the patch site)
git apply -3 /path/to/sm37-trait.patch

# Option C — GNU patch's --forward flag skips already-applied hunks
patch -p1 --forward < /path/to/sm37-trait.patch
```

Option B (`git apply -3`) is the most durable when CUTLASS-side context drifts; future maintainers should prefer it unless there's a specific reason not to.

## Conventions for new patches

When future stories add patches:

- Name patches descriptively: `<feature>-<scope>.patch` (e.g. `sm37-trait.patch`, `sm37-default-mma-core.patch`).
- Each patch must apply cleanly to **both** the pinned version (v4.0.0 today) and CUTLASS HEAD. Test via `git apply --check`.
- Add a section to this README describing what the patch does, why, and what it does *not* do.
- Patches should be additive whenever possible. Removing or modifying upstream code is allowed but should be explicitly justified.

## See also

- [`docs/port/cutlass-arch-system.md`](../../../docs/port/cutlass-arch-system.md) — Story 0.1's CUTLASS dispatch analysis (the why behind these patches).
- [`docs/port/kepler-vs-maxwell.md`](../../../docs/port/kepler-vs-maxwell.md) — Story 0.4's hardware feature gap analysis.
- [`docs/port/cuda-11.4-version-pins.md`](../../../docs/port/cuda-11.4-version-pins.md) — Story 0.3's version pin survey.
