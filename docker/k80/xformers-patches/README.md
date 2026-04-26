# XFormers patches for sm_37 (Tesla K80, Kepler)

Git-format patches applied to an XFormers source tree at build time, enabling Tesla K80 (sm_37 / Kepler) targets that upstream XFormers does not natively support.

The patches are small and additive — none of them remove or modify upstream functionality. They register sm_37 in the kernel generator and, eventually (Story #32), lower the Python-level minimum compute capability so the dispatcher actually selects the cutlass FMHA op on K80.

## Critical version note — XFormers must be pinned to v0.0.28.post3

After XFormers v0.0.29 (commit `3d947a6`, 2024-12-16, "Remove the cutlass backend of mem-eff, as it is now embedded in PyTorch"), the C++/CUDA mem-eff attention kernels were **removed from XFormers entirely** — they now live in PyTorch upstream. Our K80 fork uses **PyTorch 2.0.1** (per `CLAUDE.md`), which predates the embedded version.

That means **XFormers HEAD has nothing for our K80 build to consume** — the `xformers/csrc/attention/cuda/fmha/` directory doesn't exist there. The patches in this directory are designed to apply to the last XFormers tag that still ships the cutlass mem-eff kernels in-tree:

- **Pin: `v0.0.28.post3`** (2024-10-30)
- The Python wrapper `xformers/ops/fmha/cutlass.py` exists in both old and new XFormers, but only the old version has the C++/CUDA kernels behind it.

If our K80 fork ever upgrades to PyTorch ≥ 2.4 (which would mean abandoning the K80-supporting CUDA 11.4 toolchain — see `docs/port/cuda-11.4-version-pins.md`), this pin can revisit; until then v0.0.28.post3 is the constraint.

## Why patches and not a fork

Same reasoning as `cutlass-patches/`: minimal additive changes are smaller than maintaining a fork, and the patches stay visible in this repository. If patch volume grows, Story #34 (run XFormers' own test suite on K80) will be the natural point to revisit.

## Patches

### `sm37-generate-kernels.patch` (Story #31)

Adds `37` to the SM list at `xformers/csrc/attention/cuda/fmha/generate_kernels.py:23`:

```python
# Before:
SM = [50, 70, 75, 80, 100]  # Sm80 kernels support up to Sm100

# After:
SM = [37, 50, 70, 75, 80, 100]  # Sm80 kernels support up to Sm100; Sm37 added for Tesla K80 (K80 attention port; v0.0.28.post3)
```

**Why:** Per Story 0.2 §3.3, XFormers' kernel autogen uses this list to materialize per-SM kernel instantiations. Without `37` in the list, `generate_kernels.py` produces zero matching FMHA kernel instantiations for sm_37 — the build silently emits no kernel, and runtime falls through with "no kernel image is available."

**What this unlocks:** When Story #33 builds XFormers from source against our patched CUTLASS (which has `cutlass::arch::Sm37` from `cutlass-patches/sm37-trait.patch`, Phase 1 verified), the autogen will emit a sm_37 instantiation per the SM-list dispatch. The expectation, per Story 0.1 / 0.4, is that those kernels dispatch through CUTLASS's SIMT path and run correctly on K80 — verified empirically by Phase 1's reproducer for standalone GEMM. **Story #33 is the equivalent verification gate for the XFormers-side integration**; this PR alone doesn't prove the XFormers attention kernels work end-to-end on K80.

**What this does NOT do:**

- Does not by itself produce a working sm_37 XFormers binary — that is Story #33.
- Does not patch the Python compute-capability gate at `common.py:391` — that's Story #32, separate one-line patch.
- Does not change XFormers' bundled CUTLASS submodule. XFormers v0.0.28.post3 bundles CUTLASS at commit `7d49e6c7e2` (CUTLASS v3.5.0, April 2024). **Verified pre-merge: our `cutlass-patches/sm37-trait.patch` applies cleanly to that bundled CUTLASS as-is** — the `arch.h` structure at the patch site is stable across CUTLASS v3.5.0 → v4.0.0 → HEAD. Story #33's CUTLASS-side work is therefore: apply the existing patch to the submodule, no separate version is needed.

**Applies to:** XFormers v0.0.28.post3.

## Applying the patches

The patches are not yet wired into a build pipeline (Story #33's territory). Manually apply with:

```bash
cd /path/to/xformers-source-at-v0.0.28.post3
git apply /path/to/vllm/docker/k80/xformers-patches/sm37-generate-kernels.patch
```

A dry-run check:

```bash
git apply --check docker/k80/xformers-patches/sm37-generate-kernels.patch
```

### Idempotency

Same as `cutlass-patches/`: patches are not idempotent against `git apply`. Recommended apply patterns for build pipelines:

```bash
# Option A — explicit pre-check via grep marker
grep -q "37, 50, 70, 75, 80, 100" \
    xformers/csrc/attention/cuda/fmha/generate_kernels.py \
  || git apply /path/to/sm37-generate-kernels.patch

# Option B — 3-way merge handles already-applied + minor context drift
git apply -3 /path/to/sm37-generate-kernels.patch

# Option C — GNU patch's --forward flag
patch -p1 --forward < /path/to/sm37-generate-kernels.patch
```

Option B is most durable.

### Automated verification

```bash
make -C docker/k80 verify-xformers-patches
```

Clones XFormers shallow at the pinned tag, runs `git apply --check` per patch. Zero hardware required; matches the same idiom as `verify-cutlass-patches`.

## Pin maintenance

Three scenarios that require maintenance work here:

### Scenario A — XFormers refactor changes file paths

The cutlass mem-eff backend was already removed in v0.0.29; we've already pinned past that risk. But XFormers may further refactor `xformers/csrc/attention/cuda/fmha/` in a future release. Detection: `make verify-xformers-patches` will fail because the target file no longer exists at the new pin.

Recovery options:
1. Stay at v0.0.28.post3 indefinitely (current strategy).
2. Find the new location of the kernel autogen if the refactor moved rather than removed.
3. Bump to the latest pre-removal pin if there's a newer one before v0.0.29.

### Scenario B — upstream adds Sm37 natively

Detection (same idiom as cutlass-patches/README.md scenario B):

```bash
WORK=$(mktemp -d); trap "rm -rf $WORK" EXIT
git clone --depth 1 --branch v0.0.28.post3 https://github.com/facebookresearch/xformers.git $WORK/xformers
git -C $WORK/xformers apply --check        docker/k80/xformers-patches/sm37-generate-kernels.patch || \
    git -C $WORK/xformers apply --check --reverse docker/k80/xformers-patches/sm37-generate-kernels.patch \
    && echo "Sm37 already in XFormers SM list — drop the patch"
```

Recovery: delete the patch and update this README.

### Scenario C — PyTorch pin bump unlocks newer XFormers

If our K80 fork ever upgrades PyTorch past 2.4 (and somehow keeps K80 support — unlikely without a custom build), we could move to a newer XFormers and consume the embedded-in-PyTorch cutlass kernels. Story 0.3 §5.2 documented why this is currently blocked by the R470 driver constraint. If that ever changes:

```bash
# Pivot point: stop maintaining xformers-patches/ entirely.
# The kernels live in PyTorch — patch them there instead.
```

## Conventions for new patches

- Name patches `<feature>-<scope>.patch`.
- Each patch must apply cleanly to v0.0.28.post3 (the pinned version). Test via `git apply --check`.
- Add a section to this README describing what the patch does, why, and what it does *not* do.
- Patches should be additive whenever possible.

## See also

- [`docs/port/flash-attn-mma.md`](../../../docs/port/flash-attn-mma.md) — Story 0.2's analysis (XFormers reference at §3.3).
- [`docs/port/cuda-11.4-version-pins.md`](../../../docs/port/cuda-11.4-version-pins.md) — Story 0.3's pin reasoning, including XFormers HEAD details (now superseded by the v0.0.28.post3 finding above).
- [`docker/k80/cutlass-patches/`](../cutlass-patches/) — sister directory for CUTLASS patches; Phase 1 verified.
- [`requirements/cuda_k80.txt`](../../../requirements/cuda_k80.txt) — references this directory and the pin.
