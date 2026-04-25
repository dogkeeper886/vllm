# CUDA 11.4 compatibility — version pin survey

**Story:** [#21][issue-21] (Phase 0 of epic [#12][epic])
**Pinned commits / tags for citations:**
- NVIDIA/cutlass: `7a9fe055cb` (HEAD, 2026-04-24)
- Dao-AILab/flash-attention: `ac6f2eb541` (HEAD, 2026-04-23)
- facebookresearch/xformers: `ca6d2aa0d4` (HEAD, 2026-04-21)

**Methodology:** every version claim cited by commit SHA, tag, or release URL.

[issue-21]: https://github.com/dogkeeper886/vllm/issues/21
[epic]: https://github.com/dogkeeper886/vllm/issues/12

---

## 1. Scope

This doc answers Story 0.3's three questions:

1. What's the latest CUTLASS / FlashAttention / XFormers tag that builds on CUDA 11.4?
2. Are there features past those tags we'd lose?
3. Is bumping our toolkit to CUDA 11.7 or 11.8 a viable alternative?

It also corrects an overcautious framing in Story 0.1's §8 about CUTLASS's CUDA 11.4 stance.

## 2. CUTLASS

### 2.1 The CUDA-minimum policy

CUTLASS's CMake gate at HEAD (`CMakeLists.txt:86–91`):

```cmake
if (CUDA_VERSION VERSION_LESS 11.3)
  message(WARNING "CUTLASS ${CUTLASS_VERSION} requires CUDA 11.4 or higher, ...")
elseif (CUDA_VERSION VERSION_LESS 11.4)
  message(WARNING "CUTLASS ${CUTLASS_VERSION} support for CUDA ${CUDA_VERSION} is deprecated, please use CUDA 11.8 or higher.")
endif()
```

The two warnings only fire when CUDA < 11.4. **CUDA 11.4 itself produces no warning at all.** The "strongly recommends 11.8+" string lives inside the warning text emitted to users on CUDA < 11.3 — at CUDA 11.4 the user never sees it.

This corrects Story 0.1 §8's framing that "CUDA 11.4 is at the edge of CUTLASS's tested matrix" — it's not at the edge. It's inside the supported zone. Story 0.1 read the warning text and assumed it applied at 11.4; it doesn't. The exact wording for what 11.4 gets is: nothing. No warning, no error.

### 2.2 When the policy was set

The current rule (deprecate < 11.3, deprecation-warn 11.3, accept 11.4+) has been in place since **CUTLASS v3.0.0** (commit `277bd6e5`, "CUTLASS 3.0.0 #786", 2023-01-23). Earlier, at `v2.11.0`'s `CMakeLists.txt`:

```cmake
if (CUDA_VERSION VERSION_LESS 10.2)
  message(WARNING "CUTLASS ${CUTLASS_VERSION} requires CUDA 10.2 or higher, and strongly recommends CUDA 11.0 or higher.")
elseif (CUDA_VERSION VERSION_LESS 11.0)
  message(WARNING "CUTLASS ${CUTLASS_VERSION} support for CUDA ${CUDA_VERSION} is deprecated, please use CUDA 11.0 or higher.")
endif()
```

So `v2.11.0` (Nov 2022 era) silently accepted CUDA 11.0–11.4 with no warnings whatsoever.

### 2.3 What changed between v2.11 and HEAD

**Nothing critical for our purposes.** The architecture support (`Sm50`, `Sm70`, `Sm75`, `Sm80`, `Sm90`) and dispatch chain Story 0.1 mapped is structurally the same at v2.11 and at HEAD. Newer architectures (`Sm100`, `Sm103`, `Sm120`) were added later but are unreachable from sm_37 anyway. The MMA atom inventory in `mma_sm50.h:61–143` (the SIMT scalar FMA) is unchanged.

### 2.4 Recommended CUTLASS pin

**HEAD is fine.** No reason to roll back. CUDA 11.4 builds without warnings. Story 0.1's static analysis transfers cleanly to whichever CUTLASS commit we pin.

If we want to be conservative — for example because Story 0.1's risk #2 (the `mma_simt.h:113` "Hard-coded for now" comment) implies upstream may parameterize the SIMT MMA's `ArchTag` in the future and break our patches — we can pin to a specific commit. **Suggested pin: HEAD `7a9fe055cb`** unless Story 0.6 finds reason to roll back.

## 3. FlashAttention

### 3.1 The CUDA-minimum bump trail

FlashAttention's CUDA minimum has been ratcheted up three times. Confirmed via `git log -p -S` against `setup.py`:

| Version | Tag commit | Date | CUDA minimum |
|---|---|---|---|
| v2.0.0 | `4f285b3547` | 2023-07-17 | `< 11.0` → RuntimeError |
| **v2.1.0** | `9e5e8bc91e` | **2023-08-21** | `< 11.4` → RuntimeError |
| v2.2.0 | `6d673cd9610` | 2023-09-05 | `< 11.6` → RuntimeError |
| (between v2.4 and v2.5) | `e371bea04f` | 2024-09-06 | `< 11.7` → RuntimeError |
| HEAD | `ac6f2eb541` | 2026-04-23 | `< 11.7` → RuntimeError |

The bump from 11.4 to 11.6 was commit **`0c04943fa2`** ("Require CUDA 11.6+, clean up setup.py", 2023-09-03), landed in v2.2.0.

### 3.2 The 11.4-compatible boundary

**v2.1.0 is the last tag that accepts CUDA 11.4.** Its `setup.py` check:

```python
if bare_metal_version < Version("11.4"):
    raise RuntimeError("FlashAttention is only supported on CUDA 11.4 and above")
```

Verified via `git show v2.1.0:setup.py`. The check uses `<` (not `<=`), so CUDA 11.4 itself passes.

### 3.3 What we'd lose by pinning to v2.1.0

This is the harder question. Between v2.1.0 (Aug 2023) and HEAD (Apr 2026) is roughly 2.5 years of development. Major themes from release notes:

- **v2.2 – v2.5:** prefix-LM masking, ALiBi, soft-capping, dropout fixes
- **v2.6 – v2.7:** Hopper SM90 support (irrelevant to K80)
- **v2.8.x:** Blackwell SM100/SM110 support (irrelevant to K80)
- **HEAD:** ROCm / Triton AMD path (Story 0.2 §7 noted it; not relevant to K80 either)

The functional features (prefix-LM, ALiBi, soft-cap) added between v2.1 and v2.5 are "nice to have," not must-have for a basic attention kernel. **Pinning to v2.1.0 is a real loss but not a blocking one** — vLLM doesn't require those features for the smoke-test workloads we're running today.

### 3.4 But here's the wall Story 0.2 already found

Story 0.2 established that **even FA v2.1.0 has the same kernel structure** Story 0.2 mapped: tensor-core MMA atoms (`SM80_16x8x16_F32F16F16F32_TN` / `SM75_16x8x8_F32F16F16F32_TN`), no SIMT path, the `ARCH_SUPPORTS_FLASH` preprocessor gate at `__CUDA_ARCH__ >= 800`. Pinning to v2.1.0 lets the **build succeed** on CUDA 11.4. It does not let the kernels run on sm_37.

So:

- For the **port path** (Story #37): v2.1.0 is the right pin if Phase 3 ever picked the port path. But Story 0.2 already recommended against the port path.
- For the **rewrite path** (Story #38): we don't compile FA at all — we just borrow algorithm ideas. The version pin question doesn't apply.

### 3.5 Recommended FA pin

**Conditional on Phase 3 strategy.** If 0.6 ratifies rewrite (Story #38): no pin needed. If the call ever changes: **v2.1.0** (`9e5e8bc91e`). I do not expect Phase 3 to need FA as a build dependency.

## 4. XFormers

### 4.1 No hard CUDA-version gate

`xformers/setup.py` at HEAD has **no `RuntimeError` based on CUDA toolkit version**. Verified by grep across the file.

The only version-related logic is conditional optimization:

- `setup.py:42` — `if cuda_version >= 1201 and cuda_version < 1202` — disables `--generate-line-info` for CUDA 12.1 (a known segfault). Skipped on 11.4.
- `setup.py:185` — `if cuda_version < 1205: sources.remove(...swiglu_fairinternal.cu...)` — drops a single optional kernel that needs CUDA 12.5's `cuda::ptx::cp_async_bulk`. We lose that kernel on 11.4; we don't care.

### 4.2 What XFormers does require

XFormers' CUDA gate is implicit, via its bundled CUTLASS submodule. From `.gitmodules` and `git ls-tree HEAD third_party/cutlass`:

- XFormers HEAD pins CUTLASS at commit **`8afb19d9`** (between CUTLASS tags v4.3.0 and v4.4.0).
- That CUTLASS commit inherits the CMake rule from §2 — accepts CUDA 11.4 with no warning.

So: **XFormers HEAD will build on CUDA 11.4** as long as the bundled CUTLASS does, which it does.

### 4.3 The XFormers wheel disclaimer

XFormers' published PyPI wheels target CUDA 12.6 / 12.8 / 13.0. **This is a distribution choice, not a build requirement.** Source builds against whatever local nvcc you have. Our K80 fork is already a source-build flow inside a custom builder image — wheels are not on the path.

### 4.4 Recommended XFormers pin

**HEAD is fine.** The Story 0.1 / 2.x patches needed are about adding `cutlass::arch::Sm37` and lowering XFormers' Python `CUDA_MINIMUM_COMPUTE_CAPABILITY` — not about toolkit-version gates.

## 5. CUDA toolkit + driver compatibility for Kepler

Story 0.1 left this as an open question. Resolving it here is critical because it determines whether we even *could* upgrade to CUDA 11.7+ to use a newer FA.

### 5.1 The toolkit side — Kepler in NVCC

NVIDIA's CUDA Toolkit release notes give a clear three-state timeline:

- **CUDA 11.0** ([release notes][cuda-11.0-rn]) — *"Support for the following compute capabilities are deprecated in the CUDA Toolkit: sm_35 (Kepler), sm_37 (Kepler), sm_50 (Maxwell)"*. This is when sm_37 became "deprecated but supported."
- **CUDA 11.8** ([release notes][cuda-11.8-rn], the last 11.x) — *"Support for the following compute capabilities are deprecated for all libraries: sm_35 (Kepler), sm_37 (Kepler)"*. Same deprecated-but-supported state.
- **CUDA 12.0** ([release notes][cuda-12.0-rn]) — *"Kepler architecture support is removed from CUDA 12.0."* Plus: *"Support for the following compute capabilities is removed for all libraries: sm_35 (Kepler), sm_37 (Kepler)"*. NVCC can no longer emit sm_37 binaries.

So at the **toolkit level**, CUDA 11.4, 11.6, 11.7, 11.8 all build sm_37 fine. There is no "going up to 11.7 loses Kepler" toolkit-side constraint.

[cuda-11.0-rn]: https://docs.nvidia.com/cuda/archive/11.0/cuda-toolkit-release-notes/index.html
[cuda-11.8-rn]: https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html
[cuda-12.0-rn]: https://docs.nvidia.com/cuda/archive/12.0.0/cuda-toolkit-release-notes/index.html

### 5.2 The driver side — this is the actual constraint

NVIDIA's R470 data-center driver release notes ([latest 470.256.02][r470-rn], 2024-06-04) state under "Supported NVIDIA Data Center GPUs":

> *"Release 470 will be the last driver branch to support Data Center GPUs based on the NVIDIA Kepler architecture."*

And under §1.1 "Software Versions":

> *"CUDA Toolkit 11: 11.4"*

Tesla K80 is listed in R470's supported "Data Center K-Series Products" table.

So:

- **R470 is the last NVIDIA driver branch that supports K80** (per the R470 release notes' own assertion).
- **R470 supports CUDA Toolkit 11.4 at runtime** (per the same release notes).
- Branches after R470 (R515 onwards in the production lineage) drop Kepler.

A K80 host running R470 **cannot load** binaries linked against the CUDA 11.7 runtime — the driver is too old. The binary fails at process start with a "driver version is insufficient for CUDA runtime version" error class.

This means our CUDA 11.4 pin is enforced by the **host driver**, not by the toolkit. The K80 self-hosted CI runner is on R470 because that's the last driver branch that supports K80. **Bumping the toolkit to 11.7 is not feasible without changing the driver, and there is no Kepler-supporting driver newer than R470.**

[r470-rn]: https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-470-256-02/index.html

This makes the FA bump from v2.1.0 (CUDA 11.4 OK) to v2.2.0+ (CUDA 11.6 required) a **hard wall for any port-path strategy**. Not because CUDA 11.6 lacks Kepler (it has it), but because the K80 host's R470 driver can't run anything compiled against 11.5+.

### 5.3 Implication for the epic timeline

The K80 fork's CUDA 11.4 pin is **not a deprecation curve we're trying to outrun**; it's a permanent constraint of the hardware-driver pairing. Story 0.1 §9's risk #3 was framed as "CUTLASS may eventually drop CUDA 11.4 entirely, capping the port's long-term value." That risk is real, but it doesn't get *worse* — there is no escape route through driver upgrades. The fork is permanently pinned at the (R470 driver, CUDA 11.4 toolkit) pair, and any library version that won't build against 11.4 is outside our reach forever.

This matters for Story 0.6's GO/NO-GO call. The work we land has a defined and bounded ecosystem to live in.

## 6. Recommended pin matrix

Subject to Story 0.6's consolidation:

| Library | Recommended pin | Justification | Builds on CUDA 11.4? |
|---|---|---|---|
| **CUTLASS** | HEAD (`7a9fe055cb`) | Builds on 11.4 with no warnings since v3.0.0; SIMT path Story 0.1 found is unchanged | ✅ |
| **FlashAttention** | Not pinned (rewrite path, Story #38) | Story 0.2 recommends rewrite, not port. v2.1.0 is the fallback if the recommendation is overturned. | v2.1.0: ✅ / HEAD: ❌ |
| **XFormers** | HEAD (`ca6d2aa0d4`) | No hard CUDA gate; bundled CUTLASS submodule (currently `8afb19d9`) accepts 11.4. Patches identified in Story 0.1 / 2.x are independent of toolkit version. *Caveat:* future XFormers releases could bump the bundled CUTLASS to a commit that drops 11.4 acceptance — risk lives one transitive dependency away. | ✅ |

## 7. Forced-upgrade dilemma — none, given the right pins

The pins above let the K80 fork stay on CUDA 11.4 indefinitely without losing anything that's actually reachable on sm_37 hardware. The libraries' newer features are either Hopper/Blackwell-only (irrelevant to K80) or quality-of-life additions (acceptable loss).

**The actual forced-upgrade pressure is one direction only:** if upstream CUTLASS ever drops the CUDA 11.4 acceptance from its CMakeLists.txt, we'd need to fork CUTLASS (or pin to a specific commit before the change). Currently CUTLASS shows no signal of doing this — the same rule has been in place since v3.0.0 (Jan 2023), so over 3 years of stability.

## 8. Open questions

1. **Driver R470 / K80 lifetime.** R470 is an LTS branch but NVIDIA could end LTS support at some point. If R470 stops receiving security updates, the host's other concerns (kernel updates, CVE patches) may push the operator to retire the K80 entirely. Story 0.6 should note this as a project-lifetime ceiling separate from any library version pin.

2. **vLLM minimum PyTorch / CUDA expectations.** This story didn't survey what vLLM (the parent project) requires. Our K80 fork already pins PyTorch 2.0.1 + CUDA 11.4 in the builder image — but as upstream vLLM raises its own floors, our fork will diverge further. Story 0.6 should note this as a separate divergence concern.

3. **The sm_37 deprecation warning at NVCC.** NVCC emits a deprecation warning for sm_37 codegen. We accept it; it is not a build failure. Worth verifying once Phase 1 actually compiles a CUTLASS sm_37 kernel (Story #28), to confirm no surprises.

## 9. Implications for Phase 1, 2, 3

**Phase 1 (CUTLASS):** No version-pinning blocker. Use HEAD. Story 0.1's findings transfer.

**Phase 2 (XFormers):** No version-pinning blocker. Use HEAD. The patches identified in Story 0.1 / 2.x (Sm37 arch tag in CUTLASS, lowered Python CC gate in XFormers) are independent of toolkit version. CUDA 11.4 build path works.

**Phase 3 (FlashAttention):** The version pin question is downstream of the port-vs-rewrite question. Story 0.2 recommended rewrite; this story does not change that. If Story 0.6 ratifies rewrite, FA is not a build dependency at all and v2.1.0 is irrelevant.

## 10. CUDA-only conclusions

- CUDA 11.4 sits comfortably inside CUTLASS's supported range; Story 0.1's "edge of supported" framing was overcautious.
- FA v2.1.0 is the last tag accepting CUDA 11.4. Newer FA versions require 11.6, 11.7. Pinning is not the right solution — Story 0.2's rewrite recommendation makes FA's version irrelevant.
- XFormers has no hard CUDA gate; its bundled CUTLASS submodule is the only effective constraint, and it accepts 11.4.
- The K80 fork's CUDA 11.4 pin is enforced by the **host driver R470**, not by the toolkit. Upgrading the toolkit to 11.7+ to use newer FA is not feasible because no Kepler-supporting driver newer than R470 exists.

## 11. Correction to Story 0.1 §8 + §9 risk 3

Story 0.1's "edge of supported matrix" framing for CUDA 11.4 + CUTLASS was overcautious. CUDA 11.4 produces no warning at any current CUTLASS version since v3.0.0 (Jan 2023). The "strongly recommends 11.8+" string is in a message text only displayed to users on CUDA < 11.3. **At our pin we see neither warning.**

Story 0.1 §9 risk 3 ("CUTLASS deprecation curve") is still valid — *future* CUTLASS releases could drop 11.4 acceptance — but the framing should reflect that the current state is safe, not that we're already on borrowed time.

I won't roll those edits into Story 0.1's doc directly (one-story-per-PR discipline); instead, Story 0.6 should consolidate this correction when it produces the final feasibility writeup.

---

**End of Story 0.3 deliverable.** Story 0.4 (Kepler hardware reality check, [#22][s04]) is the next pickup. It quantifies what sm_37 actually has vs sm_50 — feeding the assumption underlying Story 0.1 §5 that the SIMT path's hardware requirements are met by Kepler.

[s04]: https://github.com/dogkeeper886/vllm/issues/22
