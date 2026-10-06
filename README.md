# MI50 local LLM tuning log

**Card:** AMD Instinct MI50 32 GB (gfx906 / Vega 20 / GCN5, ~1 TB/s HBM2, **no matrix cores**, passive-cooled). Replaced an Intel Arc B580 (permanent swap).
**Host:** Nobara/Fedora, Ryzen 7 8845HS (8c/16t), 30 GB RAM. SELinux disabled, Docker.
**Workload:** n8n LLM node. "Big data in → small data out" (prefill-dominated), single-user (`--parallel 1`).
**Compiled:** 2026-07-03. Consolidates every test from 2026-06-20 to 2026-07-03.
**Source reports:** `MI50-report.md`, `MI50-MTP-speculative-decoding-analysis.md`, `MI50-Q5-Q6-quant-test.md`, `DFlash-setup.md`, `dflash-bench-2026-06-29.md`, `ornith-mtp-optimization-report.md`, `ornith-model-docker-logs-2026-07-01.md`, `llama-stats-2026-06-23.md`.

> **Current production (2026-07-03):** `llama-hipgraphs` container, image `llama-hipgraphs:upstream-rocm-7.2.4` (build commit `0eca4d4`, 2026-06-30), model **Ornith-1.0-35B Q4_K_M + embedded MTP**, `--spec-type draft-mtp --spec-draft-n-max 2`, 262 k ctx, q8_0 KV, `--parallel 1`, port 8089. About 70 t/s.

---

## Timeline at a glance

| Date | Phase | Headline result |
|---|---|---|
| 06-20 | Hardware + Vulkan era | Thermal validated; MoE beats dense; **prefill is the weak spot** |
| 06-20 | **ROCm vs Vulkan** | **8.7× decode** (9.4 → 82 t/s). Vulkan was crippling the card |
| 06-20 | MTP economics on MoE | Net-negative on hard content; break-even ≈ 42 % accept |
| 06-20 | Q5/Q6 quant test | No quality gain over QAT-Q4_0; stay Q4 |
| 06-29 | DFlash on Qwen3.6-35B | Custom build needed; +14-25 %; peak n_max 2-3 |
| 06-30 → 07-01 | **Ornith-1.0-35B + MTP** (current) | ubatch +46 %, HIP-graphs +12 %, **n-max tuned 6 → 2 = +19 %** |
| 07-02 → 03 | Drafter ecosystem scans | Nothing beats stock MTP for Ornith |
| **07-03** | **Today: GLM / DFlash / EAGLE3 / bare-metal** | DFlash > EAGLE3 on Gemma; **Docker = 0 % overhead** |

---

## Phase 0: Hardware and thermal validation (2026-06-20)

Thermal test: 3-min sustained Vulkan load, `rocm-smi -d 0` every 2 s.

| Time | Junction | sclk | State |
|---|---|---|---|
| idle | 34 °C | 1725 | - |
| 40 s | 90 °C | 1725 | full boost |
| 60 s | **100 °C** | 1725 | **throttle onset** |
| 111 s | 100 °C | **930** | hard throttle |
| peak | **102 °C** | - | mem 87 °C (not the limiter; the die hotspot is) |
| +4 s idle | 77 °C | - | cools fast |

Verdict: the passive MI50 throttles at about 100 °C after about 60 s of sustained load (1725 → ~930-1143 MHz). It protects itself and does not get damaged. **Bursty n8n loads are fine** (they never reach the 60 s threshold). Longer prefill sustains more, so it throttles harder.

---

## Phase 1: Native Vulkan era (2026-06-20), later fully superseded

Stack: native `llama.cpp` b9565 + Vulkan (RADV), `--device Vulkan1`. Models: Gemma 4 **12B/31B QAT** + **26B-A4B MoE QAT**.
Note: `llama-bench`/`llama-cli` are ABI-broken here, so **all benchmarks used `llama-server` `/completion` timings** (also the only way to read MTP acceptance).

### 1a. MTP draft-depth sweep: Gemma 31B dense, 128k, F16 KV, temp 0

| k (`--spec-draft-n-max`) | gen t/s | MTP accept | peak junction |
|---|---|---|---|
| 1 | 8.36 | 78.5 % | 66 °C |
| 2 | 8.88 | 66.9 % | 75 °C |
| **3** | **9.00 (peak)** | 75.8 % | 80 °C |
| 4 | 8.39 | 36.0 % | 82 °C |
| 5 | 7.86 | 46.3 % | 82 °C |
| 6 | 7.15 | 41.6 % | 87 °C |

k=3 is optimal, with an acceptance cliff at k≥4 (the head predicts about 3 steps ahead). No thermal confound here (all runs stayed under 100 °C).

### 1b. Deep-128k prefill (the weak spot) on Vulkan

Prefill runs at about **165 t/s** at 115k, with the GPU at only ~93 W / 66 °C, so it is underutilized and backend-limited, not thermal. `ub=2048` did not help. Latency: 4k ≈ 24 s, 16k ≈ 1.5 min, 60k ≈ 6 min, 115k ≈ 11.5 min. Mitigation: KV-prefix reuse (stable bulk first, varying tail last).

### 1c. Dense vs MoE three-way: k=3, 128k, F16 KV, temp 0

| Config | arch | gen t/s | prompt t/s | MTP accept | VRAM |
|---|---|---|---|---|---|
| UD-Q4_K_XL 31B + unsloth assistant (267 MB) | dense | 9.0 | ~31 | **75.8 %** | ~18 GB |
| Official Q4_0 31B + official assistant f16 (911 MB) | dense | 3.9 | 32 | ~54 % | ~17 GB |
| Official Q4_0 31B + official assistant q8_0 (491 MB), re-run | dense | 5.45 | 32 | 54.5 % | ~17 GB |
| **★ Official Q4_0 26B-A4B + q8_0 assistant (441 MB)** | **MoE, ~4B active** | **10.9** | **~120** | ~62 % | ~17 GB |

**MoE wins**: prefills about 4× faster (the metric that matters) plus the best generation speed, so it was adopted as the endpoint. (An f16 head costs ~40 % of decode; acceptance is set by base-model fidelity and content, not head quant.)

---

## Phase 2: ★★★ ROCm/HIP vs Vulkan, the decisive finding (2026-06-20)

Same MI50, same Gemma-4-26B Q4_0 raw, Docker `mixa3607/llama.cpp-gfx906:b9728-rocm-7.2.3`:

| metric | Vulkan (RADV) | **ROCm/HIP** | gap |
|---|---|---|---|
| Decode (gen) | 9.4 t/s | **~82 t/s** | **~8.7×** |

**Vulkan was crippling the card the whole time.** gfx906's dp4a INT8 MMQ kernels and efficient MoE expert path exist on ROCm/HIP but **not** on RADV/Vulkan. Every earlier "weak spot" was really the wrong backend. **Migrated the n8n endpoint to ROCm** (reversing the Phase 1 "native Vulkan" decision). This is the single biggest win in the whole log.

---

## Phase 3: MTP economics on the MoE (2026-06-20)

MTP re-tested on ROCm (the draft is now fast). The result is content-dependent:

| prompt | accept | MTP gen | raw gen |
|---|---|---|---|
| predictable | 52 % | **94 t/s** | 82 |
| harder | 29 % | 70-74 | 82 |

Break-even, reconstructed: a speculative cycle costs `C_cycle ≈ 1.66×` a normal decode (self-consistent across both runs), so the **break-even acceptance is about 41-42 %**. 52 % accept gives +15 %, 29 % gives −15 %. Root cause: **MoE breaks "free parallel verify"**. Verifying K tokens activates the *union* of each token's top-8 experts (sub-linear, about 2.5× for 4 tokens, not the full K×8). A small ~4B active footprint leaves little for spec-decode to recover.
The endpoint therefore **ran RAW (82 t/s)** on the MoE: MTP is net-negative on mixed or hard content. Untested lever flagged: confidence gating with `--spec-draft-p-min`.

### 3a. Draft-head precision sweep (γ=3, predictable prompt)

**26B MoE** (Q5_K_XL target): raw 69.0. **Q4_0 head (252 MB) → 90.5 t/s at 58 % accept.** Q8_0 (461 MB) → 85.1 / 55.7 %. F16 (855 MB) → 79.4 / 55.7 %. **F16 and Q8_0 give byte-identical acceptance** (same argmax), so a bigger head is only slower. **Use the Q4_0 head; F16 is a trap.**
**31B dense**: raw 22.66. Q4_0 (349 MB) → 37.57 t/s / 63.6 %, Q8_0 (515 MB) → 38.01 / 65.2 %, F16 (955 MB) → 37.34 / 66.9 %. Speed is flat and acceptance rises slightly with precision, so **Q8_0 is the sweet spot**.
**Rule:** on MoE use the smallest head (verification is cheap, and head bytes are visible); on dense models the head is a rounding error. "Smaller head wins" is **MoE-specific**. MTP pays far better on dense (+67 %) than on MoE.

---

## Phase 4: Q5/Q6 quant test, Gemma 26B-A4B MoE (2026-06-20)

Tested unsloth UD-Q5_K_XL (21.2 GB) and UD-Q6_K_XL (23.3 GB) against QAT-Q4_0 in an isolated container on :8090; prod was restored afterwards.

**Quality** (45-prompt greedy eval): Q4_0 / Q5 / Q6 all get **25/25 easy and 18/20 hard, with the same two failures (gsm3, gsm4) and the same wrong answer "120"**. No task-correctness difference. (Caveat: short prompts do not probe long-form; PPL was discarded as an ≈880 artifact.)
**Speed** (q8_0 KV): Q4_0 74.3, Q5 68.5 (−6 %), Q6 68.1 (−12 % vs Q4 F16). The q8_0 KV itself costs ~7-9 %.
**MTP acceptance does not rise with quant**: Q4 59.6 % > Q5 55.7 % > Q6 51.6 % (it actually trends down).
**VRAM** (after reclaim): Q5@48k F16 ~23.7 GB, Q6@48k q8_0 ~25 GB. Context is nearly free (Gemma SWA); model size is the binding constraint.
**Verdict: stay on QAT-Q4_0.** Q5/Q6 are all cost and zero measurable gain.
**Side win:** found and killed a **phantom 8.2 GB VRAM baseline**: a legacy `prompt.service` (native-Vulkan 12B on :8088, started at boot). `systemctl disable` freed the full ~32 GB.

---

## Phase 5: DFlash on Qwen3.6-35B-A3B (2026-06-29), custom build

Drafter `z-lab/Qwen3.6-35B-A3B-DFlash` (0.4B), converted to GGUF `dflash-draft.gguf` (747 MB F16). This **needed a self-built GPU llama.cpp** (the stock gfx906 images lacked `draft-dflash`; minimum b9831). Trap: a build compiled with `GGML_HIP=OFF` (CPU-only) requires mounting the whole `build-rocm/bin` plus `LD_LIBRARY_PATH`.

**Verified working:** 30.5 GB VRAM (35B Q5 + draft + 128k q8_0 KV), **86 % accept (193/225), decode 92.5 t/s** (vs ~71-79 plain), prefill 119 t/s.

**Block size** (`--spec-draft-block-size`, concurrency 1): no-spec ~58, bs8 70.5 / 54 %, **bs16 72.7 / 57 %**, bs20 74.0 / 58 %, ngram ~66. Block 16 is best for our concurrency-1 use; bs8 for high concurrency; **bs8/bs20 deleted**.

**n_max sweep** (Q5, temp 0): no-spec 63.9, **n2 72.5 / 68 % (+14 %)**, n3 72.2 / 57 %, n4 57.8 / 51 % (−10 %), n5 58.1 / 46 %, n7 52.4 / 36 %, n10 30.9 / 25 %, n15 29.2 / 17 %, n19 23.7 / 14 %. A clean inverted curve that peaks at n2-3 and collapses at ≥4. Cause: MoE cheap-verify. Determinism: greedy is not bit-reproducible across draft widths (FP non-associativity; 3/6 identical, 3/6 diverge coherently, which is benign).

---

## Phase 6: Ornith-1.0-35B + MTP (2026-06-30 → 07-01), current production

Migrated the live model to **Ornith-1.0-35B** (`qwen35moe` arch: Qwen3.5 + Gemma-4 base, hybrid SSM/attention; RL-tuned agentic coding/tool-use). The model file `ornith-1.0-35b-Q4_K_M-MTP.gguf` (21 GB) ships the MTP head **embedded** (block 40, `nextn.*`), so there is no separate draft, just `--spec-type draft-mtp`. `--no-mmap` is mandatory here (30 GB RAM page-cache thrash).

### 6a. Optimizations

| Change | Effect |
|---|---|
| ubatch 512 → **4096** | prompt +46 % (833 → 1214 t/s), short gen +50 % (~60 → ~90); VRAM 78 % → 90 % |
| **HIP-graphs build** (`GGML_HIP_GRAPHS=ON`) | +12 % gen (cuts kernel-launch overhead; batch 1 is latency-bound, ~10 % bandwidth) |
| ngram vs MTP | MTP 63.6 avg at 44 % (stable) beats ngram ~61.0 at 35 % (erratic 14-58 %). MTP wins |
| f16 K cache | 2.5× *slower*, rejected (no tensor cores) |
| Power-limit / undervolt | skipped; GPU only here and not worth the risk |

### 6b. `--spec-draft-n-max` tuning (2026-07-01), full sweep, ~4k code prompt, temp 0

| n-max | gen t/s | accept |
|---|---|---|
| 1 | 68.2 | 79 % |
| **2** | **70.2 (winner)** | 64 % |
| 3 | 67.9 | 53 % |
| 4 | 66.2 | 53 % |
| 6 | 59.0 | 41 % |

It was deployed at **n-max 6 (the slowest option)**, so switching to **2 is a +19 %** gain. Real n8n acceptance is ~40 % (temp 0.6), which pushes the sweet spot even lower, so 2 is safe.

### 6c. Live n8n request log (2026-07-01, 25 requests)

Gen **40.81-100.28 t/s (avg ~62.5)**, prompt 575-1214 t/s (avg ~931), draft accept 0.24-0.78 (avg ~0.46). Fastest 100.28 t/s (short output, high KV reuse); slowest 40.81 t/s (7.8k prompt). Generation degrades with context length.

### 6d. Hardware ceiling

~70 t/s is roughly all a single MI50 does on an A3B MoE. Independent llama-bench (Qwen3-Coder-30B-A3B, same class): Q4_K_M 73 t/s at 128 → 63 at 1k → **21 at 16k**. An RTX 3090 (similar bandwidth, mature CUDA) does Qwen3.6-35B-A3B Q4 at ~135 t/s. The ~2× gap is silicon (dp4a-only, no MFMA) and not tunable. **vLLM-gfx906 forks are dead ends** (MoE slow, no MTP, archived Feb 2026).

---

## Phase 7: Drafter ecosystem scans (2026-07-02 → 07-03)

- **57 Ornith HF repos checked.** Every usable MTP GGUF grafts the *same* official head, so nothing to gain there. A KL-**distilled** MTP head (`shisa-ai/...qwen36-distill`) measured *worse* than the official zero-training head (distilled 67 % / +17 % vs official 67 % / +21.4 %). A better head was tried and lost.
- **No Ornith-native DFlash or EAGLE3 exists.** The DFlash 35B GGUFs are all **Qwen3.6** (wrong base; Ornith is `qwen35moe`), confirmed from the running model's GGUF header. A Qwen3.6 drafter on our Qwen3.5-hybrid target rejects the load and gets ~0 % accept.
- **gfx906 upstream: zero PRs merged after our 06-30 build.** The relevant ones (#21168, #24668) are already in. The `arte-fact/...turbo` fork trades 3.3× context for −18 % speed (the wrong direction). Stanford hipkernels are MI350X-only. **Future watch:** AMD is reportedly re-adding official gfx906 ROCm support in the next major version.

---

## Phase 8: 2026-07-03

### 8a. GLM-4.7-Flash test (aborted)

`unsloth/GLM-4.7-Flash UD-Q5_K_XL` (21.7 GB, 30B-A3B MoE, deepseek2 arch, 131k ctx). It loaded fine: ~60 t/s, 27.5 GB VRAM, coherent. **MTP is impossible** because the GGUF has no `nextn` tensors and GLM/deepseek2 MTP is unmerged (PR #24868 draft). Reverted to Ornith as requested; container removed. (The GLM GGUF is still on disk, 21.7 GB, deletable.)

### 8b. Gemma-4-26B-A4B + DFlash: DFlash IS net-positive on gfx906

Target `unsloth/gemma-4-26B-A4B-it UD-Q4_K_M` (16.95 GB) plus `Alittlehammmer/...DFlash-Q8_0` (471 MB, z-lab head). Our build **has a working DFlash runtime** (`common_speculative_impl_draft_dflash ... block_size=16`, non-zero acceptance). At Q4/16k/temp 0:

| config | gen t/s | accept | vs baseline |
|---|---|---|---|
| baseline (no spec) | 55.9 | - | - |
| DFlash n-max 6 | 64.4 | 36 % | +15 % |
| **DFlash n-max 3** | **75.2** | 54 % | **+35 %** |

This overturns the earlier "DFlash net-negative at Q4 sub-Blackwell" finding. The difference is a **base-matched** drafter. Gemma+DFlash (75) edges out Ornith+MTP (70) on raw t/s, but it is a *different, general* model, so Ornith stays in production for quality.

### 8c. Gemma-4-26B-A4B + EAGLE3: DFlash wins decisively

`williamliao/...speculator.eagle3-Q8_0` (0.99 GB, RedHat's official 0.9B head), `--spec-type draft-eagle3` (runtime works). At Q4/16k/temp 0 (baseline 55.9):

| method | drafter | gen t/s | accept | vs baseline |
|---|---|---|---|---|
| **DFlash n3** | 0.4B | **75.5** | 54 % | +35 % |
| EAGLE3 n3 | 0.9B | 59.1 | 32 % | +6 % |
| EAGLE3 n5 | 0.9B | 46.9 | 21 % | **−16 %** |

EAGLE3 loses on **both** axes: lower acceptance (32 vs 54 %) plus a heavier 0.9B drafter. Caveat: the llama.cpp EAGLE3 (PR #18039) appears to be chain-draft, not tree, so this is EAGLE3 as it runs here, not its vLLM ceiling. **Ranking on gfx906 + llama.cpp: DFlash > EAGLE3 > baseline.**

### 8d. Docker vs bare-metal A/B: container overhead = 0 %

Extracted the *exact* binary and ROCm libs from the image and ran natively with `LD_LIBRARY_PATH`, versus in-container, with identical flags, model, and GPU:

| config | Docker | Bare-metal | Δ |
|---|---|---|---|
| baseline | 55.83 | 55.96 | +0.2 % |
| DFlash n3 | 75.21 | 75.41 | +0.3 % |

Acceptance is byte-identical (54 %, 308/572 both). On native Linux, container GPU work uses the **same host-kernel amdkfd driver** via `/dev/kfd`, so there is no GPU virtualization. (This is distinct from the B580 pain, which was a Vulkan→CPU *fallback*, a wrong code path, not container tax.) The 17 GB extraction was deleted afterwards.

---

## Standing conclusions (what to actually do)

1. **Production = Ornith-1.0-35B Q4_K_M + embedded MTP, n-max 2, ROCm/HIP Docker.** Don't touch it.
2. **~70 t/s is the gfx906 A3B ceiling.** 2× needs new silicon, not tweaks. It is well tuned; no speed is being left on the table.
3. **Backend: ROCm/HIP only** (8.7× over Vulkan). **Docker adds nothing** (proven on bare metal).
4. **Quant: Q4 (QAT/UD).** Higher quant is all cost and no measurable gain. Q4 is also the *fastest* (bandwidth-bound).
5. **Speculator by regime:** embedded MTP for Ornith (the only base-matched option); for a *base-matched* Gemma, **DFlash beats EAGLE3**. External drafters need a base-model match; an arch mismatch is a dead end.
6. **Head/drafter: smaller wins on MoE** (cheap verify); low n-max wins everywhere on this card (acceptance collapses with depth).

## Reclaimable disk (test leftovers)

- `GLM-4.7-Flash-UD-Q5_K_XL.gguf` (21.7 GB). Aborted 07-03 test, safe to delete.
- `models/dflash-draft/` (~1.5 GB: `dflash-draft.gguf` 747 MB + safetensors 772 MB). Qwen3.6 DFlash, not used by the live server.
- `Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf` (26.6 GB) and `ornith-1.0-35b-Q5_K_M.gguf` (24.7 GB). Non-production alternates; keep only if wanted.
