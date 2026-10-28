# Faster prompt processing (prefill) on older NVIDIA cards

Scope: Tesla P100 (sm_60, no `dp4a`, no tensor cores, 16 GB HBM2) and GTX 1070 (sm_61, has `dp4a`), and whether prefill can run on
the CPU while the GPU decodes. This is an investigation: every statement about the code cites `path:line` in this repo
(branch `feat/lookahead-expert-prefetch`). Numbers are either quoted from a file here with their line, or sit under
"Hypotheses (untested)". Nothing was benchmarked for this note.

Fleet facts used (given by the operator, not derived from this repo): cesarops4om = 2 x Xeon Gold 6148 (40 cores / 80 threads,
AVX-512F/BW, no VNNI), 117 GB RAM, 2 x P100, Tesla T10 (sm_75), GTX 1070, RTX 2060. t440 = Xeon Gold 6226 (AVX-512 VNNI), 2 x P100.
cesarops2 = Xeon E5-2450 v2 (AVX only). Measured on cesarops4om today: Tiel (Ornith-1.5-35B-A3B, Q5_K_XL) 869-877 tok/s prompt on the
T10 and 690-694 tok/s on the 2060 for a 5,125-token prompt; the Qwen3.8 coder on 2 x P100 decodes about 39 tok/s. This repo does not
say which engine binary serves Tiel; the call path below is the one in this repo.

## Where prefill runs today

1. The Python server starts the engine as a child process in serve mode: `serve/server.py:832`.
2. The engine's `generate.cpp` holds one `strata::prefill::Prefill sp` per session (`src/program/generate.cpp:1093`), initialises it with
   the weights, the expert cache and the chunk size (`src/program/generate.cpp:7610`) and reads a prompt in pieces with
   `sp.run(ids, n, pos0, err)` (`src/program/generate.cpp:11581`). The class contract is in `include/strata/prefill/prefill.hpp:129`.
3. `Prefill::run` calls `run_impl` and then drains the layer-split pipeline (`src/prefill/prefill.cpp:4338`). `run_impl` is
   `src/prefill/prefill.cpp:2087`; its chunk loop is `src/prefill/prefill.cpp:2178`, each chunk goes through all layers.
4. Per layer, the dense projections are `Gemm` calls. `Gemm::f16` ends in `cublasGemmEx` on FP16 operands with FP32 accumulate
   (`src/prefill/gemm.cu:728`, call at `src/prefill/gemm.cu:785`). BF16 weights go through `Gemm::bf16` (`src/prefill/gemm.cu:624`), which on a
   card below sm_70 is told to widen to FP32 and call `cublasSgemm` (`bf16_path` returns 2 for cc < 70 at `src/prefill/gemm.cu:528`; the
   Sgemm call is `src/prefill/gemm.cu:702`).
5. The routed experts take one of three routes, chosen at `src/prefill/prefill.cpp:3031`:
   - llama.cpp's MMQ kernels (`mmq::Context::run`, gate/up at `src/prefill/prefill.cpp:3913`, down at `src/prefill/prefill.cpp:3927`). The MMQ
     plan is built in `src/prefill/prefill.cpp:1012`; it is turned off with `STRATA_PREFILL_MMQ=0` (`src/prefill/prefill.cpp:1016`) and a
     layer falls back when no MMQ tile fits the card (`src/prefill/moe_mmq.cu:225-237`).
   - the fused kernels (`src/prefill/moe_fused.cu`, `src/prefill/moe_fused_iq.cu`) for streamed chunks;
   - dequantise to FP16 and run two `Gemm::f16` calls per expert (`src/prefill/prefill.cpp:3947`).
6. Non-resident experts are streamed over PCIe through a ring; chunks of `stream_all_min()` = 1024 tokens or more stream every expert
   the GPU does not hold (`src/prefill/prefill.cpp:164-167`).
7. A CPU share already exists: with a pool set (`include/strata/prefill/prefill.hpp:182`), chunks below the stream threshold may hand some
   non-resident experts to the decode CPU pool, which `src/prefill/prefill.cpp:3447` runs while the GPU does the rest. It is on by default for
   chunks below 1,024 tokens on one CUDA GPU without batch slots (`src/prefill/prefill.cpp:178-182`, `src/prefill/prefill.cpp:2348`).
   This is CPU help inside a prompt, not CPU prefill alongside a GPU decode.

Where the time goes on a Pascal card: the repo has one profile, on a Tesla P40 (sm_61), not a P100. For a 7,038-token prompt the
dense projection GEMMs are 36%, the experts (MMQ) 28%, the hyper-connection read 13% and attention 12%
(`bench/results/2026-10-12-p40-sm61-kernels/README.md:47-49`). That note also records that ggml's MMQ "has its own Pascal dp4a configuration"
and that no larger kernel lever was found (`bench/results/2026-10-12-p40-sm61-kernels/README.md:56-58`). No equivalent P100 profile exists in this
repo.

## What older cards lack

- `dp4a` (int8 dot product): present from sm_61, absent on sm_60. `strata_dp4a` replaces it below `__CUDA_ARCH__ < 610` with four PTX `vmad`
  instructions (`include/strata/kernels/dp4a.hpp:27-40`); the header says it is bit-exact and about 2.2x faster than the byte-wise form in
  isolation on a P100 (`include/strata/kernels/dp4a.hpp:31`, checker `tools/sm60_dp4a_check.cu:1`). A GTX 1070 (sm_61) uses the real `__dp4a`
  (`include/strata/kernels/dp4a.hpp:42`). Whether llama.cpp's own MMQ code, compiled from a fetched checkout
  (`CMakeLists.txt:1313-1314`, MMQ build at `CMakeLists.txt:1416`), uses this header or its own fallback is not visible in this repo
  (`mmq.cuh` is not vendored; `third_party/ggml` holds only `ggml-common.h`, the licence and a version file).
- `__nanosleep` (sm_70): on sm_6x the doorbell spin loop simply spins (`include/strata/kernels/dp4a.hpp:48-50`).
- Tensor cores: none on Pascal. The prompt attention MMA kernel needs sm_75 and compiles to a trap below it
  (`src/kernels/cuda/qsa_prompt_attn.cu:33-35`, `src/kernels/cuda/qsa_prompt_attn.cu:50-51`); the host keeps the older kernel there. The Volta
  `mma.m8n8k4` kernel is compiled only with the experimental flag (`src/kernels/cuda/qsa_prompt_attn.cu:743-745`) and is for sm_70, not Pascal.
- BF16 GEMMs: cuBLAS has no BF16 GEMM for Pascal, so both operands are widened to FP32 (`src/prefill/gemm.cu:474-475`); for cc >= 70 and < 75 they
  go through FP16 instead (`src/prefill/gemm.cu:530`).
- cuBLAS kernel choice: on a P40 with cuBLAS 12.4 the FP16-operand GEMM picks a slow Maxwell kernel for most chunk sizes (4.3-4.6 TFLOPS
  against 7.6 for the fast one) and `cublasSgemm` on FP32 copies reaches 8.4-9.8 (`src/prefill/gemm.cu:544-556`; sweep in
  `bench/results/2026-10-12-p40-sm61-kernels/README.md:73-82`). This is opt-in as `STRATA_DENSE_F32=1` and Pascal-only
  (`src/prefill/gemm.cu:557-572`), because the bitwise equality was measured with one cuBLAS build
  (`bench/results/2026-10-12-p40-sm61-kernels/README.md:92-94`).
- What `STRATA_EXPERIMENTAL_SM60` changes. CMake: refuses cards below sm_75 unless it is set and accepts sm_60 or newer
  (`CMakeLists.txt:183-190`, `CMakeLists.txt:165`); defines the macro for every target (`CMakeLists.txt:199-202`) and for `strata_core`
  (`CMakeLists.txt:315-317`). Runtime: lowers the device-check floor from cc 7.5 to 6.0 (`src/core/device.cu:304-306`). Kernels: the
  Pascal BF16-to-FP32 path (`src/prefill/gemm.cu:490-499`), the `STRATA_DENSE_F32` route (`src/prefill/gemm.cu:543`), the Volta prompt-attention
  kernel (`src/kernels/cuda/qsa_prompt_attn.cu:745`) and the decode attention variant (`src/kernels/cuda/qsa_decode_attn.cu:291`). It needs a
  CUDA 12.x toolkit because CUDA 13 dropped sm_60 (`CMakeLists.txt:181-182`).
- Shared memory and registers: MMQ picks a tile whose shared-memory footprint fits the card (`src/prefill/moe_mmq.cu:236`); a card with no fitting
  tile sends that product to the non-MMQ path with a log line (`src/prefill/moe_mmq.cu:243`). Register limits are not discussed anywhere in
  the files read for this note.
- CPU side, for the "prefill on CPU" question: the Strata CPU expert kernels have an AVX-512 tier that requires F, BW, VL, VNNI and VBMI
  together (`src/kernels/cpu/expert_layout.cpp:71`, feature struct `include/strata/kernels/cpu/expert.hpp:99-101`). The i-quant dispatch uses the
  AVX-512 kernels only when that check passes (`src/kernels/cpu/native_expert.cpp:150`); otherwise it falls to the AVX2 kernel. By the fleet
  facts, cesarops4om's Gold 6148 has no VNNI, so it would take the AVX2 tier; t440's Gold 6226 has VNNI and would take the AVX-512 tier.

## Options

Each effect below is a hypothesis; none was measured.

1. **P100 profile first, then `STRATA_DENSE_F32=1` on sm_60.**
   Change: no code; build with `-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=60` and run the existing opt-in route
   (`src/prefill/gemm.cu:557-572`). Files: none. Effect (hypothesis): the P40 gain (7K prompt 360-432 to 464-474 tok/s,
   `bench/results/2026-10-12-p40-sm61-kernels/README.md:18`) may or may not carry to GP100, whose FP32/FP16 rates differ.
   Measure: A/B with `STRATA_PREFILL_TIMING=1` (`src/prefill/prefill.cpp:2000`) and run `gemm_f32_parity` (`CMakeLists.txt:877`) first, because the
   bitwise equality belongs to one cuBLAS build.
2. **dp4a-free int8 path for sm_60 in the fused/native experts.**
   Change: the native expert kernels use `STRATA_DP4A` where they use int8 dots; on sm_60 that is four `vmad` per call
   (`include/strata/kernels/dp4a.hpp:33-36`). Alternatives to try are `__vabsdiff4`/`byte_perm` repacking or FP16 `HFMA2` on dequantised
   weights. Files: `include/strata/kernels/dp4a.hpp`, `src/prefill/moe_fused_iq.cu`, `src/prefill/moe_fused.cu`. Effect (hypothesis): a P100 is said to have
   double-rate FP16 (a hardware fact not checked in this repo), so dequantise-to-half2 may beat 4 x `vmad` for large chunks. Measure: `tools/sm60_dp4a_check.cu` for bit-exactness, then
   `src/prefill/mmq_expert_bench.cpp` for the expert product alone on a P100.
3. **MMQ versus cuBLAS FP16 GEMM for P100 experts.**
   Change: route experts through the existing dequant + `Gemm::f16` path (`src/prefill/prefill.cpp:3947`) instead of MMQ on sm_60. Files: none for a
   first test (`STRATA_PREFILL_MMQ=0`, `src/prefill/prefill.cpp:1016`). Effect (hypothesis): MMQ's Pascal dp4a configuration, which on sm_60 pays for
   the `vmad` fallback if it uses it, may lose to FP16 GEMM on 16 GB HBM2 card at large chunk sizes. Measure: prompt ms for 512, 2K and 8K
   tokens, flag on and off, with `STRATA_PREFILL_TIMING=1`.
4. **CPU AVX-512 prefill for the first layers or the whole prompt, KV handed to the GPU.**
   Change: extend the CPU share from experts only (`src/prefill/prefill.cpp:3447`) to whole layers, which needs the CPU to run attention/GDN
   and write KV cells in the GPU session layout. Files: `src/prefill/prefill.cpp`, `include/strata/prefill/prefill.hpp`,
   `src/kernels/cpu/*`. Effect (hypothesis): worth it only where the CPU tier is VNNI (t440); on the Gold 6148 the AVX2 tier applies
   (`src/kernels/cpu/native_expert.cpp:150`). Measure: the CPU expert bench (`src/kernels/cpu/kq_cpu_bench.cpp`) against the GPU's per-expert
   time, using the same quotient the share already measures (`src/prefill/prefill.cpp:171-175`).
5. **Widen the CPU share to larger chunks on P100.**
   Change: `STRATA_PREFILL_CPU_SHARE_MAX=3072` (`src/prefill/prefill.cpp:198-208`) is opt-in because it lost 2% at 2,048 tokens on an RTX 5070
   (`docs/DETAILS.md:1630`). On a P100 with 40 cores it may differ. Files: none. Measure: prompt ms at 1K, 2K, 3K tokens with the variable set and
   unset (`STRATA_PREFILL_CPU_SHARE=0` is the off switch, `src/prefill/prefill.cpp:181`).
6. **Pipeline/expert parallelism across the cards of one node.**
   Change: the layer split and `set_stage_helper` already let a second card stream part of the experts (`include/strata/prefill/prefill.hpp:156-172`,
   opt-in `STRATA_PREFILL_HELP=1`). Test on the two P100s of t440 or cesarops4om. Files: none. Effect (hypothesis): the helper only pays for
   one-chunk prompts of 1,024+ tokens. Measure: prompt ms with and without the variable on 2 x P100.
7. **Chunked prefill overlapped with expert streaming.**
   Change: already present (the ring and copy stream, `include/strata/prefill/prefill.hpp:6-8`; the P40 profile shows copy waits at 0.2% of the
   prompt, `bench/results/2026-10-12-p40-sm61-kernels/README.md:47-48`). So no new work is expected here for a card that holds its experts;
   re-check on the P100 profile from option 1 before dropping it.
8. **Dequantise dense weights once, not per chunk.**
   Change: the P40 profile shows every dense weight dequantised once per chunk, about 0.5 s of a 559-token prompt in `dequant_kernel<Q6_K>`
   (`bench/results/2026-10-12-p40-sm61-kernels/README.md:50-52`). Keeping an FP16 copy resident would remove it for short prompts but costs VRAM on a 16 GB
   card. Files: `src/prefill/gemm.cu`, `src/prefill/prefill.cpp`. Measure: prompt ms for 560 tokens against the VRAM the copy takes.

## Hypotheses (untested)

- H1: the P100's double-rate FP16 makes dequantise-to-FP16 + GEMM beat MMQ on sm_60 at chunk sizes of about 2K and above.
- H2: the P40 profile split (dense GEMMs 36%, experts 28%) is roughly the same on a P100; GP100 has more FP64/FP16 throughput and HBM2 bandwidth
  than a P40, so the split may shift toward the experts.
- H3: `STRATA_DENSE_F32=1` helps on P100 for T >= 512, as it did on a P40, but the cuBLAS kernel choice on GP100 may differ.
- H4: on cesarops4om, whole-prompt CPU prefill would be slower than the GPU path because its Gold 6148 lacks VNNI and the code takes the AVX2
  tier. On t440 (VNNI) the CPU tier is faster than on cesarops4om, but still unlikely to beat a GPU prefill for a whole prompt.
- H5: a prompt could be read on the CPU while a GPU decodes another request only with batch slots; the repo's batching note
  (`docs/BATCHING.md:23-28`) describes slots sharing the decode loop, not a separate CPU prefill worker, so this needs new code.
- H6: the Tiel prompt rates measured on the T10 (869-877 tok/s) and 2060 (690-694 tok/s) indicate what an sm_75 card does with tensor cores; a P100
  without them is expected to land well below that. No P100 prompt rate for Tiel exists to compare.

## Recommended first experiment

Question: where does a P100's prompt time go, and does the existing `STRATA_DENSE_F32` route or `STRATA_PREFILL_MMQ=0` move it?

Setup on t440 (2 x P100, VNNI CPU). The engine must be built for sm_60 first (CUDA 12.x, not 13: `CMakeLists.txt:181-182`):

    cmake -S . -B build60 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=60 -DSTRATA_EXPERIMENTAL_SM60=ON
    cmake --build build60 --target strata gemm_f32_parity -j

Runs, one P100 only (`CUDA_VISIBLE_DEVICES=0`), the same model and the same 4K-token prompt each time, three runs per arm, interleaved
(base, A, B, base, A, B, ...); `STRATA_PREFILL_CPU_SHARE=0` throughout so the CPU does not mix in:

    export STRATA_PREFILL_TIMING=1 STRATA_PREFILL_CPU_SHARE=0
    # base:  (nothing else)
    # arm A: STRATA_DENSE_F32=1     (only after: build60/gemm_f32_parity --quick passes on the P100)
    # arm B: STRATA_PREFILL_MMQ=0

Record prompt tokens per second from the server (`prompt_per_second`, `serve/server.py:4928`) and the per-phase GPU table that
`STRATA_PREFILL_TIMING` prints (`src/prefill/prefill.cpp:2000`).

Confirms: arm A or B is faster than base by more than the spread between the three base runs, and the phase table shows the change in the
phase it targets (GEMM for A, gate/up and down for B). That would justify options 1 or 3 as the first code to write.
Rejects: both arms inside the base spread, or `gemm_f32_parity` reporting a mismatch on the P100; then option 2 (a dp4a-free expert kernel)
is the next candidate, and the phase table says whether GEMM or experts is the larger share.
