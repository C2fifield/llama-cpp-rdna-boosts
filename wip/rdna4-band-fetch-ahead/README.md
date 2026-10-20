# wip/rdna4-band-fetch-ahead — RDNA4 decode/verify band: fetch q8_0 K/V ahead into registers (bit-identical)

One `git am` file (one commit), made on `v12-097cf33d9-r92` (net tree `da55e828…`; applied: tree `43709273…`). It touches `ggml/src/ggml-cuda/fattn-mma-f16.cuh`, `fattn-common.cuh`, `fattn-rdna.cuh` and `template-instances/generate_cu_files.py`, and adds two generated files, `template-instances/fattn-mma-f16-band-fetch-ahead-ncols1_{4,8}-ncols2_8-dkq256-dv256.cu`. CMake picks them up through its `fattn-mma*.cu` glob; re-run CMake once.

- With q8_0 K and V read in place (decode/verify, up to 8 query rows), the band launches a kernel variant. It fetches the raw q8_0 bytes of the next K/V batches into registers ahead of use, converts each batch into shared memory only when it is due, and orders only shared memory at its barriers.
- 64-column block (widths 5..8): two batches in flight within an iteration, Q in registers. 32-column block (widths 1..4): one batch in flight, streaming into the next iteration, Q read from shared memory.
- Every other launch (f16 / bf16 K/V, prefill, other head sizes) runs the kernel without it. `GGML_HIP_FA_BAND_FETCH_AHEAD=0` runs the kernel without it for q8_0 too.
- The variant lives in its own translation units, so the kernels without it compile to the same gfx1201 instructions as on plain r92 (apart from source line numbers in device asserts).

Bit-identical to plain r92 with the variant and with the switch off (`FLASH_ATTN_EXT` dumps: q8_0 474/474, head 256 all K/V types 962/962, long-KV band cases 195/195). gfx1201, head 256, q8_0 K/V, KV 110592: GQA 6 -13 % (widths 1..4) and -14 % (5..8), GQA 8 -13 % / -12 %. f16 / bf16 unchanged. Details and numbers are in the PR description.

*The patch and this note were written by Claude (AI).*
