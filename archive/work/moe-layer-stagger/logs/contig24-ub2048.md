ggml_cuda_init: found 2 ROCm devices (Total VRAM: 65248 MiB):
  Device 0: AMD Radeon AI PRO R9700, gfx1201 (0x1201), VMM: yes, Wave Size: 32, VRAM: 32624 MiB
  Device 1: AMD Radeon AI PRO R9700, gfx1201 (0x1201), VMM: yes, Wave Size: 32, VRAM: 32624 MiB
| model                          |       size |     params | backend    | ngl |  n_cpu_moe | threads | n_batch | n_ubatch |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | ------: | ------: | -------: | -----: | --: | --------------: | -------------------: |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw |  70.62 GiB |   176.94 B | ROCm       |  99 |         24 |       8 |    4096 |     2048 | tensor |   1 |          pp2048 |       1168.38 ± 6.64 |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw |  70.62 GiB |   176.94 B | ROCm       |  99 |         24 |       8 |    4096 |     2048 | tensor |   1 |          pp4096 |      1164.08 ± 10.47 |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw |  70.62 GiB |   176.94 B | ROCm       |  99 |         24 |       8 |    4096 |     2048 | tensor |   1 |          pp8192 |       1174.27 ± 7.91 |

build: 7f15fb91d (11583)
