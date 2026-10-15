ggml_cuda_init: found 2 ROCm devices (Total VRAM: 65248 MiB):
  Device 0: AMD Radeon AI PRO R9700, gfx1201 (0x1201), VMM: yes, Wave Size: 32, VRAM: 32624 MiB
  Device 1: AMD Radeon AI PRO R9700, gfx1201 (0x1201), VMM: yes, Wave Size: 32, VRAM: 32624 MiB
| model                          |       size |     params | backend    | ngl |  n_cpu_moe | threads | n_batch | n_ubatch |     sm |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | ------: | ------: | -------: | -----: | --: | --------------: | -------------------: |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw |  70.62 GiB |   176.94 B | ROCm       |  99 |          0 |       8 |    4096 |     2048 | tensor |   1 |           pp512 |       1446.30 ± 8.95 |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw |  70.62 GiB |   176.94 B | ROCm       |  99 |          0 |       8 |    4096 |     2048 | tensor |   1 |          pp2048 |       1757.23 ± 9.42 |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw |  70.62 GiB |   176.94 B | ROCm       |  99 |          0 |       8 |    4096 |     2048 | tensor |   1 |          pp4096 |       1796.68 ± 1.60 |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw |  70.62 GiB |   176.94 B | ROCm       |  99 |         24 |       8 |    4096 |     2048 | tensor |   1 |           pp512 |        296.36 ± 6.24 |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw |  70.62 GiB |   176.94 B | ROCm       |  99 |         24 |       8 |    4096 |     2048 | tensor |   1 |          pp2048 |       1207.14 ± 0.66 |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw |  70.62 GiB |   176.94 B | ROCm       |  99 |         24 |       8 |    4096 |     2048 | tensor |   1 |          pp4096 |      1196.15 ± 14.68 |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw |  70.62 GiB |   176.94 B | ROCm       |  99 |         48 |       8 |    4096 |     2048 | tensor |   1 |           pp512 |        160.56 ± 3.65 |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw |  70.62 GiB |   176.94 B | ROCm       |  99 |         48 |       8 |    4096 |     2048 | tensor |   1 |          pp2048 |       829.57 ± 10.90 |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw |  70.62 GiB |   176.94 B | ROCm       |  99 |         48 |       8 |    4096 |     2048 | tensor |   1 |          pp4096 |       812.45 ± 20.25 |

build: 7f15fb91d (11583)
