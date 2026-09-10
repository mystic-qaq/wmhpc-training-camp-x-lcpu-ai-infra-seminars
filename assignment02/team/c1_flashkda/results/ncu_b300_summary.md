# B300 Nsight Compute 摘要

原始报告：[`assignment02/FlashKDA/report_b300_fixed.ncu-rep`](../../../FlashKDA/report_b300_fixed.ncu-rep)（约 79 MB）。这是固定长度 `T=8192,H=96,D=128`、`warmup=0,iters=5,repeats=1` 的 full profile；只抓取 `prepare` 和 `recurrence` 两个 kernel。解析命令：

```bash
ncu --import assignment02/FlashKDA/report_b300_fixed.ncu-rep --page raw --csv \
  | python experiments/summarize_ncu.py
```

`% peak` 是相对于该硬件管线峰值的百分比；`active` 是管线 active 周期中的活跃度，`elapsed` 是按整个 kernel elapsed time 归一化。`gpu__time_duration` 的单位是微秒。

| 指标 | `prepare` | `recurrence` |
|---|---:|---:|
| block / grid | `(256)/(512,96,1)` | `(192)/(1,96,1)` |
| kernel duration | 272.928 us | 746.400 us |
| DRAM throughput | 58.926% | 18.734% |
| L1/TEX throughput (active) | 68.814% | 61.179% |
| SM throughput (elapsed) | 70.260% | 20.575% |
| tensor pipe active | 5.728% | 30.321% |
| HMMA pipe active | 5.728% | 30.321% |
| HMMA pipe elapsed | 5.633% | 19.375% |
| TMA pipe active | 7.940% | 6.309% |
| SM active warps | 96.599% | 9.368% |
| TMA load instructions | 245,760 | 3,442,176 |
| TMA store instructions | 2,605,056 | 787,968 |
| registers/thread (allocated) | 32 (32) | 73 (80) |
| shared memory/block | 22.272 KiB | 99.456 KiB |
| theoretical limit: registers | 8 blocks | 4 blocks |
| theoretical limit: shared memory | 9 blocks | 2 blocks |
| theoretical limit: warps | 8 blocks | 10 blocks |
| theoretical limit: barriers | 32 blocks | 7 blocks |

解释：`recurrence` 的 HMMA 管线没有达到峰值，DRAM 也远未饱和；同时只有约 9.37% 的峰值 active warps，且一个 head 的全部 chunk 必须串行更新状态。因此这个形状在 B300 上更像“并行度/状态链/小 tile 受限”，不能简单归类为 DRAM memory-bound，也不是 tensor-core compute-bound 饱和。`prepare` 是 token-parallel，指标明显更高；真正限制端到端时间的是 head-parallel 的 `recurrence`。

原始 profile 中 `smsp__mem_tensor_reads_op_utcmma_matrix_c` 为 0；结合 SASS 中有 `UTMALDG.3D` 但没有 `UTCMMA`，说明这些 TMA 只负责搬运，主乘法仍是 HMMA。
