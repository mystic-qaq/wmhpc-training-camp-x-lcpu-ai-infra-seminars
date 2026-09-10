# FlashKDA B300 benchmark

设备：NVIDIA B300 SXM6 AC，CC 10.3，CUDA 13.0。输入为 `T=8192,D=128`；`H=96` 是 K3 目标配置，`H=64` 是额外对照。每一项是 `warmup=30,iters=200,repeats=5` 的 mean，单位 ms。以下数据来自最终 `+final` 构建，运行时没有设置 mapping override，因此使用代码内的 SM10.x heuristic。

命令：

```bash
cd /home/lcpu/13168002/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/FlashKDA
export FLASH_KDA_CUTLASS_DIR=/home/lcpu/13168002/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/FlashKDA/cutlass
export FLASH_KDA_CUDA_ARCHS=103a
unset FLASH_KDA_V_TILE FLASH_KDA_BLOCKS_PER_WARP FLASH_KDA_STATE_UPDATE_WARPS FLASH_KDA_PARALLEL_STATE_UPDATE
python benchmarks/bench_fwd.py --warmup 30 --iters 200 --repeats 5 --mode all --H 96
python benchmarks/bench_fwd.py --warmup 30 --iters 200 --repeats 5 --mode all --H 64
```

## H=96

| case | FlashKDA BF16 state | FlashKDA no state | FlashKDA FP32 I/O state | FLA `chunk_kda` | FLA gated delta rule |
|---|---:|---:|---:|---:|---:|
| fixed | 1.0037 | 1.0061 | 0.9986 | 2.3654 | 1.3043 |
| mixed varlen `[1300,547,2048,963,271,3063]` | 0.8657 | 0.8585 | 0.8736 | 2.3858 | 1.3065 |
| varlen `1024×8` | 0.7016 | 0.6993 | 0.7182 | 2.3413 | 1.2571 |

相对于 `chunk_kda`，默认 BF16 state 路径约为 `2.30×/2.78×/3.34×`；相对于 gated delta rule 约为 `1.27×/1.52×/1.79×`。不同 state I/O 变体的均值差异很小，不改变结论。

## H=64

| case | FlashKDA BF16 state | FlashKDA no state | FlashKDA FP32 I/O state | FLA `chunk_kda` | FLA gated delta rule |
|---|---:|---:|---:|---:|---:|
| fixed | 0.7138 | 0.7114 | 0.7421 | 1.6465 | 0.9576 |
| mixed varlen `[1300,547,2048,963,271,3063]` | 0.6551 | 0.6529 | 0.6599 | 1.6717 | 0.9436 |
| varlen `1024×8` | 0.4750 | 0.4724 | 0.4859 | 1.5679 | 0.8542 |

这些结果和官方 GB200 表的趋势一致：FlashKDA 的收益随 varlen 形状和 head 数量变化，但在本次 B300 环境仍明显快于 Triton `chunk_kda`。当前数据的 min/max 会受到 GPU 时钟和节点负载影响，因此正文使用 mean，并把 mapping 之间的差异解释为形状相关的实测结果。
