# 2SM tcgen05 state-update 实验（B300）

这是对“把一个完整的 `128×16×16` state update 交给两个 CTA 的
`UTCHMMA.2CTA`”的验证。它比单 CTA 的 `64×16×16` 实验更公平：不再用
零填充来凑 M 维，而是让两个 peer CTA 各负责 64 行，并由 leader CTA
发起一次 2SM MMA。

## 实验定义

目标计算是：

```text
S[128, 16] = BF16(decay[128] * S[128, 16] + A[128, 16] @ B[16, 16])
```

每个 CTA 使用 128 threads，cluster 为 `2×1×1`。实验完整计入：

1. 两个 CTA 的 shared-memory 输入准备；
2. `UTCHMMA.2CTA`；
3. TMEM 到寄存器的读取；
4. BF16 decay/add epilogue；
5. state 写回。

因此这里比较的是一个 state update 的端到端 primitive，而不是只比较
tensor-core 指令的理论吞吐。

构建命令：

```bash
nvcc -std=c++17 -O3 -arch=sm_103a \
  -I$CUTLASS/include -I$CUTLASS/tools/util/include \
  -I$CUTLASS/examples/common \
  experiments/tcgen05_2sm_state_update.cu \
  -o build/tcgen05_2sm_state_update
```

本目录中的实际 `$CUTLASS` 是题目提供的：

```text
/home/lcpu/13168002/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/FlashKDA/cutlass
```

## 正确性

所有下表实验的 host scalar reference 对拍均为：

```text
bad=0, max_abs=0
```

此外，使用 256 threads/CTA 的尝试得到 `bad=1009`，说明当前 CUTLASS
TMEM copy layout 并不能直接扩展到这个线程数；该配置没有用于性能结论。
尝试把 accumulator 改成 BF16 以减少 TMEM epilogue 代价时，编译在
CUTLASS 的 `CFormat` 检查处失败（`Unknown type for CFormat`），所以不能
把“BF16 accumulator 可能更快”当作实测结果。

## TMEM load 扫描

下面是在 B300 SXM6、CUDA 13.0 上的实测值。命令参数为
`repeats=256`，时间使用 CUDA event，并在 kernel 内循环后除以循环次数。
单位是 ns/个完整 `128×16×16` state update，越小越好。

| TMEM load atom | ns | correctness |
|---|---:|---|
| `16dp256b1x` | **224.65** | `bad=0` |
| `16dp32b8x` | **224.60** | `bad=0` |
| `16dp128b2x` | 240.86 | `bad=0` |
| `16dp64b4x` | 272.48 | `bad=0` |
| `16dp128b1x` | 288.79 | `bad=0` |
| `16dp32b4x` | 256.83 | `bad=0` |
| `16dp32b2x` | 304.58 | `bad=0` |
| `16dp32b1x` | 386.87 | `bad=0` |
| `16dp64b1x` | 420.61 | `bad=0` |

结果显示，较宽的 TMEM load 能明显降低 epilogue 代价，但即便选到最好
的 atom，收益也不足以改变与 HMMA 的比较结论。

## 与 1SM tcgen05 和 HMMA 对照

在同一 B300 节点、`repeats=256` 下，当前最新构建得到：

| 实现 | 计算形状 | ns/完整 state update | 相对 HMMA |
|---|---|---:|---:|
| 1SM tcgen05 | 两次 `64×16×16` | 469.1 | 约 6.28× |
| 2SM tcgen05 | 一次 `128×16×16` | **224.65** | 约 3.01× |
| HMMA | 八个 `16×16×16` block | 约 74.7 | 1× |

三者都执行了 BF16 recurrent epilogue；2SM 的 HMMA 对照来自同一个
`tcgen05_state_update` 程序中的 `hmma_state_update_kernel`。单次运行输出
示例：

```text
device=NVIDIA B300 SXM6 AC cc=10.3 repeats=256
two_sm_state shape=128x16x16 ns_per_full_state_update=224.650 bad=0 max_abs=0

tcgen05_state V=16 shape=64x16x16 ns_per_64xV=234.538 ns_per_full_state_update=469.075 bad=0 max_abs=0
hmma_state   V=16 shape=128x16x16 ns_per_full_state_update=74.756 bad=0 max_abs=0 tcgen05/hmma=6.275x
```

宽 N 的额外实验同样正确，但没有变快：

```text
V=32: 400.80 ns/update
V=64: 884.71 ns/update
```

## SASS 证据

```bash
cuobjdump --dump-sass build/tcgen05_2sm_state_update \
  | rg 'UTCHMMA|UTCMMA|HMMA|WGMMA'
```

最新 2SM binary 中确认出现两处：

```text
UTCHMMA.2CTA
UTCHMMA.2CTA
```

带有反汇编行号的摘录另存于
[sass_tcgen05_2sm_b300.txt](sass_tcgen05_2sm_b300.txt)。

这排除了“编译器没有生成新指令”的可能。负结果的原因是：在当前小
矩阵上，TMEM→register、BF16 epilogue、同步协议等成本大于 2SM MMA
本身带来的好处。

## 结论

2SM 版本比“单 CTA + M 维零填充”的 tcgen05 版本好很多，但仍比当前
HMMA state update 慢约 3 倍。它证明的是更强的结论：即使把当前完整
`128×16` tile 精确匹配到 2SM UMMA，FlashKDA 的这个 workload 仍没有
足够收益接入 production dispatch。若未来通过更大的融合 GEMM 或 affine
scan 提高单次 UMMA 的有效工作量，才值得重新评估 tcgen05。
