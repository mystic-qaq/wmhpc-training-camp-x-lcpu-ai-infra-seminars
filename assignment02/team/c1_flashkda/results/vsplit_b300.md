# V-split / BPW1 优化：B300 实测记录（中间阶段）

这是 V-split 阶段的历史记录，保留了早期 `V32/BPW1` 结果。后续加入
persistent state fragment、parallel state-update warps 和 Direct-B layout
后的最终数字见 [`final_optimization_b300.md`](final_optimization_b300.md)。

这是本次挑战后半段的实验记录。目标不是改变 KDA 的数学公式，而是
针对 B300 上 K2 recurrence 的并行度不足，重新划分一个 CTA 负责的 value
维度。

## 1. 实验口径

- GPU：NVIDIA B300 SXM6 AC，compute capability 10.3
- CUDA：13.0；CUTLASS：`5c149f52a436782210263fb2f19b354443a61c6a`
- 输入：`D=K=V=128`，随机 BF16 `q/k/v/g/beta`，`lower_bound=-5`
- 计时：CUDA event，warmup=20，iters=100，repeats=5；覆盖公开
  `flash_kda.fwd` 调用、workspace 分配和 beta 转置
- `V128/BPW2` 是原始映射：一个 sequence-head 一个 CTA，每个 MMA warp
  处理两个 16x16 value block。

由于 GPU 时钟和节点负载会有轻微变化，以下数字用于比较趋势；最终 PR
应在目标服务的真实 batch/sequence 分布上重新校准阈值。

## 2. 改动是什么

状态更新在 value 行上是独立的。一个 `V_TILE` CTA 只负责 state 的
`[v_offset:v_offset+V_TILE, :]`，但仍加载完整的 `D=128` key/query 和
workspace。于是 K2 grid 从：

```text
(N, H)
```

变成：

```text
(N, H * (128 / V_TILE))
```

`V_TILE=32` 时，一个 head 有四个 CTA，`V_TILE=64` 时有两个 CTA。每个
CTA 的 state/output/v TMA tile 和 state gmem 指针都带上 `v_offset`；公开
的 state layout `[N,H,V,K]` 不变，因此调用方无需改 API。

原始和实验映射的线程组织如下：

| 映射 | MMA warps | load/store warps | block threads | 作用 |
|---|---:|---:|---:|---|
| V128/BPW2 | 4 | 2 | 192 | 原始密集映射 |
| V64/BPW1 | 4 | 2 | 192 | 两个 CTA/head，每 warp 一个 value block |
| V32/BPW1 | 2 | 2 | 128 | 四个 CTA/head，每 warp 一个 value block |

`BPW1` 很重要：如果 V-split 后仍让一个 warp 处理两个 block，CTA 数增加
了，但一个 CTA 内的独立工作没有增加到合适的调度粒度，实测反而更慢。

## 3. 固定长度结果

下面单位都是 ms。表中 `auto` 是最终的 SM10.x heuristic，其他行是通过
环境变量强制的映射。

### K3 TP8 目标形状：H=12, T=8192

| 映射 | no state | BF16 state | FP32 state |
|---|---:|---:|---:|
| V128/BPW2 | 0.786598 | 0.782391 | 0.755870 |
| V64/BPW1 | 0.628807 | 0.640553 | 0.655278 |
| V32/BPW1 | 0.577070 | 0.577488 | 0.596266 |
| auto | 0.577255 | 0.577477 | 0.596298 |

自动映射相对原始映射的时间下降为：

- no state：约 **26.6%**；
- BF16 state：约 **26.2%**；
- FP32 state：约 **21.1%**。

这不是“tcgen05 指令更快”的结果：这些 kernel 仍使用 HMMA。收益来自
把一个长 state chain 的 value 行拆到更多 CTA，让 B300 有更多独立的
CTA 可以调度。

### K3 完整头数：H=96, T=8192

| 映射 | no state | BF16 state | FP32 state |
|---|---:|---:|---:|
| V128/BPW2 | 1.028168 | 1.027978 | 0.997724 |
| V64/BPW1 | 1.008056 | 1.020766 | 1.036841 |
| V32/BPW1 | 1.129163 | 1.133833 | 1.133432 |
| auto | 1.008306 | 1.020715 | 0.997545 |

H=96 已有更多 head，继续拆成 V32 会产生太多重复的 q/k/workspace
加载和短 CTA。BF16/no-state 时 V64 只有约 0.7%/1.9% 的收益；FP32
state 时保持 V128，避免约 3.9% 的回归。这是必须保留 heuristic 的原因。

### 短序列 crossover

对 `H=12` 的 fixed case 还扫描了短序列。`T=16` 时 V64 更稳妥；从
`T=32` 开始，V32 已经不慢，并在 FP32 state、`T=37/64/127` 上明显
更好。例如 no-state 的代表值为：

| T | V128/BPW2 | V64/BPW1 | V32/BPW1 | 自动选择 |
|---:|---:|---:|---:|---|
| 16 | 0.020765 ms | 0.020101 ms | 0.020123 ms | V64 |
| 32 | 0.020210 ms | 0.020298 ms | 0.020285 ms | V32 |
| 37 | 0.021610 ms | 0.020363 ms | 0.020325 ms | V32 |
| 64 | 0.020785 ms | 0.020171 ms | 0.020229 ms | V64/V32 接近 |
| 127 | 0.032906 ms | 0.026744 ms | 0.024699 ms | V32 |

绝对差异很小，所以最终策略只把 crossover 用作启发式，而不把短序列
的测量差异当作普适硬件定律。

## 4. packed/varlen 回归检查

对 `N=8` 个长度为 1024 的 sequence、`H=12`：

| 映射 | no state | BF16 state | FP32 state |
|---|---:|---:|---:|
| V128/BPW2 | 0.150273 | 0.152175 | 0.154340 |
| V64/BPW1 | 0.147991 | 0.149909 | 0.157187 |
| V32/BPW1 | 0.163219 | 0.164852 | 0.168553 |
| auto | 0.148063 | 0.149811 | 0.154252 |

因此最终 dispatch 对 `N>=8, H=5..16, FP32 state` 保持 V128；该区域中
若无这个保护，V64 会比 V128 慢约 1.8%，V32 会慢约 9.2%。同一批次的
BF16/no-state 则使用 V64，仍有约 1.5% 的小收益。

## 5. NCU 交叉证据

显式 mapping 的 Nsight Compute profile 也符合“并行度受限”的解释：

| profile | block | grid | shared memory/block | registers (allocated) | HMMA active | active warps | DRAM |
|---|---:|---|---:|---:|---:|---:|---:|
| H=12,V32/BPW1 | 128 | `(1,48,1)` | 56.320 KiB | 58 (64) | 10.64% | 6.25% | 1.74% |
| H=96,V64/BPW1 | 192 | `(1,192,1)` | 69.632 KiB | 58 (64) | 23.03% | 12.60% | 10.92% |

这两份 profile 分别保存在 `results/ncu_v32_bpw1_h12_summary.txt` 和
`results/ncu_v64_bpw1_h96_summary.txt`。V-split 提高了 grid 中的独立
CTA 数，但 HMMA/DRAM 仍没有接近峰值，说明收益主要来自隐藏 recurrence
latency，而不是把同一个 kernel 变成峰值 GEMM。

## 6. 最终自动策略

只有 `deviceProp.major == 10` 时启用；其他架构保持原始映射。环境变量
显式设置后优先于自动策略：

| 条件 | 自动选择 |
|---|---|
| 单 sequence，`H=5..32,T>=32` | V32/BPW1 |
| 单 sequence，`H<=4,T>=64` | V32/BPW1 |
| 单 sequence，其他 `H<=64` | V64/BPW1 |
| 单 sequence，`H=65..96` 且非 FP32 state | V64/BPW1 |
| 单 sequence，其他情况 | V128/BPW2 |
| packed/varlen，`N>=8,H=5..16` 且 FP32 state | V128/BPW2 |
| packed/varlen，`H<=4` / `H<=16` / 其他 | V32 / V64 / V128 |

短序列的 crossover 只带来几十微秒以下的绝对差异，表格是基于 B300
微基准的保守近似，不应被理解为所有 GPU 和所有服务的全局最优解。

## 7. 正确性和边界

V-split 没有改变 K1，也没有改变 state 的公开 layout；每个 split CTA
只写自己的 value 行。最终版本在 fixed/varlen、batched、tail、四种 state
I/O、BF16/FP32 state 和 long cases 上复跑原有 correctness matrix，结果为：

```text
580 passed in 684.91s (0:11:24)
```

关键实现文件：

- `FlashKDA/csrc/smxx/fwd_kernel2.cuh`：V-slice layout、TMA offset、CTA
  warp 映射和 recurrence；
- `FlashKDA/csrc/smxx/fwd_launch.cu`：模板实例化、grid 和线程数；
- `FlashKDA/csrc/flash_kda.cpp`：SM10.x heuristic 和显式 override；
- `experiments/bench_variants.py`：同一输入下的 mapping 对照计时。

## 8. 失败尝试

在保持 correctness 的前提下还测试了：`kInputStages=2`、
`kOutputStages=1`、`PREFETCH=2`、V16、V128/BPW1、V32/V64 的 BPW2。
它们分别受到 TMA 隐藏不足、输出流水线不足、寄存器/调度开销、CTA
过小或单 CTA 内工作过密的影响，均不如最终的：

```text
kInputStages=3, kOutputStages=2, PREFETCH=1
V128/BPW2, V64/BPW1, V32/BPW1
```

这组结果支持一个更窄但更可靠的结论：在现有 CHUNK=16 recurrence
算法中，最有价值的 SM10.x 优化面是 CTA-level parallelism，而不是
继续盲目替换 MMA opcode。
