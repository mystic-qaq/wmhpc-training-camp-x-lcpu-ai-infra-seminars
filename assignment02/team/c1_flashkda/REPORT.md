# C1 FlashKDA：B300 上是否值得为 SM100a 单独做一版？

## 结论先说

这次实验的最终结论分成两层：**不建议发布“只把 HMMA 换成
tcgen05”的 SM100a 专版，但建议把 V-split/BPW1 的 SM10.x 并行度优化
整理成 PR 发布**。这不是因为 B300 不能执行新指令，而是因为 FlashKDA
的主要瓶颈不是“SM80 MMA 的峰值不够高”，而是 K2 recurrence 中一个
长状态链可同时运行的独立工作太少：

1. 最终团队构建在 B300 上完整测试得到 **580 passed**，并覆盖 fixed、varlen、batched、tail、长序列以及 BF16/FP32 state I/O。
2. 在 K3 目标形状 `T=8192,H=96,D=128` 上，最终 FlashKDA BF16-state 路径为 **1.0037 ms**，比 Triton `chunk_kda` 的 **2.3654 ms** 快约 **2.36×**。
3. SASS 反汇编确认：最终扩展使用 HMMA.16816.F32.BF16，没有 UTCHMMA/UTCMMA/TCGEN05。也就是说，它在 B300 上仍走 SM80 风格的 mma.sync，但 TMA 搬运仍然使用了 Blackwell 可以执行的路径。
4. “只换指令/执行协议”的 tcgen05 microbenchmark 在有用工作量归一化后约 **慢 1.85×**；进一步把完整 `128×16` tile 精确映射到两个 CTA 的 `UTCHMMA.2CTA` 后，仍比 HMMA state update 慢约 **3.0×**，且还需要 TMEM、mbarrier 和新的布局/epilogue，因此它本身不值得发布。
5. 随后真正改变并行划分：把 state/value 行切成 `V_TILE=16/32/64`，让每个 MMA warp 只负责一个 16×16 block，并把独立的 state update warp 化、与输出重叠。K3 TP8 的每卡形状 `T=8192,H=12` 上，最终 V16/BPW1 使端到端 GPU 时间下降约 **44.7%–46.2%**，且通过完整 correctness matrix。
6. 这个正收益路径仍然使用 HMMA，收益来自更多 CTA 暴露 recurrence 的独立 value 行。因此建议以 **SM10.x 的架构/形状 heuristic + 原始 fallback** 形式提交 PR；H=96 FP32 state 等不适合 split 的形状保持原始映射。

这里的“tcgen05-only 不发布”是一个有实验依据的结论，不是说 tcgen05
永远没有价值。它在与硬件匹配的大矩阵、较高并行度的 GEMM 中很可能有
优势；只是当前 FlashKDA 的 16×16 小矩阵、长状态链和低并行度没有把这个
优势暴露出来。相反，V-split 证明了在不改变 KDA 数学结果、也不依赖新
opcode 的情况下，仍有值得发布的 SM10.x 正收益。

---
## 0. 给初学者的整体图景

FlashKDA 计算的是一种线性注意力。可以先把它看成下面这条流水线：

~~~text
q/k/v/g/beta
      │
      ▼
K1 prepare：按“序列块 × head”并行，准备每个 16-token chunk 的中间量
      │  workspace（中间结果）
      ▼
K2 recurrence：每个 head 沿 token 方向逐 chunk 更新 D×D state
      │
      ├── 输出 out
      └── 最终 state
~~~

几个术语的含义：

| 术语 | 这里的含义 |
|---|---|
| token | 序列中的一个位置；本报告用 T 表示序列长度 |
| head | 注意力头；K3 的目标配置是 H=96 |
| D | 每个 head 的向量维度，这里是 128 |
| chunk/tile | 连续的一小段 token；FlashKDA 固定 CHUNK=16 |
| CUDA thread block / CTA | GPU 调度的一个工作组；可以理解成一个可共享 shared memory 的小团队 |
| warp | 32 个线程组成的执行小组 |
| MMA | 矩阵乘加指令（Matrix Multiply-Accumulate） |
| TMA | GPU 的异步张量搬运单元；它负责搬数据，不等于矩阵乘法 |
| TMEM | Blackwell tcgen05 的专用矩阵累加器存储 |
| SASS | GPU 最终执行的机器指令；比 C++/CUDA 源码更接近事实 |
| NCU | Nsight Compute，用来统计 kernel 的时间、带宽、占用率和指令管线 |

tile 形状就是一次矩阵乘法实际计算多少行、多少列、多少个 K 维元素。例如 16×16×16 表示 A[16,16] @ B[16,16]。

---

## 1. 实验对象、环境和可复现方法

### 1.1 代码和版本

本题团队目录是 assignment02/team/c1_flashkda/。官方 FlashKDA 的完整 checkout 和 CUTLASS 位于：

~~~text
assignment02/FlashKDA/
assignment02/FlashKDA/cutlass/
~~~

版本固定为：

| 项目 | 实测版本 |
|---|---|
| GPU | NVIDIA B300 SXM6 AC |
| Compute Capability | 10.3（SM103） |
| CUDA | 13.0 |
| PyTorch | 2.13.0+cu130 |
| Triton | 3.7.1 |
| flash-linear-attention | 0.5.2 |
| FlashKDA | 1ce47ea |
| CUTLASS | 5c149f52a436782210263fb2f19b354443a61c6a |

题目给的团队 FlashKDA 快照没有把 CUTLASS 子模块放进快照内部，所以我在 setup.py 中增加了 FLASH_KDA_CUTLASS_DIR。这只是构建路径适配，不改变 kernel 算法：

- [setup.py](FlashKDA/setup.py) 支持从环境变量读取外部 CUTLASS；
- [README.md](FlashKDA/README.md) 记录了构建命令；
- 外部 checkout 使用题目要求的 CUTLASS commit。

在 B300 节点上构建团队副本的命令为：

~~~bash
cd /home/lcpu/13168002/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/FlashKDA
export FLASH_KDA_CUTLASS_DIR=/home/lcpu/13168002/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/FlashKDA/cutlass
export FLASH_KDA_CUDA_ARCHS=103a
pip install -e . --no-build-isolation
~~~

FLASH_KDA_CUDA_ARCHS=103a 的作用是显式告诉 NVCC 生成 B300 的机器代码，避免只生成别的架构的代码。官方 checkout 也用同一个外部 CUTLASS 完成了编译。

### 1.2 正确性复现

官方完整测试覆盖：

- fixed length 和 variable length；
- batch 大于 1；
- H=1,4,32,96；
- 多个短序列、长序列和不整除 16 的长度；
- initial state / final state 的四种组合；
- BF16 和 FP32 的外部 state I/O。

最终 `+final` 构建的复验结果：

~~~text
580 passed in 704.18s (0:11:44)
~~~

对应命令（在团队副本的 `FlashKDA/tests/` 目录下执行）为：

~~~bash
export FLASH_KDA_DIST_GPU=1
cd /home/lcpu/13168002/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/FlashKDA/tests
pytest test_fwd_full.py -x -v -n 4
~~~

这确认了构建路径适配没有只在 smoke test 中成立，团队副本本身也通过了完整 correctness 矩阵。

官方测试的 kernel 对照实现见 [tests/test_fwd_full.py](FlashKDA/tests/test_fwd_full.py)，测试使用的逐步 reference 见 [tests/torch_ref.py](FlashKDA/tests/torch_ref.py)。团队副本在重新指向外部 CUTLASS 后先做了 fixed/varlen smoke test（2 passed in 46.13s），随后完成了上面的全量测试。

另外，挑战实验使用题目附带的 [`fla_kda_ref`](fla_kda_ref/) 做了独立小规模对拍：`naive_recurrent_kda` 和 `naive_chunk_kda(chunk_size=16)` 在 T=16/64/256 上均与生产 kernel 接近，output relative RMS 为 0.53%–0.61%，state relative RMS 为 0.43%–0.50%。详细输出见 [results/reference_compare_b300.md](results/reference_compare_b300.md)。

这里要区分两种“正确”：

- torch_ref.py 尽量复现生产 kernel 的 BF16、FP16 inverse、近似 sigmoid 和 FTZ 行为，所以可以做 bitwise exact 检查；
- [state_precision.py](experiments/state_precision.py) 还使用 FLA 的 recurrent 实现保留 FP64 state，检查生产实现相对于更高精度数学路径的误差。这一项不是要求 bitwise 相同，而是观察误差是否随长度或 gate 分布失控。

### 1.3 benchmark 口径

命令与官方设置一致：warmup=30, iters=200, repeats=5。

~~~bash
cd /home/lcpu/13168002/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/team/c1_flashkda/FlashKDA
export FLASH_KDA_CUTLASS_DIR=/home/lcpu/13168002/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/FlashKDA/cutlass
export FLASH_KDA_CUDA_ARCHS=103a
unset FLASH_KDA_V_TILE FLASH_KDA_BLOCKS_PER_WARP FLASH_KDA_STATE_UPDATE_WARPS FLASH_KDA_PARALLEL_STATE_UPDATE
python benchmarks/bench_fwd.py --warmup 30 --iters 200 --repeats 5 --mode all --H 96
python benchmarks/bench_fwd.py --warmup 30 --iters 200 --repeats 5 --mode all --H 64
~~~

完整记录在 [results/benchmark_b300.md](results/benchmark_b300.md)。每个数字是 CUDA event 的 GPU 时间均值，不是 Python wall-clock 时间。

---

## 2. B300 上的实际性能和 SM80 MMA 证据

### 2.1 端到端性能

H=96, D=128 是 K3 目标形状。单位是 ms，越小越好：

| case | FlashKDA BF16 state | FlashKDA no state | FlashKDA FP32 I/O state | FLA chunk_kda | FLA gated delta rule |
|---|---:|---:|---:|---:|---:|
| fixed 8192 | 1.0037 | 1.0061 | 0.9986 | 2.3654 | 1.3043 |
| mixed varlen | 0.8657 | 0.8585 | 0.8736 | 2.3858 | 1.3065 |
| varlen 1024×8 | 0.7016 | 0.6993 | 0.7182 | 2.3413 | 1.2571 |

相对 `chunk_kda`，FlashKDA BF16-state 路径分别约为 **2.36×、2.75×、3.34×**。H=64 的结果也保留在 benchmark 文档中；趋势与官方 GB200 表一致。

这里的 state I/O 差别很小。原因是“FP32 I/O”不代表内部一直使用 FP32 state：生产 kernel 仍然把 FP32 输入转换成片上 BF16，最后再把片上 BF16 转回 FP32。

### 2.2 SASS 反汇编

使用：

~~~bash
cuobjdump --dump-sass flash_kda_C.cpython-312-x86_64-linux-gnu.so
~~~

官方原始扩展的统计仍保存在 [results/sass_b300.txt](results/sass_b300.txt)；
最终团队扩展的统计见 [results/sass_final_b300.txt](results/sass_final_b300.txt)：

| 指令/字符串 | 最终团队扩展中的数量 |
|---|---:|
| HMMA.16816.F32.BF16 | 23696 |
| HMMA.16816.F16 | 24 |
| UTMALDG.1D / 2D / 3D | 324 / 324 / 22298 |
| UTCHMMA / UTCMMA / TCGEN05 | 0 |
| WGMMA | 0 |

因此结论很具体：

- TMA 已经存在，说明数据搬运路径并没有停在老架构；
- 乘法主路径是 HMMA，即 SM80 风格的 warp-level mma.sync；
- 官方扩展里没有 tcgen05，并不是 B300 或 CUDA 13 不能生成 tcgen05。后面的独立 microbenchmark 成功生成了 UTCHMMA，说明这是官方代码选择，而非工具链故障。

最终团队构建的完整统计见
[results/sass_final_b300.txt](results/sass_final_b300.txt)。它包含 V128、
V64、V32、V16 以及 state-update warp 的 HMMA 模板实例，但仍没有
UTCHMMA/UTCMMA/TCGEN05/WGMMA；因此后续的正收益可以明确归因于
CTA-level parallelism 和 warp overlap，而不是误把新指令的收益算进来。

---

## 3. 讨论点 1：为什么是 CHUNK=16

### 3.1 先看实际 gate 公式

FlashKDA 的 lower-bound gate 近似为：

~~~text
g_act = lower_bound × sigmoid(exp(A_log) × (g + dt_bias))
~~~

题目和 kernel 使用 lower_bound=-5，所以每个维度的 activated gate 大致落在 [-5, 0]。kernel 在 [fwd_kernel1.cuh](FlashKDA/csrc/smxx/fwd_kernel1.cuh) 中先做 chunk 内 cumsum，再调用 ex2.approx.ftz.f32。C++ 侧把自然指数的输入乘以 log2(e)，见 [flash_kda.cpp](FlashKDA/csrc/flash_kda.cpp#L128)。

“FTZ”是 flush-to-zero：太小的浮点数不保留为 subnormal，而直接变成 0。它对吞吐有帮助，但会让非常小的衰减项消失。

### 3.2 数值范围：CHUNK=32 已经危险

取每个 token 都接近最坏的 -5，chunk 末端的累计值为 -5C：

| C | cumsum | exp(cumsum) | 相对 BF16 最小 normal | exp(-cumsum) | 相对 FP32 最大值 |
|---:|---:|---:|---:|---:|---:|
| 16 | -80 | 1.8049e-35 | 1.54e3×，仍在范围内 | 5.5406e34 | 1.63e-4× |
| 32 | -160 | 3.2575e-70 | 3.61e31 倍小于最小 normal | 3.0698e69 | 9.02e30× |
| 64 | -320 | 1.0611e-139 | 1.11e101 倍小于最小 normal | 9.4240e138 | 2.77e100× |

范围常数是：

~~~text
BF16 最小 normal = 2^-126 = 1.17549435e-38
BF16 最大值      ≈ 3.38953139e38
FP32 最大值      ≈ 3.40282347e38
~~~

因此有两个独立问题：

1. **正向衰减** exp(cumsum)：C=32 的最坏值已经远低于 BF16 最小 normal，并且 kernel 使用 FTZ；
2. **恢复 K 的倒数** exp(-cumsum)：C=32 的最坏值已经超过 FP32 最大值，甚至在转换成 BF16 之前就可能在 ex2 输出阶段变成 inf。

这不是说真实模型的每个 token 都会达到 -5，而是说没有 rescale 时，C=32 没有 C=16 那样的数值安全余量。C=64 更不用说。纸面计算脚本和输出分别是 [numeric_analysis.py](experiments/numeric_analysis.py) 与 [numeric_analysis.md](results/numeric_analysis.md)。

### 3.3 Neumann 求逆的计算代价

chunk 内会出现一个严格下三角矩阵 L。因为严格下三角矩阵满足 L^C=0，有：

~~~text
(I - L)^(-1) = I + L + L^2 + ... + L^(C-1)
~~~

这就是 Neumann series。实现没有逐项做 C 次乘法，而是利用二进制幂：先得到 L^2, L^4, ...，再把这些幂逐步合并到 inverse 中。对于 C=16，源码 [utils.cuh](FlashKDA/csrc/smxx/utils.cuh#L256) 中正好有：

~~~text
L^2、INV += INV×L^2
L^4、INV += INV×L^4
L^8、INV += INV×L^8
~~~

纸面 dense GEMM 估计如下。一次 C×C 矩阵乘的 FLOPs 取 2C^3，即一次乘加算 2 个浮点操作；这是便于比较的估计，不是硬件实际周期：

| C | series 项数 | 需要的幂 | 乘法次数 | 每次乘法 FLOPs | 求逆段 FLOPs | 单个 C×C BF16 矩阵 |
|---:|---:|---|---:|---:|---:|---:|
| 16 | 16 | L^2,L^4,L^8 | 6 | 8,192 | 49,152 | 512 B |
| 32 | 32 | L^2,L^4,L^8,L^16 | 8 | 65,536 | 524,288 | 2,048 B |
| 64 | 64 | L^2,L^4,L^8,L^16,L^32 | 10 | 524,288 | 5,242,880 | 8,192 B |

相对 C=16，求逆段的 dense-FLOP 比例是：

~~~text
1 : 10.67 : 106.67
~~~

这里还没有算更大的 register fragment、同步、布局转换和 tile 拆分，所以大 chunk 的实际代价只会更高。按 D=128 计算，单 tile 的工作区也大约从 13.5 KiB 增至 28.5 KiB 和 64.5 KiB。

### 3.4 SM80 MMA 的形状原因

SM80 的基本 BF16 atom 是 m16n8k16。一个逻辑 16×16×16 GEMM 需要沿 N 方向放两个 atom：

| C | 一次 C×C GEMM 的逻辑 16×16 tile 数 | HMMA atom 数 | 求逆段 HMMA atom 数（估算） |
|---:|---:|---:|---:|
| 16 | 1 | 2 | 12 |
| 32 | 8 | 16 | 128 |
| 64 | 64 | 128 | 1,280 |

所以三个理由共同作用：

- **数值**：C=16 的 exp(-80) 和 exp(+80) 都还留有余量；C=32 在最坏 gate 下先出现下溢/倒数溢出；
- **算法**：16 阶 inverse 能用少量、规整的 16×16 产品完成；
- **硬件**：16 阶正好匹配 SM80 m16n8k16，不需要为一个小矩阵填充很多无用行。

如果只看“求逆要多几个幂”，C=32 似乎不算多；但把数值范围、矩阵面积、工作区和布局成本合在一起后，C=16 就很合理。

---

## 4. 讨论点 2：tcgen05 tile 是否匹配 CHUNK=16

### 4.1 纸面形状对比

FlashKDA 的关键小乘法是：

~~~text
16 × 16 × 16，BF16 输入，FP16 或 FP32 累加/输出
~~~

在 CUTLASS 5c149f5 的 SM100 路径中，我用的最小单 CTA BF16 UMMA 形状是：

~~~text
64 × 16 × 16
~~~

N 和 K 可以对上，但 M 方向从 16 变成 64。若不改变 CHUNK=16，只能把 A 的后 48 行补零，同时让 tcgen05 计算四倍行数；真正有用的只有前 16 行。这已经不是“把一个 opcode 的名字替换掉”了：

- tcgen05 的累加器在 TMEM，不在普通寄存器 fragment；
- MMA 是异步完成的，需要 mbarrier/commit/wait；
- TMEM load 的布局和每个 warp 能读取的 lane 范围有额外限制；
- 输出还要从 TMEM 读回，再转换成 FlashKDA 现有 shared-memory/寄存器布局。

### 4.2 CUTLASS layout 实验

我用 [layout_dump.cu](experiments/layout_dump.cu) 打印了 64×16 FP32 accumulator 与 TMEM load layout：

~~~text
64x16 accumulator:
  tmem layout=(((_16,_4),_16),_1,_1):(((_65536,_2097152),_1),_0,_0)

SM100_TMEM_LOAD_32dp32b1x:
  upcast=(_1,_32):(_1,_65536)

SM100_TMEM_LOAD_16dp32b1x:
  upcast=(_2,_16):(_1,_65536)
~~~

在这个 fragment 上，16dp32b1x 可以匹配并正确运行，而默认的 32dp32b1x 不匹配。记录见 [results/layout_b300.txt](results/layout_b300.txt)。这说明迁移到 tcgen05 至少要重新设计 layout 和 epilogue；不是只改 gemm() 调用就结束。

### 4.3 B300 microbenchmark

挑战代码是 [tcgen05_vs_mma.cu](experiments/tcgen05_vs_mma.cu)。它做的是一个可单独验证的最小实验：

1. 对同一组小整数 BF16 输入，计算逻辑 16×16×16 结果；
2. tcgen05 路径实际执行 64×16×16，后 48 行填充；
3. mma.sync 路径执行两个 m16n8k16 atom；
4. 两边都与 host scalar reference 对拍；
5. 用 5000 次重复把一次指令的时间测稳定，并做两次连续运行。

结果见 [results/tcgen05_vs_mma_b300.txt](results/tcgen05_vs_mma_b300.txt)：

| 路径 | 原始时间 | 换算到一个有用 16×16 结果 |
|---|---:|---:|
| tcgen05 64×16×16 | 85.20–85.21 ns | 21.30 ns |
| mma.sync 16×16×16 | 11.50–11.51 ns | 11.50–11.51 ns |

tcgen05 的原始时间是 mma.sync 的约 7.4×，但其中 4× 来自它无论如何都要算额外的 48 行。按有用工作量除以 4 后，仍为：

~~~text
21.30 / 11.50 ≈ 1.85×
~~~

正确性结果：

~~~text
tcgen05 bad=0
mma_sync bad=0
max(tc-mma)=0
~~~

SASS 也交叉检查到：

~~~text
UTCHMMA = 10
LDTM    = 16
HMMA.16816.F32.BF16 = 58   # 对照 kernel
~~~

这里的挑战选择要明确说明：这是“**只替换指令/执行协议的最小可证伪 microbenchmark**”，不是把一个没有匹配 tile 的 tcgen05 路径塞进生产 kernel。它证明了在 CHUNK=16 的局部矩阵产品上，tcgen05 没有自动带来收益；也证明了 CUTLASS/NVCC/B300 工具链确实可以生成并执行 tcgen05。

这个实验的正确性是对 host 数学 reference 的 primitive-level 对拍；此外，挑战使用 [reference_compare.py](experiments/reference_compare.py) 对题目提供的 `fla_kda_ref.naive_recurrent_kda` 和 `naive_chunk_kda` 做了 output/state 对拍。整个 FlashKDA baseline 则已经分别通过了 torch_ref 的 580 个完整测试，并通过 [state_precision.py](experiments/state_precision.py) 与 FLA recurrent reference 做了高精度误差检查。由于没有把 tcgen05 版本接入生产 FlashKDA，所以不会把 microbenchmark 的 bad=0 夸大成“tcgen05 版完整 KDA 已通过 580 测试”，也不会虚报一个不存在的端到端加速。

### 4.4 进一步的 2SM tcgen05 验证：消除零填充反例

上面的 1SM 实验还有一个可能的反驳：`M=64` 不是当前完整 state tile 的
形状，tcgen05 为了覆盖 16 行有效结果而计算了 48 行无效结果。为排除
这个反例，我又实现了 [tcgen05_2sm_state_update.cu](experiments/tcgen05_2sm_state_update.cu)。

这个版本使用 `SM100_MMA_F16BF16_2x1SM_SS`，两个 peer CTA 各处理 64 行，
一次 `UTCHMMA.2CTA` 正好计算完整的 `128×16×16` 产品，然后完成 TMEM
读取、BF16 decay/add 和 state 写回。B300 上最好的 TMEM load 组合为：

~~~text
2SM tcgen05: 224.65 ns / full 128×16×16 state update, bad=0
HMMA:         约 74.7 ns / full 128×16×16 state update, bad=0
ratio:        约 3.01×
~~~

这次比较不再有 M 维 padding，因此结论比 1SM 的 1.85× 有用工作归一化
结果更强：即使 tcgen05 的 tile 形状完全匹配，TMEM→register、同步和
recurrent epilogue 仍然让它慢约三倍。`cuobjdump` 同时确认 binary 中确实
有两处 `UTCHMMA.2CTA`。TMEM load 扫描、宽 N 形状、正确性和失败的
BF16 accumulator/256-thread 尝试记录在
[results/tcgen05_2sm_b300.md](results/tcgen05_2sm_b300.md)。

---

## 5. 讨论点 3：chunk 之间的 state 依赖和可挖掘并行度

### 5.1 当前为什么并行度低

K1 是 token/chunk 并行的：在 T=8192、CHUNK=16 时有 512 个 chunk，grid 近似为 (512, 96)。

原始 K2 的 launch grid 是 (N,H)，K3 fixed case 就是约 96 个 CTA；每个
CTA 负责一个 sequence-head，并在自己的循环中处理 512 个 chunk。团队
实现保留了这个 fallback，同时在 SM10.x 上可将 grid 扩为
`(N, H * (D/V_TILE))`。源码在 [fwd_launch.cu](FlashKDA/csrc/smxx/fwd_launch.cu#L203)
设置新 grid，在 [fwd_kernel2.cuh](FlashKDA/csrc/smxx/fwd_kernel2.cuh#L254)
从 `blockIdx.y` 解出 head 和 value slice。

每个 chunk 的状态更新可以抽象为：

~~~text
S_next = decay ⊙ S_now + U
~~~

下一块的 S_now 就是上一块的 S_next，所以一个 head 的 chunk 链不能像 K1 那样随意打乱。输出也依赖当前状态，不能只把最后一个 state 算出来再补输出。

### 5.2 候选方案和反例

| 候选 | 可能收益 | 主要反例/代价 |
|---|---|---|
| 一个 CTA 内合并多个 head | 同一个 CTA 内可交错多个较短状态链，减少低占用造成的空转 | 每个 head 都有 128×128 state；复制 shared memory 会迅速超过容量，时间复用又可能不增加真正的 tensor-core 并行度 |
| persistent kernel | 让常驻 CTA 从全局任务队列领取 (sequence,head)，改善 varlen 长短不均和小 batch 调度 | 任务队列、边界和 state 生命周期更复杂；若只有一个很长的 head，persistent 仍不能打破它内部的依赖 |
| 沿 sequence 切成多个 CTA | 可以让不同 chunk 先并行准备局部结果 | 每个 chunk 的起始 state 未知；必须先做 scan 或全局同步，且每个局部变换可能需要携带一个 D×D 矩阵，内存流量可能抵消收益 |
| 2-CTA UMMA | 能尝试利用 tcgen05 的 CTA group 或把一个大 tile 分给两个 CTA | 当前工作不是一个自然的大 GEMM；两个 CTA 仍要协调同一个 state、barrier 和 epilogue，容易把低并行度变成更多同步 |
| prefix-scan / affine scan | 将每个 chunk 表示成 S_out=A_chunk(S_in)+B_chunk，组合操作是结合的，可在 chunk 维度并行 | 需要保存/搬运大量 A/B，而这里的 state 是 128×128；还要产生每个 chunk 的起始 state 才能得到输出，算法改动大 |
| TP 或 batch 增加 head/sequence 数 | 多卡部署下每卡 head 数、batch 数足够时自然增加 CTA 数 | TP8 时每卡约只有 12 个 KDA head；恰好是小 batch decode/服务场景最容易暴露当前问题的地方 |

最值得继续研究的是“persistent + affine scan”的组合：persistent 解决任务调度，scan 解决 chunk 链依赖。但这已经是算法/执行模型重构，不是换一条 MMA 指令。

完整的 B300 mixed-varlen 排列实验和“不接入全局 task queue”的证据见
[results/persistent_dispatch_b300.md](results/persistent_dispatch_b300.md)。

### 5.3 为什么 H=96 仍然不够填满 B300

原始单个 K2 CTA 使用约 192 个线程、约 99.456 KiB shared memory，NCU
给出的 shared-memory 理论上限是每 SM 2 个 block；它还使用 73 个实际
寄存器/线程、按分配粒度是 80 个。更重要的是，一个 CTA 只有一个 head
的长循环。

V-split 后，V32/BPW1 的 CTA 是 128 threads、约 56.320 KiB shared
memory；同一个 head 变成四个 CTA。V64/BPW1 是 192 threads、约
69.632 KiB，变成两个 CTA。这不是把同一个 state 的写入交给多个 CTA：
每个 CTA 只写不重叠的 value 行，所以不会引入 atomic 或跨 CTA reduction。
代价是 q/k/workspace 会随 slice 重复加载，这也是 H=96、FP32 state 等
形状不应无条件 split 的原因。

因此“grid 有 96 个 CTA”不能直接等同于“GPU 有很高利用率”；而“grid
扩大到 192/384 个 CTA”也不能单独保证收益，必须同时考虑重复加载、
shared memory、state conversion 和 tail/launch overhead。

### 5.4 实际采用的 V-split 优化

这次把可挖掘的并行度真正接入生产 FlashKDA，而不是只停留在候选方案：

- q/k/workspace 仍按完整 D=128 处理，保持 K1 和 recurrence 数学流程不变；
- v、out、state 只处理当前 CTA 的 V slice，并修正 TMA 的 gmem offset；
- V32/V64 使用 BPW1，让一个 warp 对应一个 16×16 output/state block；
- `V128/BPW2` 保留为原始映射，非 SM10.x 和不适合 split 的 shape 回退；
- 外部 state layout 仍为 `[N,H,V,K]`，因此 ABI/API 不变。

完整的最终 benchmark、warp 扫描和 heuristic 表见
[results/final_optimization_b300.md](results/final_optimization_b300.md)。
最终 `+final` 构建的关键 B300 结果是（单位 ms）：

| H / 自动映射 | no state | BF16 state | FP32 state |
|---|---:|---:|---:|
| H=12，V16/BPW1 + 8 update warps | 0.4244 | 0.4251 | 0.4195 |
| H=24，V32/BPW1 + 4 update warps | 0.4732 | 0.4756 | 0.4943 |
| H=64，V64/BPW1 + 8 update warps | 0.7090 | 0.7110 | 0.7391 |
| H=96，V64/BPW1（BF16/no-state）或 V128/BPW2（FP32） | 1.0055 | 1.0025 | 0.9972 |

相对于同一输入强制 `V128/BPW2`，H=12 的时间下降约 44.7%–46.2%，
H=24 下降约 37.6%–42.5%，H=64 下降约 18.5%–24.2%。H=96 已有较多
head，继续切分的收益很小；FP32 state 直接保持 V128，避免回归。当前实现
在最终构建上经过 fixed、varlen、batched、tail 和 long case 的完整测试
矩阵验证，共 **580 passed**。

---

## 6. 讨论点 4：这是 compute-bound 还是 memory-bound？

### 6.1 NCU 采样方法

按官方模板 [benchmarks/ncu.sh](FlashKDA/benchmarks/ncu.sh) 只抓 prepare 和 recurrence 两个 kernel；完整报告保存在 [report_b300_fixed.ncu-rep](../../FlashKDA/report_b300_fixed.ncu-rep)，摘要在 [results/ncu_b300_summary.md](results/ncu_b300_summary.md)。固定形状为 T=8192,H=96,D=128。

关键数据：

| metric | K1 prepare | K2 recurrence |
|---|---:|---:|
| kernel duration | 272.928 us | 746.400 us |
| DRAM throughput (% peak) | 58.926% | 18.734% |
| L1/TEX throughput (% peak, active) | 68.814% | 61.179% |
| SM throughput (% peak, elapsed) | 70.260% | 20.575% |
| HMMA/tensor pipe active | 5.728% | 30.321% |
| TMA pipe active | 7.940% | 6.309% |
| SM active warps | 96.599% | 9.368% |
| registers/thread（allocated） | 32（32） | 73（80） |
| shared memory/block | 22.272 KiB | 99.456 KiB |
| shared-memory theoretical block limit | 9 | 2 |

### 6.2 结论

如果只看 K1，它有相当的 DRAM 和 SM 活跃度，属于 token-parallel 的准备阶段；但端到端主要时间在 K2：

- K2 占这两个主要 kernel 时间的大头（约 746 us）；
- DRAM 只有 18.734%，远没有把显存带宽打满；
- HMMA active 只有 30.321%，也不是 tensor core 峰值饱和；
- active warps 只有 9.368%，并且 shared memory 只允许理论上 2 个 block；
- 一个 head 的 chunk 还必须按顺序更新 state。

所以最准确的说法是：**这个负载不是经典的 DRAM memory-bound，也不是经典的 tensor-core compute-bound；端到端主要是并行度/资源占用/状态依赖造成的 latency-bound。**

这也解释了为什么把 HMMA 换成理论峰值更高的 tcgen05 不一定有用：如果没有足够的独立 tile 同时喂给新管线，峰值数字不会转化为 kernel latency 的下降。

NCU 原始数据还显示 smsp__mem_tensor_reads_op_utcmma_matrix_c=0；结合 SASS 中有 UTMALDG.3D、没有 UTCMMA，再次确认 TMA 只是搬运，主乘法仍是 HMMA。
## 7. 讨论点 5：BF16 state 的精度验证方案和数据

### 7.1 为什么不能只看官方 exact test

官方 torch_ref.py 有意模拟生产路径，包括：

- q/k 的 BF16 输入和 L2 normalization；
- tanh.approx.f32 实现 sigmoid；
- FP16 Neumann inverse；
- BF16 中间量和 BF16 state；
- ex2.approx.ftz.f32。

所以它非常适合验证“实现有没有 bug”，但它和 kernel 共享了许多精度选择。若想衡量 BF16 state 本身相对于更高精度数学计算的误差，还需要独立 reference。

### 7.2 本次方案

[state_precision.py](experiments/state_precision.py) 的做法是：

1. 先生成 BF16 的 q/k/v/g/beta 和 BF16 初始 state；
2. 在调用 FLA recurrent reference 前，用 FP64 计算 gate 和 beta；
3. reference 用 FP64 recurrent state 逐 token 更新；
4. FlashKDA 走真实生产 BF16-state 路径；
5. 对 output 和 final state 都计算 relative RMS：

~~~text
relative RMS = RMS(actual - reference) / RMS(reference)
~~~

测试三种 gate 分布：

- random：普通随机输入；
- near_lower：raw gate=8，activated gate 接近 -5，测试最快衰减；
- near_zero：raw gate=-8，activated gate 接近 0，测试状态长期保留和舍入累积。

测试长度为 T=256,1024,8192，并使用约 0.1 幅度的随机初始 state。

### 7.3 B300 数据

下表是 output relative RMS / final state relative RMS：

| gate case | T=256 | T=1024 | T=8192 |
|---|---:|---:|---:|
| random | 0.5517% / 0.4443% | 0.5271% / 0.4249% | 0.5219% / 0.4940% |
| near_lower | 0.4927% / 0.4714% | 0.4923% / 0.5334% | 0.4644% / 0.4001% |
| near_zero | 0.6993% / 0.6538% | 0.7755% / 0.7028% | 0.7894% / 0.7147% |

最难的长期保留状态 case 在 T=8192 仍约为 **0.715% state relative RMS**，没有随长度爆炸。最大绝对误差随输入尺度变化，但 near-zero T=8192 的 state max abs 为约 1.92e-2，因此不能只报百分比，也应结合真实 state 分布和任务质量判断。

此外，同一组 BF16 可表示的初始 state 分别以 BF16 和 FP32 作为外部 I/O，在 T=1024 的三个 gate case 中均得到：

~~~text
output_exact=True
state_exact=True
state_max_abs=0
~~~

这验证的是 API 行为：FP32 外部 state 进入 kernel 后也会转换成片上 BF16；它**不是**“内部 state 已经是 FP32”的证据。

最终判断：在当前 safe gate、K3 形状和 inference 使用方式下，BF16 state 的误差是可接受且稳定的；但上线前还要用真实模型 gate 分布、长上下文和任务级指标（例如 perplexity/生成质量）复验。精度预算还包括近似 sigmoid、FP16 inverse 和 BF16 中间量，不能把全部误差归咎于 state 存储。

---

## 8. 讨论点 6：如果我是作者，v2 出不出 SM100a 专版？

### 8.1 支持发布的论据

- B300/SM100 有 tcgen05、TMEM、CTA group 等新硬件，理论 tensor-core 峰值更高；
- 官方当前已经使用 TMA，说明代码并非完全停留在旧内存模型；
- V-split 实验在不改变数学公式和 API 的情况下，对 K3 TP8 每卡 H=12 取得了 44.7%–46.2% 的正收益；
- 这个收益来自更合理的 CTA-level 并行度，且可以通过 SM10.x dispatch 和原始 fallback 控制风险；
- 如果把多个 16×16 产品重排成更大的规则 GEMM，或者选择与 UMMA 匹配的 64×16×16/更大 tile，填充浪费可以下降；
- persistent kernel 或 affine scan 若提高 K2 的 active warps，新硬件才可能有足够工作量；
- 对固定部署在 B300 的服务，额外维护一条专用路径有时值得换取尾延迟收益。

### 8.2 反对现在发布的论据

- 当前 CHUNK=16 的核心小矩阵和 tcgen05 的 M=64 不匹配，单纯填充会做 4 倍行工作；
- tcgen05 是异步 TMEM 协议，不是普通 mma.sync 的 drop-in replacement；layout、TMEM load、mbarrier、commit/wait 和 epilogue 都要重做；
- B300 microbenchmark 在有用工作归一化后仍慢约 1.85×；
- 即便使用两个 CTA 的 `UTCHMMA.2CTA` 精确覆盖 `128×16×16`，完整 state update 仍比 HMMA 慢约 3.0×；
- NCU 显示 recurrence 的 DRAM/HMMA 都没有饱和，主要问题是低并行度和状态链，换更高峰值的算力管线不对症；
- 大 CHUNK 虽然可能更适合 UMMA，但当前 gate 公式会遇到下溢、倒数溢出、求逆 FLOPs 和 workspace 快速增长，必须先做 rescale/算法设计；
- 无条件 V-split 也会回归：H=96 FP32 state 时 V64 比原始 V128 慢约 3.9%，N=8/H=12 的 FP32 varlen 中 V64/V32 也分别慢约 1.8%/9.2%；
- 专版会增加编译、测试、架构 dispatch 和未来 CUTLASS 版本升级的维护矩阵，而当前通用 HMMA 路径在 B300 上已经取得 2.30× 对 Triton 的收益。

### 8.3 最终决策

**建议发布“SM10.x V-split/BPW1 优化 + heuristic fallback”的 PR，暂不发布
“tcgen05-only SM100a v2”。**前者已经在目标 TP8/H=12 形状上获得显著正
收益，后者则被 microbenchmark 证明没有当前 workload 的收益依据。

这里的“发布”指把经过 correctness 和 benchmark 验证的 CTA 划分优化提交
给官方评审；并不声称所有 SM100/SM103 shape 都会加速。默认策略只在
SM10.x 进入，显式环境变量可复现每种 mapping，其他架构仍走原始 V128/BPW2。

未来如果要进一步做真正的 tcgen05 专版，至少应满足下面一项，再值得重新 benchmark：

1. 将多个 16×16 产品融合成真正匹配 UMMA 的大 GEMM，避免 64 行中 48 行是 padding；
2. 用 affine/prefix scan 打破 chunk 间状态链，显著提高 K2 active warps；
3. 设计经过证明的更大 CHUNK + rescale/log-domain 表达，使大 tile 的数值和工作区可控；
4. 在真实 TP/batch 负载上确认 SM100 专版能把端到端 latency 降低，而不是只提高某一条指令的理论吞吐。

---

## 9. 交付物索引和局限说明

### 代码

- [FlashKDA/setup.py](FlashKDA/setup.py)：支持题目提供的外部 CUTLASS；
- [FlashKDA/README.md](FlashKDA/README.md)：构建说明；
- [FlashKDA/csrc/flash_kda.cpp](FlashKDA/csrc/flash_kda.cpp)：SM10.x heuristic、显式 mapping override 和 state dispatch；
- [FlashKDA/csrc/smxx/fwd_kernel2.cuh](FlashKDA/csrc/smxx/fwd_kernel2.cuh)：V-slice layout、TMA offset 和 BPW1 warp mapping；
- [FlashKDA/csrc/smxx/fwd_launch.cu](FlashKDA/csrc/smxx/fwd_launch.cu)：V-split 模板实例化、线程数和 K2 grid；
- [experiments/numeric_analysis.py](experiments/numeric_analysis.py)：gate 范围、Neumann FLOPs、workspace 估算；
- [experiments/state_precision.py](experiments/state_precision.py)：FP64 FLA recurrent reference 精度测试；
- [experiments/reference_compare.py](experiments/reference_compare.py)：对拍题目提供的 naive recurrent/chunk reference；
- [experiments/tcgen05_vs_mma.cu](experiments/tcgen05_vs_mma.cu)：B300 tcgen05 与 mma.sync 对照；
- [experiments/tcgen05_2sm_state_update.cu](experiments/tcgen05_2sm_state_update.cu)：精确匹配 `128×16×16` 的 2SM tcgen05 state-update 对照；
- [experiments/layout_dump.cu](experiments/layout_dump.cu)：TMEM layout 证据；
- [experiments/summarize_ncu.py](experiments/summarize_ncu.py)：把 NCU 原始 CSV 提炼成可读摘要。

### 数据

- [results/benchmark_b300.md](results/benchmark_b300.md)：端到端 benchmark；
- [results/sass_b300.txt](results/sass_b300.txt)：官方扩展 SASS 统计；
- [results/sass_vsplit_b300.txt](results/sass_vsplit_b300.txt)：最终 V-split binary 的 SASS 统计；
- [results/sass_final_b300.txt](results/sass_final_b300.txt)：最终 `+final` binary 的 SASS 统计；
- [results/ncu_b300_summary.md](results/ncu_b300_summary.md)：NCU metric；
- [results/numeric_analysis.md](results/numeric_analysis.md)：纸面数值/计算量；
- [results/state_precision_b300.md](results/state_precision_b300.md)：state 精度；
- [results/reference_compare_b300.md](results/reference_compare_b300.md)：`fla_kda_ref` 独立对拍；
- [results/layout_b300.txt](results/layout_b300.txt)：TMEM layout；
- [results/tcgen05_vs_mma_b300.txt](results/tcgen05_vs_mma_b300.txt)：microbenchmark。
- [results/tcgen05_2sm_b300.md](results/tcgen05_2sm_b300.md)：2SM tcgen05、TMEM load 扫描、SASS 和完整 state-update 对照。
- [results/sass_tcgen05_2sm_b300.txt](results/sass_tcgen05_2sm_b300.txt)：2SM binary 的 `UTCHMMA.2CTA` 反汇编摘录。
- [results/persistent_dispatch_b300.md](results/persistent_dispatch_b300.md)：mixed-varlen
  排列实验以及 task-queue persistent 的接入决策。
- [results/vsplit_b300.md](results/vsplit_b300.md)：早期 V-split/BPW1 实验记录；
- [results/final_optimization_b300.md](results/final_optimization_b300.md)：最终优化栈、warp 扫描、heuristic、NCU/SASS 和复验结果。

### 必须诚实保留的边界

这次挑战没有把未经验证的 tcgen05 版本接入 FlashKDA 生产 API，因此没有
声称“tcgen05 完整 kernel 已经通过 580 项测试”。tcgen05 部分是可运行、
可对拍、可反汇编的最小实验，结论为当前小 tile 不值得只换 opcode；V-split
部分则已经接入生产 API，通过完整 correctness matrix，并在 H=12 上取得
端到端正收益。最终建议是提交经过 heuristic 保护的 V-split PR，同时把
tcgen05-only 方案保留为未来大 tile/更高并行度重构的研究方向。


---
