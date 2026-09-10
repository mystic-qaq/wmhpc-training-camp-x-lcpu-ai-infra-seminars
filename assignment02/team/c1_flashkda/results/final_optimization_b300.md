# 最终 SM10.x 优化记录：B300 上的 FlashKDA

这份记录描述当前可以交付的版本。`results/vsplit_b300.md` 保留了早期
V-split 实验；这里的数字来自最终 `+final` 构建，补上了 persistent state
fragment、专用 state-update warps 和 Direct-B layout 后的结果。tcgen05 的
独立验证记录在 [results/tcgen05_2sm_b300.md](tcgen05_2sm_b300.md)。

## 1. 构建和计时口径

- GPU：NVIDIA B300 SXM6 AC，compute capability 10.3（SM103）。
- CUDA：13.0；PyTorch：2.13.0+cu130。
- CUTLASS：题目提供的 `5c149f52a436782210263fb2f19b354443a61c6a`。
- 输入维度：`K=V=D=128`，K2 的固定 chunk 为 16。
- 端到端计时使用 CUDA event，包含公开 `flash_kda.fwd` 调用、workspace
  分配和 beta 转置。
- 官方 benchmark 使用 `warmup=30,iters=200,repeats=5`；mapping 对照也
  使用同一输入和 CUDA event。

构建命令：

```bash
export FLASH_KDA_CUTLASS_DIR=/home/lcpu/13168002/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/FlashKDA/cutlass
export FLASH_KDA_CUDA_ARCHS=103a
unset FLASH_KDA_NVCC_DEFINES
export FLASH_KDA_VERSION_SUFFIX=+final
uv pip install --no-build-isolation --reinstall ./FlashKDA
```

最终二进制版本为 `flash_kda 0.0.1+final`。`setup.py` 默认打开
`FLASH_KDA_DIRECT_B_LAYOUT`；显式设置 `FLASH_KDA_NVCC_DEFINES=` 可以做
没有该布局优化的 A/B 构建。

## 2. 这次实际做了什么

### 2.1 把独立的 value 行拆成多个 CTA

K2 的状态更新可以写成：

```text
S_next[v, :] = decay[:] * S_now[v, :] + K_restored[:, :] @ U[:, v, :]
```

不同的 value 行 `v` 互不写冲突，所以可以把一个原来处理完整 128 行的
CTA 拆成多个 CTA。每个 CTA 只写自己的 `[v_offset:v_offset+V_TILE]`
行，公开 state layout `[N,H,V,K]` 不变：

| mapping | 每个 head 的 CTA 数 | 每个 MMA warp 的 value block | 适用意义 |
|---|---:|---:|---|
| `V128/BPW2` | 1 | 2 个 16×16 block | 原始 fallback |
| `V64/BPW1` | 2 | 1 个 block | 中等 head 数 |
| `V32/BPW1` | 4 | 1 个 block | H=24 等形状 |
| `V16/BPW1` | 8 | 1 个 block | K3 TP8 每卡 H=12 |

`BPW1`（one block per warp）很关键。仅增加 CTA 数、却让一个 warp 继续
塞两个 block，会把 CTA 内的独立工作压得太密，B300 上反而变慢。

代价是每个 slice 都会重复加载完整的 q/k/workspace；因此不能无条件选择
最小的 V_TILE，必须通过 heuristic 保护已有 head/batch 足够大的形状。

### 2.2 让 state update 真正并行，并与输出重叠

V-split 之后，一个 CTA 内的 state update 仍然可能占用原来的关键路径。
最终版本把 warp 分成四类：

```text
MMA warps  ── 计算 U、输出和 Mqk@U
update warps ── 独立计算 K_restored@U 并更新 state tile
load warp  ── TMA 预取 q/k/v/workspace
store warp ── TMA 写回 output/final state
```

MMA warp 将当前 chunk 的 `U` 发布到已完成使用的 `v` shared-memory
buffer；update warps 随即读取它。两个 NamedBarrier 保证“U 已写完”和
“state 已写完”，同时 load pipeline 不会复用仍在使用的 stage。

对于 `V_TILE>=32`，update warp 将自己负责的 state 16×16 fragment 保留在
寄存器中，后续 chunk 省掉重复的 shared-memory state load；每个 chunk
仍然写回 shared memory，因为下一次 output 计算需要从统一的 state 视图
读取。`V16` 上实测寄存器常驻的 live range 反而不划算，因此保留 load/store
路径。

### 2.3 Direct-B layout

默认让 U 以 HMMA B operand 需要的排列发布到 shared memory，update warp
可以直接用 B-copy 读取，减少 `MOVM_T` 布局转换。`V32 + FP32 state` 特别
保留 C-layout 路径，因为 FP32 state 的转换和 epilogue live range 较大，
在 B300 上 Direct-B 会回归。

## 3. 最终自动策略

只有 `deviceProp.major == 10` 才启用；其他架构以及不确定的情况继续用
原始 `V128/BPW2`。以下是代码中的最终策略，环境变量 override 优先：

| 条件 | V_TILE/BPW | state-update warps |
|---|---|---:|
| 单 sequence，`H<=16` | `16/1` | 8 |
| 单 sequence，`17<=H<=32,T<=64` | `16/1` | 8 |
| 单 sequence，`17<=H<=32,T>64` | `32/1` | 4 |
| 单 sequence，`33<=H<=64` | `64/1` | 8 |
| 单 sequence，`65<=H<=96` 且 BF16/no-state | `64/1` | 8 |
| 单 sequence，其他 FP32 state | `128/2` | 0 |
| packed，`H<=4` | `V16`（小 N）或 `V32` | 8 或 4 |
| packed，`H=5..16` 且 `N<=2` | `64/1` | 8 |
| packed，`H=5..16` 且长序列/FP32 state/大 N | `128/2` | 0 |
| 其他 packed 形状 | 保守使用 `V128/2` | 0 |

packed 策略还使用 `N` 和平均序列长度；完整条件以
`FlashKDA/csrc/flash_kda.cpp` 为准。可复现实验 mapping：

```bash
export FLASH_KDA_V_TILE=16
export FLASH_KDA_BLOCKS_PER_WARP=1
export FLASH_KDA_STATE_UPDATE_WARPS=8
```

可用的 override 是：

- `FLASH_KDA_V_TILE=128,64,32,16`；
- `FLASH_KDA_BLOCKS_PER_WARP=1` 或 `2`；
- `FLASH_KDA_STATE_UPDATE_WARPS=0,4,8,16`，或 `auto`；
- 旧变量 `FLASH_KDA_PARALLEL_STATE_UPDATE=0/1` 仍作为兼容别名。

`StateUpdateWarps=16` 只为 `V32/V64 + BPW1` 实例化，默认不启用。

## 4. 最终固定长度性能

以下是最终二进制上的 mapping 对照。单位为 ms，越小越好；`auto` 同时
选择了对应的 state-update warp 数。baseline 是同一输入下强制
`V128/BPW2`。

| H | auto mapping | state | baseline `V128/2` | final auto | 时间下降 |
|---:|---|---|---:|---:|---:|
| 12 | `V16/1 + 8` | no state | 0.7883 | 0.4246 | 46.1% |
| 12 | `V16/1 + 8` | BF16 | 0.7899 | 0.4251 | 46.2% |
| 12 | `V16/1 + 8` | FP32 | 0.7579 | 0.4195 | 44.7% |
| 24 | `V32/1 + 4` | no state | 0.8224 | 0.4732 | 42.5% |
| 24 | `V32/1 + 4` | BF16 | 0.8229 | 0.4755 | 42.2% |
| 24 | `V32/1 + 4` | FP32 | 0.7923 | 0.4943 | 37.6% |
| 64 | `V64/1 + 8` | no state | 0.9359 | 0.7091 | 24.2% |
| 64 | `V64/1 + 8` | BF16 | 0.9376 | 0.7109 | 24.2% |
| 64 | `V64/1 + 8` | FP32 | 0.9068 | 0.7391 | 18.5% |
| 96 | `V64/1 + 8` | no state | 1.0275 | 1.0057 | 2.1% |
| 96 | `V64/1 + 8` | BF16 | 1.0286 | 1.0022 | 2.6% |
| 96 | `V128/2 + 0` | FP32 | 0.9972 | 0.9971 | ≈0% |

H=12 是 K3 TP8 每卡约 12 个 head 的关键场景。这里的收益来自 CTA
并行度和 state update overlap；最终 SASS 仍是 HMMA，不是 tcgen05。

## 5. state-update warp 扫描

固定 mapping 后扫描 update warp 数，可以看到为什么不能把一个数值全局
套用到所有形状：

| mapping/形状 | update warps | no state | BF16 state | FP32 state |
|---|---:|---:|---:|---:|
| `V16,H12` | 0 | 0.5842 | 0.5672 | 0.5841 |
|  | 4 | 0.4390 | 0.4404 | 0.4345 |
|  | 8 | **0.4244** | **0.4251** | **0.4202** |
| `V32,H24` | 0 | 0.6108 | 0.6122 | 0.6300 |
|  | 4 | **0.4731** | **0.4755** | **0.4944** |
|  | 8 | 0.5018 | 0.5016 | 0.5263 |
|  | 16 | 0.5228 | 0.5228 | 0.5451 |
| `V64,H64` | 0 | 0.7728 | 0.7856 | 0.8006 |
|  | 4 | 0.7687 | 0.7775 | 0.7807 |
|  | 8 | **0.7087** | **0.7111** | **0.7392** |
|  | 16 | 0.7743 | 0.7514 | 0.7638 |

这组实验也验证了“更多 warps 不等于更快”：update warp 自身会消耗线程、
寄存器和 barrier 资源，超过拐点后调度开销反而增加。

## 6. NCU 和 SASS 证据

final binary 的 SASS 统计保存在
[`results/sass_final_b300.txt`](sass_final_b300.txt)：

```text
HMMA.16816.F16       24
HMMA.16816.F32.BF16  23696
UTMALDG.1D            324
UTMALDG.2D            324
UTMALDG.3D          22298
UTCHMMA/UTCMMA/TCGEN05/WGMMA  0
```

对应的 NCU 摘要：

| case | block | grid | shared memory | registers/thread allocated | HMMA active | active warps |
|---|---:|---|---:|---:|---:|---:|
| `H12,V16` | 352 | `(1,96,1)` | 49.664 KiB | 48 | 7.51% | 17.19% |
| `H24,V32` | 256 | `(1,96,1)` | 56.320 KiB | 72 | 14.40% | 12.50% |
| `H64,V64` | 448 | `(1,128,1)` | 69.632 KiB | 56 | 21.79% | 21.86% |

原始 `.ncu-rep` 和摘要分别见 `results/ncu_final_v16_h12*`、
`results/ncu_final_v32_h24*` 和 `results/ncu_final_v64_h64*`。NCU 会改变
运行时长，所以这里主要用它确认资源/指令结构，不拿 profile duration
直接替代 benchmark。

## 7. 正确性和精度

最终 `+final` 二进制在 B300、自动 heuristic、无环境 override 下运行：

```text
580 passed in 704.18s (0:11:44)
```

覆盖 fixed、varlen、batched、长度 tail、长序列、initial/final state 的
所有组合，以及 BF16/FP32 state I/O。命令：

```bash
export FLASH_KDA_DIST_GPU=1
cd FlashKDA/tests
pytest test_fwd_full.py -x -v -n 4
```

独立 FP64 recurrent reference 的误差数据见
[`results/state_precision_final4_b300.md`](state_precision_final4_b300.md)：
最难的 near-zero gate、`T=8192` 仍约为 `0.715%` state relative RMS，
没有随长度失控。这是数值稳定性证据，不等同于任务级质量评估。

## 8. 失败尝试和边界

- `tcgen05-only`：在与当前 16×16 工作匹配的最小可运行实验中，tcgen05
  必须执行 64×16×16，填充后三分之四的 M 行；按有用工作归一化仍约慢
  `1.85x`，且需要重做 TMEM/layout/mbarrier/epilogue。
- 进一步实现了精确覆盖完整 state tile 的 2SM `UTCHMMA.2CTA`：两个 CTA
  共同完成 `128×16×16`，并计入 TMEM 读取和 BF16 epilogue，B300 上为
  `224.65 ns`，而 HMMA 对照约 `74.7 ns`，仍慢约 `3.01x`。因此这不是
  单纯 M 维 padding 导致的负结果；详见
  [tcgen05_2sm_b300.md](tcgen05_2sm_b300.md)。
- `persistent dispatch`：当前实现已经让一个 CTA 跨 chunk 持有一条 state
  链；额外的全局 task-queue worker 在测过的 mixed-varlen 形状上没有可见
  tail 可消除，故暂不增加 atomic/task-switch 开销。完整实验见
  [persistent_dispatch_b300.md](persistent_dispatch_b300.md)。
- `InputStages=2`：最终代码上 H=12 约从 `0.424 ms` 退化到
  `0.564 ms`；不采用。
- `StateUpdateWarps=16`：只有少数实验组合可能接近，主流形状多数慢于
  4/8；只保留为实验实例。
- packed/varlen 不能无条件 V-split。比如 N=8、H=12、FP32 state 的长
  序列上 V64/V32 会回归，所以 heuristic 主动回退到 V128。

因此当前最可靠、可提交 PR 的结果是“SM10.x V-split/BPW1 + persistent
state-update overlap + measured heuristic fallback”。真正的 tcgen05
专版需要先把多个小矩阵融合成匹配 UMMA 的大 tile，或先用 affine scan
打破 chunk 链依赖；仅替换 MMA opcode 没有充分收益依据。
