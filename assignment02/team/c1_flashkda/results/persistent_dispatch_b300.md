# Persistent 调度评估（B300）

## 先澄清“persistent”是哪一层

当前 K2 recurrence 已经在 chunk 维度使用了 persistent execution：一个
CTA 被分配给一个 `(sequence, head, V-slice)`，在 kernel 内沿着所有 chunk
循环，state 保留在 shared memory/寄存器中。它不是每个 chunk 都重新
launch 一个 kernel。

还可以继续做另一种 persistent：固定数量的 worker CTA 从全局任务队列中
领取多个 `(sequence, head, V-slice)`，以缓解 variable-length 请求的长短
不均。这个方案需要在 worker 切换任务时重新初始化 TMA pipeline、重新
加载 state，并通过 atomic queue 管理任务生命周期。

## 可观测的调度实验

为了判断现有 static CTA mapping 是否已经暴露出严重的 length-order tail，
我对同一组 token 总数、同一组 sequence lengths 只改变序列排列顺序，其他
输入生成和计时方法不变：

```text
[1300, 547, 2048, 963, 271, 3063]
[3063, 271, 963, 2048, 547, 1300]
[271, 547, 963, 1300, 2048, 3063]
```

命令为：

```bash
source /home/lcpu/13168002/.venv-c1-managed/bin/activate
python experiments/bench_variants.py --H 12 \
  --seq-lens 1300 547 2048 963 271 3063 \
  --state all --variants auto 128:2 64:1 32:1 \
  --warmup 10 --iters 50 --repeats 3
```

`auto` 在这个 H=12、N=6 的形状选择 `V64/BPW1 + 8 update warps`。端到
端 FlashKDA 时间如下，单位 ms：

| sequence order | no state | BF16 state | FP32 state |
|---|---:|---:|---:|
| 原始顺序 | 0.25301 | 0.25283 | 0.25414 |
| 最长序列放最前 | 0.25382 | 0.25211 | 0.25407 |
| 最长序列放最后 | 0.25347 | 0.25313 | 0.25419 |

三种排列的最大相对差约 0.7%，小于这个级别 benchmark 的节点/时钟
波动。这个实验不能证明任意 workload 都不需要 work stealing，但没有
观察到当前目标 mixed-varlen 形状存在可由简单 persistent queue 消除的
大 tail。

## 为什么暂不接入 task-queue persistent kernel

对 K3 目标 fixed case `N=1`，所有可并行任务的 chunk 数相同，且每个
CTA 已经持有一条完整的 chunk chain；task queue 没有额外任务可供 worker
领取，只有 atomic 和任务切换开销。对常见的多 sequence case，V-split
已经把 `(sequence, head, value-slice)` 暴露为大量独立 CTA，静态 grid 可
直接让 CUDA scheduler 做粗粒度负载分配。

因此当前证据支持的决策是：保留“CTA 跨 chunk 持久化”的实现，不引入额外
的全局任务队列。对于极端长度倾斜、极小 head 数或更大的服务 batch，若
未来 profiling 显示 K2 尾部成为主导，再实现真正的 persistent worker
版本，并把它与 static mapping 做同一套 correctness/NCU/端到端对照。
