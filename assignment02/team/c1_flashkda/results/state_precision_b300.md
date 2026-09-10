# BF16 state 精度验证（B300）

脚本：[`experiments/state_precision.py`](../experiments/state_precision.py)。它把相同的 BF16 输入交给 FlashKDA，同时用 FLA `fused_recurrent_kda` 做 FP64 recurrent reference：门控在调用前用 FP64 计算，reference 保留 FP64 recurrent state；FlashKDA 则走实际 BF16 state 路径。相对 RMS 定义为

```text
sqrt(mean((actual-reference)^2)) / sqrt(mean(reference^2))
```

运行命令：

```bash
cd assignment02/FlashKDA
python ../team/c1_flashkda/experiments/state_precision.py \
  --T 256 1024 8192 --cases random near_lower near_zero
```

`near_lower` 令 raw gate=8，使激活 gate 接近 `-5`；`near_zero` 令 raw gate=-8，使激活 gate 接近 0。这两种情况分别压力测试最快衰减和长期保留状态。`h0` 是幅度约 0.1 的随机 BF16 state。

| gate case | T | output relative RMS | output max abs | final state relative RMS | final state max abs |
|---|---:|---:|---:|---:|---:|
| random | 256 | 0.5517% | 3.55e-4 | 0.4443% | 2.63e-3 |
| random | 1024 | 0.5271% | 4.69e-4 | 0.4249% | 1.26e-3 |
| random | 8192 | 0.5219% | 5.90e-4 | 0.4940% | 4.88e-3 |
| near_lower | 256 | 0.4927% | 3.34e-4 | 0.4714% | 1.83e-3 |
| near_lower | 1024 | 0.4923% | 4.43e-4 | 0.5334% | 1.73e-3 |
| near_lower | 8192 | 0.4644% | 4.71e-4 | 0.4001% | 2.07e-3 |
| near_zero | 256 | 0.6993% | 1.25e-3 | 0.6538% | 1.61e-2 |
| near_zero | 1024 | 0.7755% | 1.69e-3 | 0.7028% | 1.54e-2 |
| near_zero | 8192 | 0.7894% | 2.06e-3 | 0.7147% | 1.92e-2 |

同一组 BF16 初始 state 在 `T=1024` 还分别以 BF16 和 FP32 作为外部 I/O：三组 gate case 均为 `output_exact=True`、`state_exact=True`、`state_max_abs=0`。这符合源码行为：FP32 state 进来后也先转换为片上 BF16；FP32 输出只是把片上 BF16 再转回 FP32。

独立的官方 full correctness 还覆盖了 fixed/varlen/batched、`H∈{1,4,32,96}`、多种长度、state in/out 四种组合、BF16/FP32 两种 I/O，共 **580 passed in 703.33 s**；团队副本在外部 CUTLASS 路径下再次全量运行也得到 **580 passed in 693.96 s**。对 FP64 FLA reference 的 gate/bias sweep 则是 **2 passed**，固定和变长各 10 个 gate/bias case，输出误差均在约 0.3%–0.6% 常见范围；最难的 state case 仍低于 1% relative RMS（有 warning 的 near-zero/大 bias state 项也记录在原始 pytest 输出中）。

这说明 BF16 state 在当前安全 gate 和模型形状下足够稳定，但不能把“对 reference 小误差”误读为 bitwise 等价：生产 kernel 的 FP16 inverse、BF16 中间量和近似 sigmoid 本身也会贡献误差。因此上线前还应加入真实模型分布、长上下文和任务级质量指标。
