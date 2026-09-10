# CHUNK=16 的纸面量化

生成脚本：[experiments/numeric_analysis.py](../experiments/numeric_analysis.py)。命令：

```bash
python3 experiments/numeric_analysis.py
```

假设 lower_bound=-5，并取每个 token 的门控值接近最坏情况 -5。FlashKDA 把自然指数换成了 ex2：flash_kda.cpp 中先乘 log2(e)=1.442695040889，随后 kernel 调用 ex2.approx.ftz.f32。

## 门控累计

| CHUNK | 最小自然对数累计值 | 最小 base-2 指数 | exp(cumsum)（约） | 相对 BF16 最小 normal | exp(-cumsum)（约） | 相对 FP32 最大值 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | -80 | -115.416 | 1.8049e-35 | 1.54e+03 × min-normal | 5.5406e+34 | 0.000163 × FP32 max |
| 32 | -160 | -230.831 | 3.2575e-70 | min-normal / 3.61e+31 | 3.0698e+69 | 9.02e+30 × FP32 max |
| 64 | -320 | -461.662 | 1.0611e-139 | min-normal / 1.11e+101 | 9.4240e+138 | 2.77e+100 × FP32 max |

BF16 最小 normal=2^-126=1.17549435e-38；FlashKDA 还使用 FTZ，因此低于该范围的结果会按 0 处理。
BF16 最大值约为 3.38953139e+38，FP32 最大值约为 3.40282347e+38。
除了 exp(cumsum) 的下溢，kernel 还会计算 exp(-cumsum) 来恢复 k；CHUNK=32 的最坏值已超过 FP32 最大值，因而在 ex2 输出阶段就可能变成 inf。

## Neumann 求逆

严格三角矩阵 L 的级数最多需要 I 到 L^(C-1)，因而有 C 项。源码用二进制幂合并，需要 L^2,L^4,...,L^(C/2)；每个幂包含一次平方和一次累加矩阵乘。

| C | 级数项数 | 幂级数 | C×C 矩阵乘次数 | 每次乘 FLOPs | 求逆段 FLOPs | 一个 C×C BF16 矩阵 | 每 tile workspace (D=128) |
|---:|---:|:---|---:|---:|---:|---:|---:|
| 16 | 16 | L^2,L^4,L^8 | 6 | 8,192 | 49,152 | 512 B | 13.5 KiB |
| 32 | 32 | L^2,L^4,L^8,L^16 | 8 | 65,536 | 524,288 | 2,048 B | 28.5 KiB |
| 64 | 64 | L^2,L^4,L^8,L^16,L^32 | 10 | 524,288 | 5,242,880 | 8,192 B | 64.5 KiB |

相对 C=16，求逆段的粗略 dense-FLOP 比例是 1×、10.67×、106.67×。
这还没有计入更大的寄存器片段、同步和 tile 拆分；因此是偏乐观的下界。

## SM80 tile 数量（D=128 的另一侧）

SM80 的基本 atom 是 m16n8k16；一个逻辑 16×16 乘法由两个 N 向 atom 组成。
若把 C×C dense GEMM 按 16×16 tile 拆分，单次乘法和求逆段大致如下：

| C | 每次 C×C 乘法的逻辑 16×16 tiles | 每次 HMMA atoms | 求逆段 HMMA atoms（估算） |
|---:|---:|---:|---:|
| 16 | 1 | 2 | 12 |
| 32 | 8 | 16 | 128 |
| 64 | 64 | 128 | 1280 |
