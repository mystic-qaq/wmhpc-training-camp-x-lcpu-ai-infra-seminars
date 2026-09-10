#!/usr/bin/env python3
"""Print the paper calculation behind FlashKDA's CHUNK=16 choice.

This script has no CUDA/PyTorch dependency.  It intentionally reports the
base-2 exponent used by the kernel as well as the equivalent natural-log
value, because the kernel computes ``ex2`` after multiplying the natural
lower bound by log2(e).
"""

from __future__ import annotations

import math


LOWER_BOUND = -5.0
LOG2E = math.log2(math.e)
BF16_MIN_NORMAL_LOG2 = -126.0
BF16_MIN_NORMAL = 2.0 ** -126
BF16_MAX = (2.0 - 2.0 ** -7) * 2.0 ** 127
FP32_MAX = (2.0 - 2.0 ** -23) * 2.0 ** 127
D = 128


def scientific_from_log10(log10_value: float) -> str:
    """Format a positive number given its base-10 logarithm without underflow."""
    exponent = math.floor(log10_value)
    mantissa = 10.0 ** (log10_value - exponent)
    return f"{mantissa:.4f}e{exponent:+d}"


def fmt_int(value: int) -> str:
    return f"{value:,}"


def main() -> None:
    print("# FlashKDA CHUNK 数值/计算量纸面分析")
    print()
    print("假设 lower_bound=-5，且每个 token 的门控值接近最坏情况 -5。")
    print("源码先乘 log2(e)=%.12f，再调用 ex2.approx.ftz.f32。" % LOG2E)
    print()
    print("## 门控累计值")
    print()
    print("| CHUNK | 最小自然对数累计值 | 最小 base-2 指数 | exp(cumsum)（约） | 相对 BF16 最小 normal | exp(-cumsum)（约） | 相对 FP32 最大值 |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for chunk in (16, 32, 64):
        log_e = LOWER_BOUND * chunk
        log2_value = log_e * LOG2E
        log10_value = log_e / math.log(10.0)
        ratio_log2 = log2_value - BF16_MIN_NORMAL_LOG2
        if ratio_log2 >= 0:
            ratio = f"{2.0 ** ratio_log2:.3g} × min-normal"
        else:
            ratio = f"min-normal / {2.0 ** (-ratio_log2):.3g}"
        reciprocal_log10 = -log_e / math.log(10.0)
        reciprocal_ratio = math.exp(-log_e) / FP32_MAX
        print(
            f"| {chunk} | {log_e:.0f} | {log2_value:.3f} | "
            f"{scientific_from_log10(log10_value)} | {ratio} | "
            f"{scientific_from_log10(reciprocal_log10)} | {reciprocal_ratio:.3g} × FP32 max |"
        )
    print()
    print("BF16 最小 normal=2^-126=%.8e；FlashKDA 还使用 FTZ，因此低于该范围的结果会按 0 处理。" % BF16_MIN_NORMAL)
    print("BF16 最大值约为 %.8e，FP32 最大值约为 %.8e。" % (BF16_MAX, FP32_MAX))
    print("除了 exp(cumsum) 的下溢，kernel 还会计算 exp(-cumsum) 来恢复 k；CHUNK=32 的最坏值已超过 FP32 最大值，因而在 ex2 输出阶段就可能变成 inf。")
    print()

    print("## Neumann 求逆与存储/计算量")
    print()
    print("对严格三角矩阵 L，级数最多需要 I 到 L^(C-1)，所以项数是 C。")
    print("源码用二进制幂合并：需要 L^2、L^4、...、L^(C/2)，每个幂一次平方和一次累加。")
    print()
    print("| CHUNK C | 级数项数 | 幂级数 | C×C 矩阵乘次数 | 每次乘 FLOPs | 求逆段 FLOPs | 一个 C×C BF16 矩阵 | 每 tile workspace (D=128) |")
    print("|---:|---:|:---|---:|---:|---:|---:|---:|")
    for chunk in (16, 32, 64):
        powers = list(range(2, chunk, 2))  # replaced below by powers of two
        powers = [2**i for i in range(1, int(math.log2(chunk)))]
        matmuls = 2 * len(powers)
        one_flop = 2 * chunk**3
        total_flop = matmuls * one_flop
        matrix_bytes = chunk * chunk * 2
        workspace = 3 * chunk * D * 2 + D * 4 + 2 * chunk * chunk * 2
        powers_text = ", ".join(f"L^{p}" for p in powers)
        print(
            f"| {chunk} | {chunk} | {powers_text} | {matmuls} | "
            f"{fmt_int(one_flop)} | {fmt_int(total_flop)} | "
            f"{fmt_int(matrix_bytes)} B | {workspace / 1024:.1f} KiB |"
        )
    print()
    print("相对 C=16，求逆段的粗略 dense-FLOP 比例是 1×、10.67×、106.67×。")
    print("这还没有计入更大的寄存器片段、同步和 tile 拆分；因此是偏乐观的下界。")
    print()
    print("## SM80 tile 数量（D=128 的另一侧）")
    print()
    print("SM80 的基本 atom 是 m16n8k16；一个逻辑 16×16 乘法由两个 N 向 atom 组成。")
    print("若把 C×C dense GEMM 按 16×16 tile 拆分，单次乘法和求逆段大致如下：")
    print()
    print("| C | 每次 C×C 乘法的逻辑 16×16 tiles | 每次 HMMA atoms | 求逆段 HMMA atoms（估算） |")
    print("|---:|---:|---:|---:|")
    for chunk in (16, 32, 64):
        tiles = (chunk // 16) ** 3
        atoms = 2 * tiles
        matmuls = 2 * (int(math.log2(chunk)) - 1)
        print(f"| {chunk} | {tiles} | {atoms} | {atoms * matmuls} |")


if __name__ == "__main__":
    main()
