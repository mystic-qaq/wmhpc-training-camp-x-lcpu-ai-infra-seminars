"""问题 7.8（选做）：softmax in Triton（FROM-SCRATCH）。

注：此题可以不用GPU (conftest.py 会自动切到 interpreter 模式)。

contract：
- softmax(x) 接收形状 (M, N) 的 2D tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 自己写，一个 program 处理一行；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意（用 mask 处理），可以假设 N <= 4096，BLOCK_SIZE 用
  triton.next_power_of_2(N) 是常见做法；
- 通过 pytest tests/test_softmax.py 即为完成。
"""

import torch
import triton
import triton.language as tl


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2, "softmax 只支持二维输入"
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    y = torch.empty_like(x)

    @triton.jit
    def kernel(
        X_ptr, Y_ptr,
        M, N,
        BLOCK_SIZE: tl.constexpr,
        stride_xm, stride_xn,
        stride_ym, stride_yn,
    ):
        row_idx = tl.program_id(0)
        col_idx = tl.arange(0, BLOCK_SIZE)
        mask = col_idx < N

        # ====== 空 1：加载当前行，注意 mask ======
        x_row = tl.load(X_ptr + row_idx * stride_xm + col_idx * stride_xn, mask=mask, other=-float("inf"))

        # ====== 空 2：先减最大值，再做 exp 与求和 ======
        max_val = tl.max(x_row, axis=0)
        x_row_stable = x_row - max_val
        exp_row = tl.exp(x_row_stable)
        sum_exp = tl.sum(exp_row, axis=0)

        # ====== 空 3：写回 softmax 结果 ======
        softmax_row = exp_row / sum_exp
        tl.store(Y_ptr + row_idx * stride_ym + col_idx * stride_yn, softmax_row, mask=mask)

    grid = (M,)
    kernel[grid](
        x, y,
        M, N,
        BLOCK_SIZE,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
    )
    return y
