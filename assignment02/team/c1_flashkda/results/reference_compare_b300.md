# FlashKDA 与 `fla_kda_ref` 对拍（B300）

脚本：[experiments/reference_compare.py](../experiments/reference_compare.py)。它使用与生产 kernel 相同的 BF16 q/k，并复现其 16×8 warp-XOR L2 reduction；随后把 raw gate/beta 用数学上的 sigmoid 激活，交给题目提供的 `fla_kda_ref.naive_recurrent_kda` 和 `naive_chunk_kda`。生产 FlashKDA 仍使用自己的 tanh 近似 sigmoid、FP16 inverse、BF16 中间量和 BF16 state，因此这里期待的是数值接近，而不是 bitwise 相同。

命令：

```bash
cd assignment02/team/c1_flashkda
python experiments/reference_compare.py --T 16 64 256
```

B300 输出（relative RMS = `RMS(actual-reference) / RMS(reference)`）：

```text
T=  16 naive_recurrent output_rel_rms=6.062020e-03 state_rel_rms=4.318561e-03 output_max=1.831055e-04 state_max=2.354860e-03
       naive_chunk16   output_rel_rms=6.062020e-03 state_rel_rms=4.318571e-03 output_max=1.831055e-04 state_max=2.354845e-03
T=  64 naive_recurrent output_rel_rms=5.431707e-03 state_rel_rms=4.627853e-03 output_max=2.441406e-04 state_max=1.460969e-03
       naive_chunk16   output_rel_rms=5.431709e-03 state_rel_rms=4.627854e-03 output_max=2.441406e-04 state_max=1.460984e-03
T= 256 naive_recurrent output_rel_rms=5.253784e-03 state_rel_rms=4.970688e-03 output_max=2.441406e-04 state_max=1.020856e-03
       naive_chunk16   output_rel_rms=5.253654e-03 state_rel_rms=4.970699e-03 output_max=2.441406e-04 state_max=1.020849e-03
```

这里还特别处理了 state 的布局：FlashKDA 对外的 `[N,H,D,D]` 物理布局是 V-first，而两个朴素参考在递推中把矩阵看成 `[K,V]`，所以脚本在参考调用前后做了转置。否则方阵的 shape 不变，错误会很隐蔽。三组长度上两种 FLA reference 互相一致，FlashKDA 的 output/state 误差都约为 0.4%–0.6%，与独立 FP64 recurrent sweep 的结论相符。
