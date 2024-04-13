# -*- coding: utf-8 -*-
# ----------------------------
# @Time    : 2023/12/16 15:05
# @Author  : acedar
# @FileName: verify.py
# ----------------------------

import math
import torch


def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        # n = 16
        # start = 2^(-2^-(log2(16) - 3)) = 2^(-2^(-1))
        # = 2^(-2^(-1)) = 2^-(1)=1/(2^1)
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
                _get_interleave_power_of_2(closest_power_of_2)
                + _get_interleave(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def _fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _gen_alibi_mask(tensor, n_head, max_pos):
    slopes = torch.Tensor(_get_interleave(n_head))
    print("slopes:", slopes)
    # slopes: [0.5000, 0.2500, 0.1250, 0.0625, 0.0312, 0.0156, 0.0078, 0.0039]
    # m=slopes, 每个头的因子

    position_point = torch.arange(max_pos) - max_pos + 1
    # position_point = [-max_pos + 1, -max_pos, -max-pos-1, ..., 0]

    position_point = position_point.unsqueeze(0).unsqueeze(0).expand(n_head, -1, -1)
    diag = torch.diag(position_point[0])
    position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
    print("position_point:", position_point.size())
    # torch.Size([8, 1, 5])

    print("slopes.unsqueeze: ", slopes.unsqueeze(1).unsqueeze(1).size())
    # torch.Size([8, 1, 1])
    # slopes.unsqueeze(1).unsqueeze(1):[[[0.5000]],
    #         [[0.2500]],
    #         [[0.1250]],
    #         [[0.0625]],
    #         [[0.0312]],
    #         [[0.0156]],
    #         [[0.0078]],
    #         [[0.0039]]])

    alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
    print("alibi", alibi.size(), alibi)
    # [8, 1, 5]
    # [[[0.0000, 0.5000, 1.0000, 1.5000, 2.0000]],
    # [[0.0000, 0.2500, 0.5000, 0.7500, 1.0000]], ...

    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = torch.triu(_fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1)
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask


if __name__ == "__main__":
    n_head, max_pos = 8, 5
    alibi_mask = _gen_alibi_mask(None, n_head, max_pos)
    print(alibi_mask.size())
    # torch.Size([8, 5, 5])

    print(alibi_mask)
    # tensor([[[0.0000,   -inf,   -inf,   -inf,   -inf],
    #          [0.0000, 0.5000,   -inf,   -inf,   -inf],
    #          [0.0000, 0.5000, 1.0000,   -inf,   -inf],
    #          [0.0000, 0.5000, 1.0000, 1.5000,   -inf],
    #          [0.0000, 0.5000, 1.0000, 1.5000, 2.0000]],
    #  ...
