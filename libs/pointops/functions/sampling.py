import torch
from torch.autograd import Function

from pointops._C import (
    farthest_point_sampling_cuda,
    farthest_point_sampling_with_dist_cuda,
)


class FarthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, offset, new_offset):
        """
        input: coords: (n, 3), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        assert xyz.is_contiguous()
        n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i - 1], n_max)
        idx = torch.zeros(new_offset[b - 1].item(), dtype=torch.int, device=xyz.device)
        tmp = torch.ones(n, dtype=torch.float, device=xyz.device) * 1e10
        farthest_point_sampling_cuda(
            b, n_max, xyz, offset.int(), new_offset.int(), tmp, idx
        )
        del tmp
        return idx


farthest_point_sampling = FarthestPointSampling.apply


class FarthestPointSamplingWithDist(Function):
    @staticmethod
    def forward(ctx, distance, offset, new_offset):
        """
        input: distance: (n, n), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        assert distance.is_contiguous()
        n, b, n_max = distance.shape[0], offset.shape[0], offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i - 1], n_max)
        idx = torch.cuda.IntTensor(new_offset[b - 1].item()).zero_()
        tmp = torch.cuda.FloatTensor(n).fill_(1e10)
        farthest_point_sampling_with_dist_cuda(
            b, n_max, n, distance, offset.int(), new_offset.int(), tmp, idx
        )
        del tmp
        return idx


farthest_point_sampling_with_dist = FarthestPointSamplingWithDist.apply
