import torch.nn as nn

from search_space.norm_pool import NormPool as _BaseNormPool, LinearNorm


class NormPool(_BaseNormPool, nn.Module):
    def get_candidate(self, candidate_name):
        return self.get_norm(candidate_name)

    def forward(self, x):
        return self.norm_operation(x)
