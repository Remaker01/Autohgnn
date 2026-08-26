import torch.nn as nn

from search_space.act_pool import ActPool as _BaseActPool


class ActPool(_BaseActPool, nn.Module):
    def get_candidate(self, candidate_name):
        return self.get_act(candidate_name)

    def forward(self, x):
        return self.act_operation(x)
