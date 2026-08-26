import torch
from torch_geometric.nn import GraphNorm, \
                               InstanceNorm, \
                               LayerNorm, \
                               BatchNorm

class NormPool(torch.nn.Module):
    candidate_list = ["GraphNorm", "InstanceNorm",
                           "LayerNorm", "BatchNorm",
                           "LinearNorm"]
    def __init__(self, input_dim=128, norm_name="GraphNorm"):
        super(NormPool, self).__init__()

        if norm_name == "GraphNorm":
            self.norm_operation = GraphNorm(input_dim)
        elif norm_name == "InstanceNorm":
            self.norm_operation = InstanceNorm(input_dim)
        elif norm_name == "LayerNorm":
            self.norm_operation = LayerNorm(input_dim)
        elif norm_name == "BatchNorm":
            self.norm_operation = BatchNorm(input_dim)
        elif norm_name == "LinearNorm":
            self.norm_operation = LinearNorm()
        else:
            raise Exception("Sorry current version don't "
                            "Support this default graph norm", norm_name)

    def forward(self, x):

        return self.norm_operation(x)

LinearNorm = torch.nn.Identity