import torch
from torch_geometric.nn.conv import *

class ConvPool(torch.nn.Module):
    candidate_list = ["SAGEConv","GATConv", "GraphConv",
                           "GATv2Conv","SGConv"]
    def __init__(self,
                 input_dim=128,
                 output_dim=128,
                 conv_name="GATConv"):

        super(ConvPool, self).__init__()

        if conv_name == "SAGEConv":
            self.conv_operation = SAGEConv(input_dim, output_dim, normalize=True)
        elif conv_name == "GATConv":
            self.conv_operation = GATConv(input_dim, output_dim, heads=3, concat=False)
        elif conv_name == "GraphConv":
            self.conv_operation = GraphConv(input_dim, output_dim, aggr='mean')
        elif conv_name == "GATv2Conv":
            self.conv_operation = GATv2Conv(input_dim, output_dim, heads=3, concat=False)
        elif conv_name == "SGConv":
            self.conv_operation = SGConv(input_dim, output_dim, K=2)
        # elif conv_name == "HyperGraphConv":
        #     self.conv_operation = HypergraphConv(input_dim, output_dim)
        # elif conv_name == "ClusterGCNConv":
        #     self.conv_operation = ClusterGCNConv(input_dim, output_dim)
        else:
            raise ValueError("Sorry current version don't "
                            "Support this default graph convolution", conv_name)

    def forward(self, x, edge_index):

        return self.conv_operation(x, edge_index)

if __name__=="__main__":
    a = ConvPool()
    print(type(a).__name__)