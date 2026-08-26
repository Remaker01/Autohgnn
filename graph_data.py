import torch, os.path
import random
import numpy as np
from torch_geometric.datasets.coauthor import Coauthor
from torch_geometric.datasets.amazon import Amazon
from torch_geometric.datasets import Planetoid
# from ogb.nodeproppred import PygNodePropPredDataset
import utils

class GraphData(object):

    def __init__(self, data_name, shuffle=False):

        if data_name in ["CS", "Physics", "Computers", "Photo", "Cora", "CiteSeer", "Pubmed"]: # data_name in ["CS", "Physics", "Photo", "Computers", "ogbn-arxiv",
                        #  "ogbn-products", "Cora", "CiteSeer", "Pubmed"]:

            data_path = r"D:\graph_datas"

            if data_name in ["CS", "Physics"]:
                data = Coauthor(data_path, data_name)
            elif data_name in ["Computers", "Photo"]:
                data = Amazon(data_path, data_name)
            elif data_name in ["Cora", "CiteSeer", "Pubmed"]:
                data = Planetoid(data_path, data_name)
            else:
                raise RuntimeError("Sorry current version don't "
                                "Support this default datasets", data_name)

            self.data = data[0]
            self.data.num_classes = data.num_classes

            train_ratio = None
            val_ratio = None
            test_ratio = None
            if data_name == "CS":
                train_ratio = 3000
                val_ratio = 450
            elif data_name == "Physics":
                train_ratio = 500
                val_ratio = 150
            elif data_name == "Computers":
                train_ratio = 200
                val_ratio = 300
            elif data_name == "Photo":
                train_ratio = 3500
                val_ratio = 240
            elif data_name == "Cora" or data_name == "CiteSeer" or data_name == "Pubmed":
                # train_ratio = self.count_(self.data.train_mask)
                train_ratio = self._count_(self.data.train_mask)
                val_ratio = self._count_(self.data.val_mask)
                test_ratio = self._count_(self.data.test_mask)

            # train / val / test mask construction
            index = [i for i in range(self.data.num_nodes)]

            if shuffle:
                random.shuffle(index)

            train_index = index[:train_ratio]
            val_index = index[train_ratio:train_ratio+val_ratio]

            if data_name in ["CS", "Photo", "Computers", "Physics"]:
                test_index = index[train_ratio+val_ratio:]
            else:
                test_index = index[train_ratio+val_ratio:train_ratio+val_ratio+test_ratio]

            train_mask = torch.tensor(self._mask(train_index, self.data.num_nodes), dtype=torch.bool)
            val_mask = torch.tensor(self._mask(val_index, self.data.num_nodes), dtype=torch.bool)
            test_mask = torch.tensor(self._mask(test_index, self.data.num_nodes), dtype=torch.bool)

            self.data.train_index = train_index
            self.data.val_index = val_index
            self.data.test_index = test_index

            self.data.data_name = data_name
            self.data.train_mask = train_mask
            self.data.val_mask = val_mask
            self.data.test_mask = test_mask

            self.data.to(utils.global_device)
        # else:
        #     raise Exception("Sorry current version don't "
        #                     "Support this default datasets", data_name)

    @staticmethod
    def _mask(index, num_node):
        """ create mask """
        mask = np.zeros(num_node, dtype=np.bool_)
        mask[index] = 1 # 优化一下
        return mask

    @staticmethod
    def _count_(mask:torch.Tensor):
        return mask.sum().item()

if __name__=="__main__":
    from torch_geometric.nn.conv import *
    from experiment_gnn import gnnmodel_test
    data_names = ["Pubmed", "Computers", "Physics"]
    # 可调参数：基线 GNN 训练轮次，与主方法训练预算保持一致（默认 100）
    epoch = 100
    gnn_model_list = [GATConv, SAGEConv, SGConv, GraphConv, GATv2Conv]
    gnn_model_args = [
        {'heads':2, 'concat':False},
        {'normalize':True},
        {},
        {},
        {'heads':2, 'concat':False}
    ]
    fp2 = open("./Performance/graph_data_results.txt", "a+")
    for data_name in data_names:
        data = GraphData(data_name).data
        print(f"train/val/test:{data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")
        # data.data.x = data.data.x.half()
        # data.data.y = data.data.y.half()
        for (model,arg) in zip(gnn_model_list,gnn_model_args):
            try:
                res = np.empty((5,))
                for i in range(0, 5):
                    res[i] = gnnmodel_test(model, data, epoch=epoch, verbose=False, lr=0.005, l2=0.0001, **arg)
                info = f"Accuracy on dataset {data_name} using model {model.__name__}:{res.mean():.4f}+-{res.std():.4f}"
                fp2.write(info + '\n')
                print(info)
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA out of memory on data {data_name} model {model.__name__}")
    fp2.close()
