import numpy as np

from dhg.data import CoauthorshipCora, CocitationCora, CocitationCiteseer, CoauthorshipDBLP, Cooking200
from dhg.nn.convs.hypergraphs import *
from dhg.metrics import VertexClassificationEvaluator as Evaluator
from dhg.structure import Hypergraph, Graph
from experiment_gnn import gnnmodel_test

import torch, random
from torch import optim, nn
import torch.nn.functional as F
from torch_geometric.data import Data
import utils
from torch_geometric.nn.conv import *

class HypergraphData:
    _ROOT = r"D:/graph_datas"
    def __init__(self, data_name:str, device=utils.global_device):
        hdata = None
        if data_name == "Cora_CA":
            hdata = CoauthorshipCora(data_root=self._ROOT)
        elif data_name == "Cora_CC":
            hdata = CocitationCora(self._ROOT)
        elif data_name == "Citeseer_CC":
            hdata = CocitationCiteseer(self._ROOT)
        elif data_name == "DBLP":
            hdata = CoauthorshipDBLP(self._ROOT)
        if hdata is None:
            raise ValueError(f"{data_name} not supported")
        y = hdata["labels"]
        X = hdata["features"]
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X,device=device)
        gdata = Data(x=X, y=y).to(device)
        gdata.train_mask = hdata["train_mask"]
        gdata.val_mask = hdata["val_mask"]
        gdata.test_mask = hdata["test_mask"]
        val_index = torch.where(gdata.val_mask == True)[0].tolist()
        gdata.num_classes = hdata["num_classes"]
        self.graph = gdata
        self.hypergraph = Hypergraph(num_v=hdata["num_vertices"], e_list=hdata["edge_list"], v_weight=X, device=device)

class _HypergraphModule(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.layer0 = layers[0]
        self.layer1 = layers[1]
    def forward(self, X: torch.Tensor, hg, cached_G=None) -> torch.Tensor:
        X = self.layer0(X, hg, cached_G) if isinstance(self.layer0, HyperGCNConv) else self.layer0(X, hg)
        # X = F.relu(X, inplace=True)
        return self.layer1(X, hg, cached_G) if isinstance(self.layer1, HyperGCNConv) else self.layer1(X, hg)

ev = Evaluator(["accuracy"])
def hgnnmodel_test(model, data:HypergraphData, verbose=True, epoch=100, lr=0.001, l2=1e-4):
    '''超图对照实验，返回准确率 '''
    global ev

    def train(net, X, hg, lbls, train_idx, optimizer: optim.Optimizer, g=None):
        net.train()
        optimizer.zero_grad()
        outs = net(X, hg, g)
        outs, lbls = outs[train_idx], lbls[train_idx]
        loss = F.cross_entropy(outs, lbls)
        loss.backward()
        optimizer.step()
        return loss.item()

    def infer(net, X, hg, lbls, idx, test=False, g=None):
        net.eval()
        outs = net(X, hg, g)
        outs, lbls = outs[idx], lbls[idx]
        if not test:
            return ev.validate(lbls, outs)
        return ev.test(lbls, outs)

    X, lbl = data.graph.x, data.graph.y
    hg, g = data.hypergraph, None
    if model == HyperGCNConv:
        g = Graph.from_hypergraph_hypergcn(hg, X, True, device=utils.global_device)
    layer1 = model(X.shape[1], 64, use_bn=True, bias=False, drop_rate=0).to(utils.global_device)
    layer2 = model(64, data.graph["num_classes"], bias=False, is_last=True).to(utils.global_device)
    net = _HypergraphModule([layer1, layer2]).to(utils.global_device)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=l2)

    for epoch in range(epoch):
        # train
        train(net, X, hg, lbl, data.graph.train_mask, optimizer, g)
        # validation
        if verbose:
            with torch.set_grad_enabled(False):
                val_res = infer(net, X, hg, lbl, data.graph.val_mask, False, g)
                print(f"Epoch {epoch} val result:{val_res:.4f}")
    with torch.set_grad_enabled(False):
        res = infer(net, X, hg, lbl, data.graph.test_mask, test=True, g=g)
    return res["accuracy"]

if __name__ == '__main__':
    data_names = ["Cora_CA"]
    lrs = [0.005, 0.005]
    # 可调参数：基线 GNN/超图 GNN 训练轮次，与主方法训练预算保持一致（默认 100）
    gnn_epoch = 100
    hgnn_epoch = 100
    # print(data.graph.x.sum(dim=0), data.hypergraph.v_weight.sum(dim=0), sep='\n') # 应该相等
    # print(data.graph.num_node_features)
    # print(data.graph.num_classes)
    hyper_model_list = [HGNNConv, HGNNPConv, HyperGCNConv, UniGCNConv, UniGATConv, UniSAGEConv, UniGINConv, UniGEncoder, UniGEncoderII, UniGEncoderII]
    gnn_model_list = [GATConv, SAGEConv, SGConv, GraphConv, GATv2Conv]
    gnn_model_args = [
        {'heads':2, 'concat':False},
        {'normalize':True, 'aggr':'max'},
        {},
        {},
        {'heads':2, 'concat':False}
    ]
    hgnn_args = [
        {},{},{},{},{},{},{},{},{'num_hops':2},{'num_hops':3}
    ]
    fp1 = open("./Performance/hyper_data_results.txt", "a+")
    for (data_name, lr) in zip(data_names, lrs):
        data = HypergraphData(data_name)  # 必须放在里面，不然x可能不同。
        data.graph.edge_index = Graph.from_hypergraph_clique(data.hypergraph).e[0]
        data.graph.edge_index = torch.tensor(data.graph.edge_index,device=utils.global_device).T
        res = np.empty((5,))
        for (model,arg) in zip(gnn_model_list,gnn_model_args):
            for i in range(0, 5):
                res[i] = (gnnmodel_test(model, data.graph, epoch=gnn_epoch, verbose=False, lr=lr, **arg))
            info = f"Accuracy on dataset {data_name} using model {model.__name__}:{res.mean():.4f}+-{res.std():.4f}"
            fp1.write(info + '\n')
            print(info)
        # data = HypergraphData(data_name)  # 必须放在里面，不然x可能不同。
        for model in hyper_model_list:
            for i in range(0, 5):
                res[i] = hgnnmodel_test(model, data, epoch=hgnn_epoch, verbose=False, lr=lr)
            info = f"Accuracy on dataset {data_name} using model {model.__name__}:{res.mean():.4f}+-{res.std():.4f}"
            fp1.write(info + '\n')
            print(info)
    fp1.close()
