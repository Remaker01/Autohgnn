import torch
from torch import nn, optim
from torch.nn import functional as F
import utils
from dhg.metrics import VertexClassificationEvaluator as Evaluator

ev = Evaluator(["accuracy"])
class _GNNModule(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.linear = layers[0]
        self.layer0 = layers[1]
        self.layer1 = layers[2]

    def forward(self, X: torch.Tensor, edge_index) -> torch.Tensor:
        x = self.linear(X)
        x = self.layer0(x, edge_index)
        return self.layer1(x, edge_index)

def gnnmodel_test(model, graph, verbose=True, hidden=64,  epoch=100, lr=0.001, l2=1e-4, **kwargs):
    '''图对照实验，返回准确率 '''
    global ev

    def train(net, X, edge_index, lbls, train_mask, optimizer: optim.Optimizer):
        global ev
        net.train()
        optimizer.zero_grad()
        outs = net(X, edge_index)
        outs, lbls = outs[train_mask], lbls[train_mask]
        loss = F.cross_entropy(outs, lbls)
        loss.backward()
        optimizer.step()
        return loss.item()

    def infer(net, X, edge_index, lbls, val_mask, test=False):
        global ev
        net.eval()
        outs = net(X, edge_index)
        outs, lbls = outs[val_mask], lbls[val_mask]
        if not test:
            return ev.validate(lbls, outs)
        return ev.test(lbls, outs)
    kwargs.pop('bias','')
    linear = nn.Linear(graph.x.shape[1], hidden, bias=False).to(utils.global_device)
    layer1 = model(hidden, hidden, bias=False, **kwargs).to(utils.global_device)
    layer2 = model(hidden, graph["num_classes"], bias=False, **kwargs).to(utils.global_device)
    net = _GNNModule([linear, layer1, layer2]).to(utils.global_device)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=l2)

    for epoch in range(epoch):
        # train
        train(net, graph.x, graph.edge_index, graph.y, graph.train_mask, optimizer)
        # validation
        if verbose:
            with torch.set_grad_enabled(False):
                val_res = infer(net, graph.x, graph.edge_index, graph.y, graph.val_mask, False)
                print(f"Epoch {epoch} val result:{val_res:.4f}")
    with torch.set_grad_enabled(False):
        res = infer(net, graph.x, graph.edge_index, graph.y, graph.test_mask, test=True)
        return res["accuracy"]