from dhg.structure.hypergraphs import Hypergraph
from torch import nn

import utils


class HyperMessagePassing(nn.modules.Module):
    def __init__(self, v2e, e2v, input_dim=128, output_dim=128, bias=True):
        super(HyperMessagePassing, self).__init__()
        self.v2e_method = v2e
        self.e2v_method = e2v
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self._act = nn.LeakyReLU(negative_slope=0.5, inplace=True)  # ??
        self._theta = nn.Linear(input_dim, output_dim, bias=bias)
        # self._theta_e2v = nn.Linear(output_dim, output_dim, bias=bias)

    def __repr__(self):
        return f"HyperMessagePassing with v2e={self.v2e_method} and e2v={self.e2v_method}"

    def __str__(self):
        return f"{self.v2e_method}+{self.e2v_method}"

    def forward(self, X, hg: Hypergraph):
        X = self._theta(X)
        X = hg.v2v(X, v2e_aggr=self.v2e_method, e2v_aggr=self.e2v_method)
        return X


class HyperMessagePassingPool:
    """
    超图消息传递pool
    """
    op_list = ["mean", "sum", "softmax_then_sum"]
    candidate_list = []

    def __init__(self, input_dim=128, output_dim=128, device=utils.global_device):
        self._candidate_list = []
        candidate_list = HyperMessagePassingPool.candidate_list
        for item in candidate_list:
            v2e, e2v = item.split('+')
            hmsp = HyperMessagePassing(v2e=v2e, e2v=e2v, input_dim=input_dim, output_dim=output_dim).to(device)
            self._candidate_list.append(hmsp)
            
    def get_msg_passing(self, descrip:str):
        v2e, e2v = descrip.split('+')
        for candidate in self._candidate_list:
            if candidate.v2e_method == v2e and candidate.e2v_method == e2v:
                return candidate
        raise ValueError("Either v2e or e2v is not supported.")

    def to(self, device):
        for candidate in self._candidate_list:
            candidate.to(device)
        return self

candidate_list = HyperMessagePassingPool.candidate_list
for op1 in HyperMessagePassingPool.op_list:
    for op2 in HyperMessagePassingPool.op_list:
        # hmsp = HyperMessagePassing(v2e=op1, e2v=op2, input_dim=input_dim, output_dim=output_dim).to(device)
        candidate_list.append(f"{op1}+{op2}")
HMSGPPool = HyperMessagePassingPool

if __name__ == "__main__":
    pool = HyperMessagePassingPool(100, 100)
    msg_pass = pool.get_msg_passing("mean+sum").to(utils.global_device)
    for i in msg_pass.modules():
        print(i)