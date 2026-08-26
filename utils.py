import torch
import torch.nn.functional as F
from dhg.structure import Hypergraph
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
global_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def deprecated(func):
    import warnings, functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(f"'{func.__name__}' is deprecated", DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return wrapper

def pyg_to_hypergraph(pyg_G:Data,khop=2,device=None):
    assert isinstance(khop, (int, bool))
    if device is None:
        device = global_device
    if khop in (False, 0):
        print("Converting graph to hypergraph...")
        return Hypergraph(num_v=pyg_G.num_nodes, e_list=pyg_G.edge_index.T.tolist(), device=device)
    # print("Converting graph format...")
    # dhg_G = Graph(num_v=pyg_G.num_nodes,e_list=pyg_G.edge_index.T.to(device),device=device)
    print("Converting graph to hypergraph...")
    return Hypergraph.from_graph_kHop(pyg_G, khop, device=device)

def hypergraph_structure_aware_distance(hg:Hypergraph,Y:torch.Tensor,Y_hat:torch.Tensor, num_classes=-1, eps=1e-6):
    assert hg.device == Y.device and Y.device == Y_hat.device
    if num_classes == -1:
        num_classes = Y.max() + 1
    # Number of nodes
    n = hg.H.size(0)

    # Convert labels to one-hot encoding (n_nodes x k)
    Y_onehot     = F.one_hot(Y, num_classes).float() if Y.ndim == 1 else Y.float()
    Yhat_onehot  = F.one_hot(Y_hat, num_classes).float() if Y_hat.ndim == 1 else Y_hat.float()

    # Build hypergraph adjacency-like matrix Theta = H W D_e^{-1} H^T
    H = hg.H  # (n x m)
    
    # Prepare hyperedge weight and inverse degree matrices
    # If provided as vectors, convert to diagonal matrices
    def to_diag(mat):
        if mat.ndim == 1:
            return torch.diag(mat)
        return mat

    W_e       = to_diag(hg.W_e)
    D_e_inv   = to_diag(hg.D_e_neg_1)

    # Theta: (n x n)
    if not hasattr(hg, "theta"):
        hg.theta = H @ W_e @ D_e_inv @ hg.H_T
    Theta = hg.theta

    # All-ones matrix E of shape (n x k)
    E = torch.ones((n, num_classes), device=H.device, dtype=H.dtype)
    _cache = Y_onehot.t() @ Theta
    # Compute S and S_hat, both (k x k)
    S     = (_cache @ Y_onehot) / (_cache @ E + eps)
    _cache = Yhat_onehot.t() @ Theta
    S_hat = (_cache @ Yhat_onehot) / (_cache @ E + eps)

    # Return Frobenius norm of their difference
    return torch.norm(S - S_hat, p="fro").item() / torch.norm(S, p="fro").item()

if __name__ == "__main__":
    torch.random.manual_seed(42)

    cora = Planetoid(root='D:/graph_datas',name='Cora').to(global_device)
    hg = pyg_to_hypergraph(cora[0])
    y = F.one_hot(cora[0].y)
    y_hat = y + (torch.rand_like(y.float())/5)
    y_hat = F.softmax(y_hat,dim=1)
    print(y_hat)
    print(hypergraph_structure_aware_distance(hg,Y=y,Y_hat=y_hat,num_classes=cora.num_classes))