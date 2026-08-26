import torch
import torch.nn.functional as F
from search_space.mlp import MLP
from mixed_search_space_with_forward import ActPool, NormPool, HMSGPPool, HyperMessagePassing
from dhg.structure import Hypergraph

from graph_data import GraphData

class MixedSuperNet(torch.nn.Module): 

    def __init__(self,
                 supernet_config,
                 operation_weight_optim_config,
                 device,
                 fusion="sum"):

        super(MixedSuperNet, self).__init__()
        self.operation_weight_optimizer = None
        self.mixed_supernet = []
        if fusion not in ("sum", "max", "mean", "concat", "skip", None):
            raise ValueError(f"fusion method {fusion} is not supported yet")
        self.fusion = fusion

        self.input_dimension = supernet_config["input_dimension"]
        self.hidden_dimension = supernet_config["hidden_dimension"]
        self.output_dimension = supernet_config["output_dimension"]
        self.edge_probability = supernet_config["edge_dropout_probability"]
        self.node_element_probability = supernet_config["node_element_dropout_probability"]

        self.operation_weight_learn_rate = operation_weight_optim_config["operation_weight_learn_rate"]
        self.operation_weight_weight_decay = operation_weight_optim_config["operation_weight_weight_decay"]
        self.device = device

        # pre process mlp initialization
        # self.pre_process_mlp = MLP(input_dim=self.input_dimension,
        #                            output_dim=self.hidden_dimension).to(device)
        # post process mlp initialization
        self.post_process_mlp = MLP(input_dim=self.hidden_dimension * 2 if fusion == "concat" else self.hidden_dimension,
                                    output_dim=self.output_dimension).to(device)

        # supernet operation pool construction initialization
        # convolution pool initialization
        layer1_conv_pool = HMSGPPool(self.input_dimension, self.hidden_dimension).to(device)
        layer2_conv_pool = HMSGPPool(self.hidden_dimension, self.hidden_dimension).to(device)

        # normalization pool initialization
        layer1_norm_pool = NormPool(self.hidden_dimension).to(device)
        layer2_norm_pool = NormPool(self.hidden_dimension).to(device)

        # activation pool initialization
        layer1_act_pool = ActPool().to(device)
        layer2_act_pool = ActPool().to(device)

        self.supernet_operation_pool = [layer1_conv_pool, layer1_norm_pool, layer1_act_pool,
                                            layer2_conv_pool, layer2_norm_pool, layer2_act_pool]

        # coupled supernet candidate
        self.msg_candidate = HMSGPPool.candidate_list
        self.norm_candidate = NormPool.candidate_list
        self.act_candidate = ActPool.candidate_list

        self.num_gnn_layer = 2
        self.component_candidate_dict = {"M": self.msg_candidate,
                                         "N": self.norm_candidate,
                                         "A": self.act_candidate}

    def mixed_supernet_construction_with_operation_candidates(self, operation_candidates_list):

        if len(self.mixed_supernet) > 0:
            return

        operation_weights = []

        for operation_candidate, operation_pool in zip(operation_candidates_list,
                                                       self.supernet_operation_pool):

            mix_operation = []

            for operation in operation_candidate:
                operation_obj = operation_pool.get_candidate(operation)
                mix_operation.append(operation_obj)

                if type(operation_pool).__name__ != "ActPool":
                    operation_weights.append({"params": operation_obj.parameters()})

            self.mixed_supernet.append(mix_operation)

        self.operation_weight_optimizer = torch.optim.Adam(operation_weights,
                                                           lr=self.operation_weight_learn_rate,
                                                           weight_decay=self.operation_weight_weight_decay)

    def mixed_forward(self, x, hg: Hypergraph):

        # x = self.pre_process_mlp(x)

        for mix_operation in self.mixed_supernet:
            operation_output_list = []

            for operation in mix_operation:
                if isinstance(operation, HyperMessagePassing):
                    # hg = hg.drop_hyperedges(self.edge_probability)
                    operation_output_list.append(operation(x, hg))
                    # x = F.dropout(x, p=self.node_element_probability, training=self.training)
                elif "Norm" in type(operation).__name__:
                    operation_output_list.append(operation(x))
                else:
                    operation_output_list.append(operation(x))

            # calculate mix operation output
            x = sum(operation_output_list)

        if self.fusion == "concat":
            x = torch.cat((x, x), dim=-1)
        x = self.post_process_mlp(x)

        return x

    def single_path_architecture_construction(self, architecture):


        self.gnn_architecture = architecture

        self.layer1_conv = self.supernet_operation_pool[0].get_candidate(architecture[0])
        self.layer1_norm = self.supernet_operation_pool[1].get_candidate(architecture[1])
        self.layer1_act = self.supernet_operation_pool[2].get_candidate(architecture[2])

        self.layer2_conv = self.supernet_operation_pool[3].get_candidate(architecture[3])
        self.layer2_norm = self.supernet_operation_pool[4].get_candidate(architecture[4])
        self.layer2_act = self.supernet_operation_pool[5].get_candidate(architecture[5])

    def single_path_forward(self, x:torch.Tensor, hg:Hypergraph):
        # x = self.pre_process_mlp(x)

        # hg = hg.drop_hyperedges(self.edge_probability)
        x = self.layer1_conv(x, hg)
        x = self.layer1_norm(x)
        x = self.layer1_act(x)

        # hg = hg.drop_hyperedges(self.edge_probability)
        x = F.dropout(x, p=self.node_element_probability, training=self.training) # 这个算是第一层“输出”

        if self.fusion != "skip":
            x0 = x.clone() # “原始”输入，待会接到第二层上
            x = self.layer2_conv(x, hg)
            x = self.layer2_norm(x)
            x = self.layer2_act(x)
            #下面开始融合
            if self.fusion == "sum":
                x = torch.add(x, x0) # 这里不能+=，参见https://zhuanlan.zhihu.com/p/608556704
            elif self.fusion == "mean":
                x = torch.div(torch.add(x,x0), 2) #同上
            elif self.fusion == "max":
                x = torch.max(x, x0)
            elif self.fusion == "concat":
                x = torch.cat((x, x0), -1)
            x = F.dropout(x, p=self.node_element_probability, training=self.training)

        x = self.post_process_mlp(x)

        return x

if __name__=="__main__":

    data_name = "Computers"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    graph = GraphData(data_name, shuffle=False).data

    supernet_dim_config = {"input_dimension": graph.num_node_features,
                           "hidden_dimension": 128,
                           "output_dimension": graph.num_classes,
                           "edge_dropout_probability": 0.3,
                           "node_element_dropout_probability": 0.5}

    operation_candidates_list = [["GCNConv", "SAGEConv",
                                  "GATConv", "GraphConv",
                                  "TAGConv", "ARMAConv",
                                  "SGConv", "HyperGraphConv",
                                  "ClusterGCNConv"],
                                 ["GraphNorm", "InstanceNorm",
                                  "LayerNorm", "BatchNorm",
                                  "LinearNorm"],
                                 ["Elu", "LeakyRelu",
                                  "Relu", "Relu6",
                                  "Sigmoid", "Softplus",
                                  "Tanh", "Linear"],
                                 ["GCNConv", "SAGEConv",
                                  "GATConv", "GraphConv",
                                  "TAGConv", "ARMAConv",
                                  "SGConv", "HyperGraphConv",
                                  "ClusterGCNConv"],
                                 ["GraphNorm", "InstanceNorm"],
                                 ["Elu", "LeakyRelu",
                                  "Relu", "Relu6",
                                  "Sigmoid", "Softplus",
                                  "Tanh", "Linear"]]


    operation_weight_optim_config = {"operation_weight_learn_rate": 0.01,
                                     "operation_weight_weight_decay": 0.0001}

    my_supernet = MixedSuperNet(supernet_dim_config,
                                operation_weight_optim_config,
                                device)

    my_supernet.mixed_supernet_construction_with_operation_candidates(operation_candidates_list)
    loss_f = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        y_pre = my_supernet.mixed_forward(graph.x, graph.edge_index)

        train_loss = loss_f(y_pre[graph.train_mask],
                            graph.y[graph.train_mask])

        my_supernet.operation_weight_optimizer.zero_grad()
        train_loss.backward()
        my_supernet.operation_weight_optimizer.step()

        print("Train Epoch", epoch+1, "Hypernetwork Weight Loss:", train_loss.item())

    print("Single Path Training")

    for epoch in range(100):

        architecture = ["GCNConv", "GraphNorm", "Relu",
                        "GATConv", "LayerNorm", "Sigmoid"]
        my_supernet.single_path_architecture_construction(architecture)
        y_pre = my_supernet.single_path_forward(graph.x, graph.edge_index)

        train_loss = loss_f(y_pre[graph.train_mask],
                            graph.y[graph.train_mask])

        my_supernet.operation_weight_optimizer.zero_grad()
        train_loss.backward()
        my_supernet.operation_weight_optimizer.step()
        print("Train Epoch", epoch+1, "Architecture Weight Loss:", train_loss.item())