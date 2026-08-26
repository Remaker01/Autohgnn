import torch
import numpy as np
import torch.nn.functional as F
from search_space.mlp import MLP
from search_space import ActPool, NormPool, HMSGPPool

class GNNBuildWithArchitecture(torch.nn.Module):

    def __init__(self,
                 num_node_features,
                 num_classes,
                 hidden_dimension,
                 node_element_dropout_probability,
                 edge_dropout_probability,
                 architecture,
                 fusion="sum"):

        super(GNNBuildWithArchitecture, self).__init__()
        if fusion not in ("sum", "max", "mean", "concat", "skip", None):
            raise ValueError(f"fusion method {fusion} is not supported yet")
        self.fusion = fusion

        self.layer1_act_pool = ActPool()
        self.layer2_act_pool = ActPool()

        self.node_element_dropout_probability = node_element_dropout_probability
        self.edge_dropout_probability = edge_dropout_probability

        # build new gnn model based on gnn architecture
        # self.pre_process_mlp = MLP(input_dim=num_node_features,
        #                            output_dim=hidden_dimension)

        self.post_process_mlp = MLP(input_dim=hidden_dimension * 2 if fusion == "concat" else hidden_dimension,
                                    output_dim=num_classes)

        self.layer1_conv = HMSGPPool(num_node_features, hidden_dimension).get_msg_passing(architecture[0])
        self.layer1_norm = NormPool(hidden_dimension).get_norm(architecture[1])
        self.layer1_act = self.layer1_act_pool.get_act(architecture[2])

        self.layer2_conv = HMSGPPool(hidden_dimension, hidden_dimension).get_msg_passing(architecture[3])
        self.layer2_norm = NormPool(hidden_dimension).get_norm(architecture[4])
        self.layer2_act = self.layer2_act_pool.get_act(architecture[5])

    def forward(self, x, hg):
        # x = self.pre_process_mlp(x)
        # hg = hg.drop_hyperedges(self.edge_dropout_probability)

        x = self.layer1_conv(x, hg)
        x = self.layer1_norm(x)
        x = self.layer1_act(x)
        x = F.dropout(x, p=self.node_element_dropout_probability, training=self.training)

        # hg = hg.drop_hyperedges(self.edge_dropout_probability)
        if self.fusion != "skip":
            x0 = x.clone() # 待会接到第二层上
            x = self.layer2_conv(x, hg)
            x = self.layer2_norm(x)
            x = self.layer2_act(x)
            if self.fusion == "sum":
                x = torch.add(x, x0)
            elif self.fusion == "mean":
                x = torch.div(torch.add(x,x0), 2)
            elif self.fusion == "max":
                x = torch.max(x, x0)
            elif self.fusion == "concat":
                x = torch.cat((x, x0), -1)
            x = F.dropout(x, p=self.node_element_dropout_probability, training=self.training)

        x = self.post_process_mlp(x)

        return x

class MultiTrailEvaluation(object):

    def __init__(self, gnn_model_config, graph, device, hypergraph=None, fusion="sum"):

        self.num_node_features = gnn_model_config["num_node_features"]
        self.num_classes = gnn_model_config["num_classes"]
        self.hidden_dimension = gnn_model_config["hidden_dimension"]
        self.node_element_dropout_probability = gnn_model_config["node_element_dropout_probability"]
        self.edge_dropout_probability = gnn_model_config["edge_dropout_probability"]
        self.learn_rate = gnn_model_config["learn_rate"]
        self.weight_decay = gnn_model_config["weight_decay"]
        self.train_epoch = gnn_model_config["train_epoch"]
        self.graph = graph
        self.hypergraph = hypergraph
        self.batch_number = len(self.graph)
        self.device = device
        self.fusion = fusion
    
    def get_estimation_score(self, architecture):

        # 构建GNN模型
        gnn_model = GNNBuildWithArchitecture(num_node_features=self.num_node_features,
                                             num_classes=self.num_classes,
                                             hidden_dimension=self.hidden_dimension,
                                             node_element_dropout_probability=self.node_element_dropout_probability,
                                             edge_dropout_probability=self.edge_dropout_probability,
                                             architecture=architecture).to(self.device)

        optimizer = torch.optim.Adam(gnn_model.parameters(),
                                     lr=self.learn_rate,
                                     weight_decay=self.weight_decay)

        loss_f = F.cross_entropy

        # GNN模型训练
        gnn_model.train()
        sub_graph = self.graph
        for epoch in range(self.train_epoch):

                y_pred = gnn_model(sub_graph.x, self.hypergraph if self.hypergraph is not None else sub_graph.edge_index)

                loss = loss_f(y_pred[sub_graph.train_mask],
                              sub_graph.y[sub_graph.train_mask])/self.batch_number
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # GNN模型评估
        gnn_model.eval()
        sum_val_acc = []

        y_pred = gnn_model(sub_graph.x, sub_graph.edge_index)
        pred = y_pred.argmax(dim=1)

        correct_val = pred[sub_graph.val_mask] == sub_graph.y[sub_graph.val_mask]
        if int(sub_graph.val_mask.sum()) > 0:
            sub_val_acc = int(correct_val.sum()) / int(sub_graph.val_mask.sum())
            sum_val_acc.append(sub_val_acc)

        val_acc = sum(sum_val_acc)

        return val_acc

    def get_best_validation_estimation(self, architecture):
        torch.cuda.empty_cache()
        # 构建GNN模型
        gnn_model = GNNBuildWithArchitecture(num_node_features=self.num_node_features,
                                             num_classes=self.num_classes,
                                             hidden_dimension=self.hidden_dimension,
                                             node_element_dropout_probability=self.node_element_dropout_probability,
                                             edge_dropout_probability=self.edge_dropout_probability,
                                             architecture=architecture).to(self.device)

        optimizer = torch.optim.Adam(gnn_model.parameters(),
                                     lr=self.learn_rate,
                                     weight_decay=self.weight_decay)

        loss_f = F.cross_entropy

        best_val_acc = 0
        sub_graph = self.graph
        # GNN模型训练
        for epoch in range(self.train_epoch):
            # 训练
            gnn_model.train()
            
            y_pred = gnn_model(sub_graph.x, self.hypergraph)

            loss = loss_f(y_pred[sub_graph.train_mask],
                          sub_graph.y[sub_graph.train_mask])/self.batch_number

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # GNN模型评估
            gnn_model.eval()
            val_iter = 0
            sum_val_acc = []
            y_pred = gnn_model(sub_graph.x, self.hypergraph)
            pred = y_pred.argmax(dim=1)

            correct_val = pred[sub_graph.val_mask] == sub_graph.y[sub_graph.val_mask]
            if int(sub_graph.val_mask.sum()) > 0:
                sub_val_acc = int(correct_val.sum()) / int(sub_graph.val_mask.sum())
                sum_val_acc.append(sub_val_acc)
                val_iter += 1

            val_acc = np.sum(sum_val_acc) / val_iter

            if best_val_acc < val_acc:
                best_val_acc = val_acc

        return best_val_acc

    def get_test_score(self, architecture):
        # 构建GNN模型
        gnn_model = GNNBuildWithArchitecture(num_node_features=self.num_node_features,
                                             num_classes=self.num_classes,
                                             hidden_dimension=self.hidden_dimension,
                                             node_element_dropout_probability=self.node_element_dropout_probability,
                                             edge_dropout_probability=0.0,
                                             architecture=architecture,
                                             fusion=self.fusion).to(self.device)

        optimizer = torch.optim.Adam(gnn_model.parameters(),
                                     lr=self.learn_rate,
                                     weight_decay=self.weight_decay)

        loss_f = F.cross_entropy

        best_test = 0
        for epoch in range(self.train_epoch):

            sub_graph = self.graph

            # GNN模型训练
            gnn_model.train()
            y_pred = gnn_model(sub_graph.x, self.hypergraph)

            loss = loss_f(y_pred[sub_graph.train_mask],
                            sub_graph.y[sub_graph.train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # GNN模型测试
            gnn_model.eval()
            y_pred = gnn_model(sub_graph.x, self.hypergraph)
            pred = y_pred.argmax(dim=1)
            correct_test = pred[sub_graph.test_mask] == sub_graph.y[sub_graph.test_mask]
            test_acc = int(correct_test.sum()) / int(sub_graph.test_mask.sum())

            if test_acc > best_test:
                best_test = test_acc

        return best_test

    def rank_based_estimation_score(self, gnn_list, val_score_list, top_k):

        gnn_dict = {}

        for key, value in zip(gnn_list, val_score_list):
            gnn_dict[str(key)] = value
        rank_gnn_dict = sorted(gnn_dict.items(), key=lambda x: x[1], reverse=True)

        rank_gnn = []
        rank_gnn_val_score = []

        i = 0
        for key, value in rank_gnn_dict:

            if i == top_k:
                break
            else:
                rank_gnn.append(eval(key))
                rank_gnn_val_score.append(value)
                i += 1
        return rank_gnn, rank_gnn_val_score