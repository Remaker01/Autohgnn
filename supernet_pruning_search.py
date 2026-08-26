# 同时保存图和超图结构，需要结构就是超图结构，需要x,y等就从原来数据里取
# 目前performance_test_with_hpo, search_from_scratch, scalable_gnn等均未使用，后续如果需要记得在这些文件里加上修改部分比如跳跃连接等。
import time

import torch.cuda
from torch import nn
import numpy as np
from mixed_supernet import MixedSuperNet
from graph_data import GraphData
from dhg.structure.hypergraphs import Hypergraph
from search_strategy.differentiable import DifferentiableSearch
import utils, hyper_data
from mixed_search_space_with_forward import HMSGPPool, NormPool, ActPool

from time import perf_counter
from warnings import warn

class SupernetPruningSearch(object):

    def __init__(self,
                 supernet:MixedSuperNet,
                 loss_f,
                 graph,
                 supernet_config,
                 differentiable_searcher_config,
                 khop=1,
                 device=utils.global_device,
                 hypergraph=None):

        self.supernet = supernet
        self.loss_f = loss_f
        self.graph = graph

        self.warm_up_training_epoch = supernet_config["warm_up_train_epoch"]
        self.single_path_training_sample_size_list = supernet_config["single_path_training_sample_size_list"]

        self.temperature = differentiable_searcher_config["temperature"]
        self.differentiable_search_optimizer_config_dict = differentiable_searcher_config["differentiable_search_optimizer_config_dict"]
        self.differentiable_search_epoch_list = differentiable_searcher_config["differentiable_search_epoch_list"]
        self.differentiable_search_num_return_top_k_gnn = differentiable_searcher_config["differentiable_search_num_return_top_k_gnn"]
        if hypergraph is None or not isinstance(hypergraph, Hypergraph):
            self.hypergraph = utils.pyg_to_hypergraph(graph, khop=khop)
            torch.cuda.empty_cache()
        else: self.hypergraph = hypergraph
        self.differentiable_searcher = DifferentiableSearch(supernet=self.supernet,
                                                            graph=self.graph,
                                                            hypergraph=self.hypergraph,
                                                            differentiable_search_optimizer_config_dict=self.differentiable_search_optimizer_config_dict,
                                                            temperature=self.temperature,
                                                            concat=self.supernet.fusion == "concat",
                                                            skip=self.supernet.fusion == "skip",
                                                            device=device)

    def warm_up_training(self):

        warn("Deprecated.", DeprecationWarning)
        self.supernet.train()
        for epoch in range(self.warm_up_training_epoch):
            y_pre = self.supernet.mixed_forward(self.graph.x, self.hypergraph)

            loss = self.loss_f(y_pre[self.graph.train_mask],
                               self.graph.y[self.graph.train_mask])

            self.supernet.operation_weight_optimizer.zero_grad()
            loss.backward()
            self.supernet.operation_weight_optimizer.step()

            print("Supernet Warm Up Training Epoch", epoch + 1,
                  "Supernet Weights Total Loss:{:.4f}".format(loss.item()))

    def uniform_random_single_path_sample(self, search_paths, sample_size):
        # Input validation
        if not search_paths:
            raise ValueError("search_paths cannot be empty")
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")

        space_size = len(search_paths)

        if sample_size > space_size:
            raise ValueError(f"sample_size ({sample_size}) cannot be greater than space_size ({space_size}) when replace=False")

        # sample mode is replace=True
        uniform_random_sample_index = np.random.choice(range(space_size), size=sample_size, replace=False)

        # Use list comprehension for better performance and readability
        sampled_architectures = [search_paths[index] for index in uniform_random_sample_index]

        return sampled_architectures

    def single_path_training(self, search_paths, sample_size):

        uniform_random_sample_gnn_architecture_list = self.uniform_random_single_path_sample(search_paths,
                                                                                             sample_size)
        self.supernet.train()
        print("Single path training...")
        for sample_gnn_architecture in uniform_random_sample_gnn_architecture_list:
            self.supernet.single_path_architecture_construction(sample_gnn_architecture)
            y_pre = self.supernet.single_path_forward(self.graph.x, self.hypergraph)
            train_loss = self.loss_f(y_pre[self.graph.train_mask],
                                     self.graph.y[self.graph.train_mask])

            self.supernet.operation_weight_optimizer.zero_grad()
            train_loss.backward()
            self.supernet.operation_weight_optimizer.step()

    @staticmethod
    def whole_search_paths_read():

        dir_ = "./hyper_search_space_gnn_candidates.txt"

        with open(dir_, "r") as f:
            whole_search_paths = f.readlines()
            whole_search_paths = [path.replace("\n", "").split(" ") for path in whole_search_paths]
        return whole_search_paths

    def search(self):

        # self.warm_up_training()
        search_paths = self.whole_search_paths_read()
        top_gnn_list = []
        search_time = 0.0
        # 猜想：d...[0],s...[0]:对应算法step3；d...[1],s...[1]对应算法step4-5。j=d...[1].
        for differentiable_search_epoch, single_path_training_sample_size in zip(self.differentiable_search_epoch_list,
                                                                                 self.single_path_training_sample_size_list):

            self.single_path_training(search_paths=search_paths, sample_size=single_path_training_sample_size)
            start = perf_counter()
            top_gnn_list, search_paths = self.differentiable_searcher.search(supernet=self.supernet,
                                                                             search_paths=search_paths,
                                                                             search_epoch=differentiable_search_epoch,
                                                                             return_top_k=self.differentiable_search_num_return_top_k_gnn)
            search_time += (perf_counter() - start)

        return top_gnn_list, search_time

if __name__=="__main__":
    from performance_test_with_hpo import test_record, hpo
    from multi_trial_gnas.multi_trail_evaluation import MultiTrailEvaluation
    from argparse import ArgumentParser
    parser = ArgumentParser("AutoHGNN search")
    parser.add_argument("--data_name", default="Computers", choices=("Cora_CA", "DBLP", "Pubmed", "Computers", "Physics"))
    parser.add_argument("--fusion", default="mean", choices=("sum", "mean", "max", "concat", "none"))
    
    args = parser.parse_args()
    data_name = args.data_name
    device = utils.global_device
    graph, hypergraph = None, None
    if data_name in ["Cora_CA", "DBLP"]:
        data = hyper_data.HypergraphData(data_name, device)
        graph = data.graph
        hypergraph = data.hypergraph
    elif data_name in ["Pubmed", "Computers", "Physics"]:
        graph = GraphData(data_name, False).data
        hypergraph = utils.pyg_to_hypergraph(graph, khop=1, device=utils.global_device)
    else:
        raise AttributeError(f"{data_name} not supported")
    supernet_dim_config = {"input_dimension": graph.num_node_features,
                           "hidden_dimension": 128,
                           "output_dimension": graph.num_classes,
                           "edge_dropout_probability": 0.3,
                           "node_element_dropout_probability": 0.5}
    fusion = args.fusion
    if fusion == "none":
        fusion = None
    # khop = 1

    operation_candidates_list = [HMSGPPool.candidate_list,
                                 NormPool.candidate_list,
                                 ActPool.candidate_list,
                                 HMSGPPool.candidate_list,
                                 NormPool.candidate_list,
                                 ActPool.candidate_list]

    operation_weight_optimizer_config = {"operation_weight_learn_rate": 0.01,
                                         "operation_weight_weight_decay": 0.0001}

    supernet = MixedSuperNet(supernet_dim_config,
                             operation_weight_optimizer_config,
                             device,
                             fusion=fusion)

    supernet.mixed_supernet_construction_with_operation_candidates(operation_candidates_list) # 构建超网络

    loss_f = nn.functional.cross_entropy

    supernet_config = {"warm_up_train_epoch": 0,
                       "single_path_training_sample_size_list": (100, )*2}

    differentiable_search_optimizer_config_dict = {"lr": 0.1,
                                                   "decay": 0.005}

    differentiable_searcher_config = {"temperature": 0.1,
                                      "differentiable_search_optimizer_config_dict": differentiable_search_optimizer_config_dict,
                                      "differentiable_search_epoch_list": (500,)*2,
                                      "differentiable_search_num_return_top_k_gnn": 5}

    searcher = SupernetPruningSearch(supernet=supernet,
                                     loss_f=loss_f,
                                     graph=graph,
                                     supernet_config=supernet_config,
                                     differentiable_searcher_config=differentiable_searcher_config,
                                     khop=1,
                                     device=device,
                                     hypergraph=hypergraph)
    top_gnn_list, search_time = searcher.search()
    best_val, best_arch, best_hp = 0.0, None, None

    with open(f"./Performance/d2gnas_{data_name}.txt","a+") as f:
        f.write(f"search time per epoch {search_time/sum(differentiable_searcher_config['differentiable_search_epoch_list']):.4f}s\n")
    for gnn_model in top_gnn_list:
        print(f"--------evaluating {gnn_model} on validation set--------")
        hp = hpo(gnn_architecture=gnn_model,
                 graph=graph,
                 hypergraph=searcher.hypergraph,
                 fusion=fusion,
                 gnn_train_epoch=100,
                 tuning_epoch=6,
                 data=data_name,
                 search_strategy="d2gnas"
                 )
        gnn_model_config = {"num_node_features": graph.num_node_features,
                            "num_classes": graph.num_classes,
                            "hidden_dimension": hp["dim"],
                            "learn_rate": hp["lr"],
                            "node_element_dropout_probability": hp["node_drop"],
                            "edge_dropout_probability": 0,
                            "weight_decay": hp["decay"],
                            "train_epoch": 100}
        estimator = MultiTrailEvaluation(gnn_model_config=gnn_model_config,
                                         graph=graph,
                                         hypergraph=searcher.hypergraph,
                                         fusion=fusion,
                                         device=device)
        cur_val = estimator.get_best_validation_estimation(gnn_model)
        print(f"validation accuracy of {gnn_model}: {cur_val:.4f}")
        if cur_val > best_val:
            best_val = cur_val
            best_arch = gnn_model
            best_hp = hp
    print(f"Best arch for {data_name} (selected by validation) is {best_arch} with {best_hp}")

    # 在测试集上评估验证集选出的最优架构
    if best_arch is not None and best_hp is not None:
        test_record(graph=graph,
                    hypergraph=searcher.hypergraph,
                    gnn_architecture=best_arch,
                    learning_rate=best_hp["lr"],
                    weight_decay=best_hp["decay"],
                    edge_dropout_probability=0,
                    node_element_dropout_probability=best_hp["node_drop"],
                    hidden_dimension=best_hp["dim"],
                    gnn_train_epoch=100,
                    test_epoch=10,
                    device=device,
                    data=data_name,
                    fusion=fusion,
                    search_strategy="d2gnas",
                    )