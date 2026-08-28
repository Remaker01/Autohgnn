'''用预先搜索出来的架构进行测试，先不用管'''
import os

import numpy as np
import torch

import utils
from multi_trial_gnas.multi_trail_evaluation import MultiTrailEvaluation

# from scalable_gnn.pasca_v3 import PaScaV3, edge_index_to_sparse_adj
# from scalable_gnn.base_op import x_row_normalization

def test_record(graph,
                gnn_architecture,
                learning_rate,
                weight_decay,
                node_element_dropout_probability,
                edge_dropout_probability,
                gnn_train_epoch,
                hidden_dimension,
                device,
                test_epoch,
                data,
                search_strategy,
                fusion="sum",
                hypergraph=None,
                information=None,
                manner="gnas"):

    gnn_model_config = {"num_node_features": graph.num_node_features,
                        "num_classes": graph.num_classes,
                        "hidden_dimension": hidden_dimension,
                        "learn_rate": learning_rate,
                        "node_element_dropout_probability": node_element_dropout_probability,
                        "edge_dropout_probability": edge_dropout_probability,
                        "weight_decay": weight_decay,
                        "train_epoch": gnn_train_epoch}
    avg_test_acc = []
    # std_test_acc = []

    # cluster_data = ClusterData(graph, num_parts=1)
    # graph_loader = DataLoader(graph, batch_size=1, shuffle=False)
    print("Search Strategy:" + search_strategy + " Dataset:" + data)
    estimator = MultiTrailEvaluation(gnn_model_config=gnn_model_config,
                                     graph=graph,
                                     hypergraph=utils.pyg_to_hypergraph(graph) if hypergraph is None else hypergraph,
                                     device=device,
                                     fusion=fusion)
    for epoch in range(test_epoch):
        torch.cuda.empty_cache()
        # print(information)
        # if manner=="manual":
        #     estimator = MultiTrailEvaluation_manual(gnn_model_config=gnn_model_config,
        #                                              graph=graph,
        #                                              device=device)
        # else:

        score = estimator.get_test_score(gnn_architecture)
        avg_test_acc.append(score)
        print(f"epoch {epoch} finished with score={score:.4f}")
        # std_test_acc.append(score)

    avg_test = np.mean(avg_test_acc)
    std_test = np.std(avg_test_acc)

    dir = "./Performance/"

    if not os.path.exists(dir):
        os.makedirs(dir)

    path = dir + search_strategy + "_" + data + ".txt"
    info = f"{gnn_architecture}: avg test accuracy: {avg_test:.4f} std test accuracy: {std_test:.4f}\n"
    with open(path, "a+") as f:
        f.write(info)
    print(info)
    return avg_test
