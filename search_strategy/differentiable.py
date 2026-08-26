import _warnings
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils

class ArchitectureGradientOptimizer(torch.nn.Module):
    def __init__(self, supernet, lr, decay, concat=False, skip=False, device=utils.global_device):

        super(ArchitectureGradientOptimizer, self).__init__()

        # build learnable architecture alpha parameter based on supernet.
        self.architecture_alpha_list = []
        self.concat = concat
        self.skip = skip

        for component, candidates in supernet.component_candidate_dict.items():

            architecture_alpha = Variable(torch.Tensor(supernet.num_gnn_layer, len(candidates))).to(device)
            architecture_alpha.requires_grad = True
            nn.init.uniform_(architecture_alpha)

            self.architecture_alpha_list.append(architecture_alpha)

        # optimizer for learnable architecture alpha parameter
        self.optimizer = torch.optim.Adam(self.architecture_alpha_list,
                                          lr=lr,
                                          weight_decay=decay)

        # loss function for learnable architecture alpha parameter
        self.loss = F.cross_entropy
    def sample_gnn_model_build(self, sample_architecture, supernet):
        """
        passing current supernet and construct the gnn model based the gnn architecture sampled by gumbel softmax
        """

        # self.pre_process_mlp = supernet.pre_process_mlp
        self.post_process_mlp = supernet.post_process_mlp

        self.layer1_conv = supernet.supernet_operation_pool[0].get_candidate(sample_architecture[0])
        self.layer1_norm = supernet.supernet_operation_pool[1].get_candidate(sample_architecture[1])
        self.layer1_act = supernet.supernet_operation_pool[2].get_candidate(sample_architecture[2])

        self.layer2_conv = supernet.supernet_operation_pool[3].get_candidate(sample_architecture[3])
        self.layer2_norm = supernet.supernet_operation_pool[4].get_candidate(sample_architecture[4])
        self.layer2_act = supernet.supernet_operation_pool[5].get_candidate(sample_architecture[5])
    def forward(self,
                x,
                hg,
                gumbel_softmax_sample_ret_list,
                sample_candidate_index_list):

        # add architecture alpha parameter to Torch Computation Graph of sampled gnn model for getting validation gradient
        # x = self.pre_process_mlp(x)
        x = self.layer1_conv(x, hg) * gumbel_softmax_sample_ret_list[0][0][sample_candidate_index_list[0]]
        x = self.layer1_norm(x) * gumbel_softmax_sample_ret_list[1][0][sample_candidate_index_list[1]]
        x = self.layer1_act(x) * gumbel_softmax_sample_ret_list[2][0][sample_candidate_index_list[2]]

        if not self.skip:
            x = self.layer2_conv(x, hg) * gumbel_softmax_sample_ret_list[0][1][sample_candidate_index_list[3]]
            x = self.layer2_norm(x) * gumbel_softmax_sample_ret_list[1][1][sample_candidate_index_list[4]]
            x = self.layer2_act(x) * gumbel_softmax_sample_ret_list[2][1][sample_candidate_index_list[5]]

            if self.concat:
                x0 = x.clone()
                x = torch.concat((x, x0), dim=-1)
        x = self.post_process_mlp(x)

        return x
    def forward_test(self, x:torch.Tensor, hg):

        # add architecture alpha parameter to Torch Computation Graph of sampled gnn model for getting validation gradient
        # x = self.pre_process_mlp(x)
        x = self.layer1_conv(x, hg)
        x = self.layer1_norm(x)
        x = self.layer1_act(x)

        if not self.skip:
            x = self.layer2_conv(x, hg)
            x = self.layer2_norm(x)
            x = self.layer2_act(x)

            if self.concat:
                x0 = x.clone()
                x = torch.concat((x, x0), dim=-1)
        x = self.post_process_mlp(x)

        return x

class DifferentiableSearch(object):
    def __init__(self,
                 supernet,
                 graph,
                 hypergraph,
                 differentiable_search_optimizer_config_dict,
                 temperature,
                 concat=False,
                 skip=False,
                 device=None):

        if skip and concat:
            _warnings.warn("Attribute concat is ignored because skip=True",stacklevel=2)
            concat = False
        if device:
            self.device = device
        else:
            self.device = utils.global_device

        self.differentiable_search_pruning_best_gnn_history = []
        self.graph = graph
        self.hypergraph = hypergraph
        self.temperature = temperature
        self.best_architecture_history = []
        self.architecture_gradient_optimizer = ArchitectureGradientOptimizer(supernet,
                                                                             differentiable_search_optimizer_config_dict["lr"],
                                                                             differentiable_search_optimizer_config_dict["decay"],
                                                                             concat=concat,
                                                                             skip=skip,
                                                                             device=self.device)

    def search(self,
               supernet,
               search_paths,
               search_epoch,
               return_top_k):

        print("Differentiable Search Starting")
        differentiable_pruning_search_path = []

        # get the architecture alpha parameter sample distribution for gumbel softmax sample
        architecture_alpha_list = self.architecture_gradient_optimizer.architecture_alpha_list

        for epoch in range(search_epoch):
            if epoch % 5 == 0:
                print(32 * "=")
                print("Search Epoch:", epoch)
            gumbel_softmax_sample_output_list = []

            for architecture_alpha in architecture_alpha_list:
                gumbel_softmax_sample_output_list.append(self.hard_gumbel_softmax_sample(F.softmax(architecture_alpha, dim=-1)))

            # pruning search space sample constraint
            re_sample, \
            sample_candidate_index_list, \
            sample_architecture = self.sample_gnn_architecture_check(gumbel_softmax_sample_output_list,
                                                                     supernet,
                                                                     search_paths,
                                                                     epoch % 5 == 0)

            while re_sample:

                gumbel_softmax_sample_output_list = []

                for architecture_alpha in architecture_alpha_list:
                    gumbel_softmax_sample_output_list.append(self.hard_gumbel_softmax_sample(F.softmax(architecture_alpha, dim=-1)))

                re_sample, \
                sample_candidate_index_list, \
                sample_architecture = self.sample_gnn_architecture_check(gumbel_softmax_sample_output_list,
                                                                         supernet,
                                                                         search_paths,
                                                                         epoch % 5 == 0)

            # save the gnn architecture searched by differentiable search
            differentiable_pruning_search_path.append(sample_architecture)

            # architecture alpha parameter optimization based on the sampled gnn model using the validation gradient
            self.architecture_gradient_optimizer.train()
            self.architecture_gradient_optimizer.sample_gnn_model_build(sample_architecture, supernet)

            # architecture alpha parameter optimization
            y_pred = self.architecture_gradient_optimizer(self.graph.x,
                                                          self.hypergraph,
                                                          gumbel_softmax_sample_output_list,
                                                          sample_candidate_index_list)

            loss = self.architecture_gradient_optimizer.loss(y_pred[self.graph.val_mask],
                                                             self.graph.y[self.graph.val_mask])

            self.architecture_gradient_optimizer.optimizer.zero_grad()
            loss.backward()
            self.architecture_gradient_optimizer.optimizer.step()

            best_gnn = self.best_alpha_gnn_architecture(self.architecture_gradient_optimizer.architecture_alpha_list,
                                                        supernet)
            time.sleep(0.2)
            if epoch % 5 == 0:
                print(f"Best GNN Architecture:{best_gnn}")

        print(32 * "=")

        print("differentiable Search Ending")
        # best_alpha_gnn_architecture_list里面存的是上面搜索出来的架构。
        if int(return_top_k) <= len(self.best_architecture_history):
            best_alpha_gnn_architecture_list = self.best_architecture_history[-int(return_top_k):]
        else:
            best_alpha_gnn_architecture_list = self.best_architecture_history

        for gnn in best_alpha_gnn_architecture_list:
            if gnn not in self.differentiable_search_pruning_best_gnn_history:
                self.differentiable_search_pruning_best_gnn_history.append(gnn)

        if len(self.differentiable_search_pruning_best_gnn_history) > int(return_top_k):
            # 下面根据那个distance选择最优架构
            print("Selecting best architectures...")
            distances = {}
            with torch.set_grad_enabled(False):
                for (i, sample_architecture) in enumerate(self.differentiable_search_pruning_best_gnn_history):
                    # 由于参数是共享的，不用担心由字符串初始化出来的架构没有训练过。
                    if sample_architecture[0] == "sum+sum" or sample_architecture[3] == "sum+sum" or (sample_architecture[1] == "PairNorm" and sample_architecture[4] == "PairNorm"):
                        distances.update({i: 999.99})
                    else:
                        self.architecture_gradient_optimizer.sample_gnn_model_build(sample_architecture, supernet)
                        self.architecture_gradient_optimizer.eval()
                        y_pred = self.architecture_gradient_optimizer.forward_test(self.graph.x, self.hypergraph)
                        distance = utils.hypergraph_structure_aware_distance(self.hypergraph, Y=self.graph.y, Y_hat=y_pred)
                        distances.update({i: distance})
                # 1. 将字典按值排序
                sorted_items = sorted(distances.items(), key=lambda x: x[1], reverse=False)
                # 2. 获取前k个键
                top_k_keys = [item[0] for item in sorted_items[:return_top_k]]
                # 3. 从列表l中取出对应下标的元素
                self.differentiable_search_pruning_best_gnn_history = [self.differentiable_search_pruning_best_gnn_history[key] for key in top_k_keys]

                # 选出最优的几个
            # self.differentiable_search_pruning_best_gnn_history = self.differentiable_search_pruning_best_gnn_history[-return_top_k:]

        print("differentiable Search Final Output Best GNN Architectures:")

        for gnn in self.differentiable_search_pruning_best_gnn_history:
            print(gnn)
        
        return self.differentiable_search_pruning_best_gnn_history, differentiable_pruning_search_path
    def hard_gumbel_softmax_sample(self, sample_probability):

        hard_gumbel_softmax_sample_output = F.gumbel_softmax(logits=sample_probability,
                                                             tau=self.temperature,
                                                             hard=True)
        return hard_gumbel_softmax_sample_output
    def sample_gnn_architecture_check(self, gumbel_softmax_sample_ret_list, supernet, search_paths, output=True):

        candidate_list = []
        candidate_index_list = []

        for component_one_hot, component in zip(gumbel_softmax_sample_ret_list, supernet.component_candidate_dict.keys()):

            for candidate_one_hot in component_one_hot.detach():
                candidate_index = candidate_one_hot.argmax().item()
                candidate_list.append(supernet.component_candidate_dict[component][candidate_index])
                candidate_index_list.append(candidate_index)

        sample_architecture = candidate_list[::2] + candidate_list[1::2]
        sample_candidate_index = candidate_index_list[::2] + candidate_index_list[1::2]

        if sample_architecture not in search_paths:
            re_sample = True
        else:
            if output:
                print("Gumbel Softmax Sample GNN Architecture:", sample_architecture)
            re_sample = False

        return re_sample, sample_candidate_index, sample_architecture
    def best_alpha_gnn_architecture(self, architecture_alpha_list, supernet):

        best_alpha_architecture = []

        for architecture_alpha_vector_list, component in zip(architecture_alpha_list,
                                                             supernet.component_candidate_dict.keys()):
            # architecture_alpha_vector_list:[层数*某种组件（msgpass, norm, act）的选择数]，里面是每一层该组件的所有候选的alpha
            for architecture_alpha_vector in architecture_alpha_vector_list.detach():
                best_alpha_index = architecture_alpha_vector.argmax().item() #architecture_alpha_vector.index(max(architecture_alpha_vector))
                best_alpha_architecture.append(supernet.component_candidate_dict[component][best_alpha_index])

        gnn_architecture = best_alpha_architecture[::2] + best_alpha_architecture[1::2]

        if gnn_architecture not in self.best_architecture_history:
            self.best_architecture_history.append(gnn_architecture)

        return gnn_architecture

if __name__=="__main__":

    pass
