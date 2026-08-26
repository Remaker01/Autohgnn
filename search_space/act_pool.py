import torch.nn.functional as F
from torch.nn.functional import softplus


class ActPool:
    candidate_list = ["Elu", "LeakyRelu",
                           "Relu", "Relu6",
                           "Sigmoid", "Softplus",
                           "Tanh", "Linear"]

    def __init__(self):
        super(ActPool, self).__init__()

    def get_act(self, act_name):

        if act_name == "Elu":
            act = F.elu
        elif act_name == "LeakyRelu":
            act = F.leaky_relu
        elif act_name == "Relu":
            act = F.relu
        elif act_name == "Relu6":
            act = F.relu6
        elif act_name == "Sigmoid":
            act = F.sigmoid
        elif act_name == "Softplus":
            act = F.softplus
        elif act_name == "Tanh":
            act = F.tanh
        elif act_name == "Linear":
            # act = Linear()
            act = lambda x: x
        else:
            raise Exception("Sorry current version don't "
                            "Support this default act", act_name)
        return act

