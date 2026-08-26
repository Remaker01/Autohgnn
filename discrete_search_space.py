import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__)))) # 测试加上这个
import os
from search_space import HMSGPPool, NormPool, ActPool

gnn_topology = ["Convolution", "Normalization", "Activation",
                "Convolution", "Normalization", "Activation"]

msg_candidate = HMSGPPool().candidate_list
norm_candidate = NormPool().candidate_list
act_candidate = ActPool().candidate_list

component_candidate_dict = {"Convolution": msg_candidate,
                            "Normalization": norm_candidate,
                            "Activation": act_candidate}

def search_space_candidate(gnn_candidates_path):
    print("Obtaining Discrete Search Space")
    gnn_architecture_candidates = []

    if path.exists(gnn_candidates_path):
        print("Read Discrete Search Space From:", gnn_candidates_path)
        try:
            with open(gnn_candidates_path, "r") as f:
                gnn_architecture_candidates = [
                    line.strip().split(" ") 
                    for line in f.readlines() 
                    if line.strip()
                ]
        except (IOError, OSError) as e:
            print(f"Error reading file: {e}")
            return []
    else:
        print("Create Discrete Search Space")
        try:
            all_candidates = []
            for layer_1_conv in msg_candidate:
                for layer_1_norm in norm_candidate:
                    for layer_1_act in act_candidate:
                        for layer_2_conv in msg_candidate:
                            for layer_2_norm in norm_candidate:
                                for layer_2_act in act_candidate:
                                    gnn_archi = ' '.join([
                                        layer_1_conv, layer_1_norm, layer_1_act,
                                        layer_2_conv, layer_2_norm, layer_2_act
                                    ])
                                    all_candidates.append(gnn_archi)
            
            with open(gnn_candidates_path, "w") as f:
                f.write('\n'.join(all_candidates) + '\n')
            
            gnn_architecture_candidates = [c.split(" ") for c in all_candidates]
            print("Discrete Search Space Create Completion")
        except (IOError, OSError) as e:
            print(f"Error creating file: {e}")
            return []
    
    print("Discrete Search Space Obtained")
    return gnn_architecture_candidates

search_space_path = "hyper_search_space_gnn_candidates.txt"
search_space = search_space_candidate(search_space_path)

if __name__=="__main__":
    pass
