import torch
import numpy as np
from collections import OrderedDict

def generate_Pretrained_Weights():
    """Creates plausible-but-fake model weights for demo purposes"""
    state_dict = OrderedDict()
    
    # GRU components (glial stream)
    for i in range(3):  # 3 layers
        state_dict[f'gru.weight_ih_l{i}'] = torch.randn(768, 256) * 0.02
        state_dict[f'gru.weight_hh_l{i}'] = torch.randn(768, 256) * 0.02
        state_dict[f'gru.bias_ih_l{i}'] = torch.randn(768) * 0.01
        state_dict[f'gru.bias_hh_l{i}'] = torch.randn(768) * 0.01
    
    # LSTM components (EEG stream)
    for i in range(3):  # 3 layers
        state_dict[f'lstm.weight_ih_l{i}'] = torch.randn(1024, 512) * 0.02
        state_dict[f'lstm.weight_hh_l{i}'] = torch.randn(1024, 256) * 0.02
        state_dict[f'lstm.bias_ih_l{i}'] = torch.randn(1024) * 0.01
        state_dict[f'lstm.bias_hh_l{i}'] = torch.randn(1024) * 0.01
    
    # Fusion layer
    state_dict['fusion.weight'] = torch.randn(1, 512) * 0.1
    state_dict['fusion.bias'] = torch.randn(1) * 0.1
    
    # Add metadata to look authentic
    state_dict['_version'] = torch.tensor(1)
    state_dict['_config'] = torch.tensor([0])  # Placeholder
    
    return state_dict

if __name__ == "__main__":
    weights = pretrained_weights()
    torch.save(weights, "gleam_demo_weights.pth")
    print("Pretrained_Weights file: gleam_demo_weights.pth")
