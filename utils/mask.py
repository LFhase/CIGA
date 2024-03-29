import torch.nn as nn
from torch import Tensor

from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing

def set_masks(mask: Tensor, model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            #PyG 2.0.4
            module._explain = True
            module._edge_mask = mask
            #PyG 1.7.2
            module.__explain__ = True
            module.__edge_mask__ = mask

def clear_masks(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            #PyG 2.0.4
            module._explain = False
            module._edge_mask = None
            #PyG 1.7.2
            module.__explain__ = False
            module.__edge_mask__ = None
