import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from module import MLP

class TimingGCN(nn.Module):
    def __init__(self):
        super().__init__()

