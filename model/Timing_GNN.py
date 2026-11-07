import torch
import torch.nn as nn

class Timing_GNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)