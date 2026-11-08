import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from module import MLP

class NetConv(MessagePassing):
    def __init__(self,
            input_dim:int,
            latent_dim:int,
            output_dim:int
        ):
        super().__init__(aggr=None)




class NetEmbedding(nn.Module):
    def __init__(self):
        super().__init__()


