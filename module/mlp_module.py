import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,
            input_dim:int,
            latent_dim:int,
            output_dim:int
        ):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(in_features=input_dim,out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim,out_features=output_dim)
        )
    def forward(self,input:torch.Tensor):
        return self.mlp(input) 