import torch
import torch.nn as nn

class Net_Delay_Prediction_Decoder(nn.Module):
    def __init__(self,
            embed_dim:int=32,
            latent_dim:int=32
        ):
        super().__init__()
        self.decoder=nn.Sequential(
            nn.Linear(
                in_features=embed_dim+embed_dim,
                out_features=latent_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=latent_dim,
                out_features=4
            )
        )

    def forward(self,
            z:torch.Tensor,
            net_out_edge_index:torch.Tensor
        ):
        """
        Input:
            x: [N,embed_dim]
            net_out_edge_index: [2,E]
        Output:
            logit: [E,4]
        """
        src,dst=net_out_edge_index
        src_z=z[src] # [E,embed_dim]
        dst_z=z[dst] # [E,embed_dim]
        edge_z=torch.cat(
            [src_z,dst_z],
            dim=-1
        )  # [E,2*embed_dim]
        return self.decoder(edge_z) # [E,4]