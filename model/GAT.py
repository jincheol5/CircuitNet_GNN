import torch
import torch.nn as nn
from typing import Literal
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn import GraphNorm
from module import Net_Delay_Prediction_Decoder

class GAT_Encoder(nn.Module):
    def __init__(self,
            node_dim:int=32,
            latent_dim:int=32,
            embed_dim:int=32,
            n_layer:int=2,
            n_head:int=3,
            is_graph_norm:bool=False,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.n_layer=n_layer
        self.n_head=n_head
        self.node_dim=node_dim
        self.latent_dim=latent_dim
        self.embed_dim=embed_dim
        self.convs=nn.ModuleList()
        self.relu=nn.ReLU()
        self.is_graph_norm=is_graph_norm
        if is_graph_norm:
            self.graph_norms=nn.ModuleList()

        # encoder 입력층
        if n_layer>1:
            self.convs.append(
                GATConv(
                    in_channels=node_dim,
                    out_channels=latent_dim,
                    heads=n_head,
                    concat=True # output_dim = n_head * out_channels 
                )
            )
            if is_graph_norm:
                self.graph_norms.append(
                    GraphNorm(in_channels=n_head*latent_dim)
                )

            # encoder 은닉층
            for _ in range(n_layer-2):
                self.convs.append(
                    GATConv(
                        in_channels=n_head*latent_dim,
                        out_channels=latent_dim,
                        heads=n_head,
                        concat=True
                    )
                )
                if is_graph_norm:
                    self.graph_norms.append(
                        GraphNorm(in_channels=n_head*latent_dim)
                    )

            # encoder 출력층
            self.convs.append(
                GATConv(
                    in_channels=n_head*latent_dim,
                    out_channels=embed_dim,
                    heads=n_head,
                    concat=False # mean
                )
            )
        else:
            self.convs.append(
                GATConv(
                    in_channels=node_dim,
                    out_channels=embed_dim,
                    heads=n_head,
                    concat=False
                )
            )

    def forward(self,
            x:torch.Tensor,
            edge_index:torch.Tensor,
            batch:torch.Tensor|None=None
        ):
        if self.n_layer>1:
            if self.is_graph_norm:
                for conv,graph_norm in zip(self.convs[:-1],self.graph_norms):
                    x=conv(x,edge_index)
                    x=graph_norm(x,batch)
                    x=self.relu(x)
            else:   
                for conv in self.convs[:-1]:
                    x=conv(x,edge_index)
                    x=self.relu(x)
            z=self.convs[-1](x,edge_index)
        else:
            z=self.convs[0](x,edge_index)
        return z

class GAT_Net_Delay_Predictor(nn.Module):
    def __init__(self,
            node_dim:int=32,
            latent_dim:int=32,
            embed_dim:int=32,
            n_layer:int=2,
            n_head:int=3,
            is_graph_norm:bool=False,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.encoder=GAT_Encoder(
            node_dim=node_dim,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            n_layer=n_layer,
            n_head=n_head,
            is_graph_norm=is_graph_norm,
            **kwargs
        )
        self.decoder=Net_Delay_Prediction_Decoder(
            embed_dim=embed_dim,
            latent_dim=latent_dim
        )

    def forward(self,
            x:torch.Tensor,
            edge_index:torch.Tensor,
            net_out_edge_index:torch.Tensor
        ):
        z=self.encoder(
            x=x,
            edge_index=edge_index
        )
        pred=self.decoder(
            z=z,
            net_out_edge_index=net_out_edge_index
        )
        return pred # [E,4]