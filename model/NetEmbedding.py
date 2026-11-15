import torch
import torch.nn as nn
from torch_geometric.nn import MLP
from torch_geometric.utils import scatter

"""
paper:
- A Timing Engine Inspired Graph Neural Network Model for Pre-Routing Slack Prediction.
"""
class NetConv(nn.Module):
    """
    Graph Broadcast와 Graph Reduce를 통해 pin feature를 변환.
    - Graph Broadcase: source pin -> sink pin
    - Graph Reduce: sink pin -> source pin

    조건:
        - sink_pin에 중복 Node ID가 없음
        - 모든 sink_pin은 각각 하나의 source_pin에만 연결됨
        - Source와 Sink Node ID 집합이 겹치지 않음
        - 모든 Node가 Source 또는 Sink에 속함
    """
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            latent_dim:int,
            output_dim:int
        ):
        super().__init__()
        self.latent_dim=latent_dim
        self.broadcast_mlp=MLP(
            in_channels=2*node_dim+edge_dim,
            hidden_channels=latent_dim,
            out_channels=output_dim,
            num_layers=2,
            act="relu",
            plain_last=True # 마지막 계층에 활성화 함수(Activation), 정규화(Normalization), Dropout을 적용 X
        )
        self.message_mlp=MLP(
            in_channels=2*node_dim+edge_dim,
            hidden_channels=latent_dim,
            out_channels=2*latent_dim,
            num_layers=2,
            act="relu",
            plain_last=True
        )
        self.reduce_mlp=MLP(
            in_channels=2*latent_dim+node_dim,
            hidden_channels=latent_dim,
            out_channels=output_dim,
            num_layers=2,
            act="relu",
            plain_last=True
        )

    def graph_broadcast(self,
            x:torch.Tensor,
            source_pin:torch.Tensor,
            sink_pin:torch.Tensor,
            edge_attr:torch.Tensor
        ):
        """
        Input:
            x: [N,node_dim]
            source_pin: [E,]
            sink_pin: [E,]
            edge_attr: [E,edge_dim]
        Return:
            new_sink_pin_ft: [N,output_dim]
        """
        source_ft=x[source_pin]
        sink_ft=x[sink_pin]
        msg=torch.cat([source_ft,sink_ft,edge_attr],dim=-1)
        broadcast_ft=self.broadcast_mlp(msg) # [E,output_dim]

        # new_sink_pin_ft: [N,output_dim]
        new_sink_pin_ft=broadcast_ft.new_zeros((x.size(0),broadcast_ft.size(1))) # [N,output_dim]
        new_sink_pin_ft=new_sink_pin_ft.index_copy(
            dim=0,
            index=sink_pin,
            source=broadcast_ft
        )
        return new_sink_pin_ft 

    def graph_reduce(self,
            x:torch.Tensor,
            source_pin:torch.Tensor,
            sink_pin:torch.Tensor,
            edge_attr:torch.Tensor
        ):
        """
        Input:
            x: [N,node_dim]
            source_pin: [E,]
            sink_pin: [E,]
            edge_attr: [E,edge_dim]
        Return:
            new_source_pin_ft: [N,output_dim]
        """
        source_ft=x[source_pin]
        sink_ft=x[sink_pin]
        msg=torch.cat([source_ft,sink_ft,edge_attr],dim=-1)
        msg_ft=self.message_mlp(msg) # [E,2*latent_dim]

        # split msg_ft for sum/max 
        msg_ft_1,msg_ft_2=torch.split(
            msg_ft,
            [self.latent_dim,self.latent_dim],
            dim=-1
        )  # 각각 [E,latent_dim]

        # Sum Aggregation
        sum_ft=scatter(
            msg_ft_1,
            source_pin,
            dim=0,
            dim_size=x.size(0),
            reduce="sum"
        ) # [N,latent_dim]

        # Max Aggregation
        max_ft=scatter(
            msg_ft_2,
            source_pin,
            dim=0,
            dim_size=x.size(0),
            reduce="max"
        ) # [N,latent_dim]

        # 중복 없는 Source Node ID
        unique_source=torch.unique(source_pin)  # [S]

        # Source Node Feature만 추출
        sum_ft=sum_ft[unique_source] # [S,latent_dim]
        max_ft=max_ft[unique_source] # [S,latent_dim]
        source_ft=x[unique_source] # [S,node_dim]

        # Source Feature 갱신
        aggr_ft=torch.cat(
            [sum_ft,max_ft,source_ft],
            dim=-1
        ) # [S,2*latent_dim+node_dim]
        reduce_ft=self.reduce_mlp(aggr_ft) # [S,output_dim]

        # 원본 Node ID 위치에 저장
        new_source_pin_ft=reduce_ft.new_zeros(
            (x.size(0),reduce_ft.size(1))
        )
        new_source_pin_ft=new_source_pin_ft.index_copy(
            dim=0,
            index=unique_source,
            source=reduce_ft
        )
        return new_source_pin_ft # [N,output_dim]

    def forward(self,
            x:torch.Tensor,
            edge_index:torch.Tensor,
            edge_attr:torch.Tensor
        ):
        """
        Input:
            x: [N,node_dim]
            edge_index: [2,E]
            edge_attr: [E,edge_dim]
        Return:
            z: [N,output_dim]
        """
        source_pin,sink_pin=edge_index
        new_sink_pin_ft=self.graph_broadcast(
            x=x,
            source_pin=source_pin,
            sink_pin=sink_pin,
            edge_attr=edge_attr
        )
        new_source_pin_ft=self.graph_reduce(
            x=x,
            source_pin=source_pin,
            sink_pin=sink_pin,
            edge_attr=edge_attr
        )
        z=new_sink_pin_ft+new_source_pin_ft # [N,output_dim], 각 new_source,sink_pin_ft는 source/sink가 아닌 부분은 zero 이기 때문에 단순히 + 연산으로 구할 수 있음.
        return z

class NetEmbedding(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            latent_dim:int,
            embed_dim:int
        ):
        super().__init__()
        # NetConv Module
        self.net_convs=nn.ModuleList(
            [
                NetConv(node_dim=node_dim,edge_dim=edge_dim,latent_dim=latent_dim,output_dim=embed_dim),
                NetConv(node_dim=embed_dim,edge_dim=edge_dim,latent_dim=latent_dim,output_dim=embed_dim),
                NetConv(node_dim=embed_dim,edge_dim=edge_dim,latent_dim=latent_dim,output_dim=embed_dim)
            ]
        )
        # Net Delay Prediction Decoder
        self.decoder=MLP(
            in_channels=2*embed_dim,
            hidden_channels=latent_dim,
            out_channels=4,
            num_layers=2,
            act="relu",
            plain_last=True
        )

    def forward(self,
            x:torch.Tensor,
            edge_index:torch.Tensor,
            edge_attr:torch.Tensor
        ):
        """
        Input:
            x: [N,node_dim]
            edge_index: [2,E]
            edge_attr: [E,edge_dim]
        """
        # Apply NetConv Module
        for conv in self.net_convs:
            x=conv(x,edge_index,edge_attr)

        # Net Delay Prediction Decoder
        source_pin,sink_pin=edge_index
        source_pin_ft=x[source_pin] # [E,embed_dim]
        sink_pin_ft=x[sink_pin] # [E,embed_dim]
        input_ft=torch.cat([source_pin_ft,sink_pin_ft],dim=-1) # [E,2*embed_dim]
        net_delay_output=self.decoder(input_ft) # [E,4]
        return net_delay_output