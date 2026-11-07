import numpy as np
import torch
from torch_geometric.data import Data

class TimingGraph:
    @staticmethod
    def build_graph(
            sample_name:str,
            nodes:np.ndarray,
            net_edges:np.ndarray,
            pin_positions:dict[str,list]
        ):
        """
        graph_id: sample_name
        node: sample의 각 pin
        node_ft: pin_positions
        edge: 
            net_out (실제 net_edge): source pin instance -> sink pin instance 
            net_in (양방향 지정을 위한 실제 net_edge의 역방향 edge): sink pin instance -> source pin instance
            net delay label은 net_out edge_index에 대해서만 지정

        Input: 
            sample_name
            nodes: (N,)
            net_edges: (E,6)
            pin_positions: dict
                key: str
                value: list (len=6)
        Return: PyG Data
            x: [N,4]
            edge_index: [2,2*E]
            edge_type: [2*E,], 1=net_out, 0=net_in
            edge_label: [E,4]
            graph_id: str, sample_name
        """
        # node_ft: [N,4]
        node_ft=torch.tensor(
            [
                pin_positions.get(
                    str(node),
                    [0.0,0.0,0.0,0.0] # position_dict에 없는 pin의 경우 임시로 0.0으로 feature 채움
                )[:4]
                for node in nodes
            ],
            dtype=torch.float32
        )

        # edge_index: [2,2*E]
        src=torch.tensor(
            net_edges[:,0],
            dtype=torch.long
        )
        dst=torch.tensor(
            net_edges[:,1],
            dtype=torch.long
        )
        net_out_edge_index=torch.stack(
            [src,dst],
            dim=0
        )
        net_in_edge_index=torch.stack(
            [dst,src],
            dim=0
        )
        edge_index=torch.cat(
            [net_out_edge_index,net_in_edge_index],
            dim=1
        )
        edge_type=torch.cat(
            [
                torch.ones(net_out_edge_index.size(1),dtype=torch.long),
                torch.zeros(net_in_edge_index.size(1),dtype=torch.long)
            ],
            dim=0
        ) # net_out=1, net_in=0
        edge_label=torch.tensor(
            net_edges[:,2:],
            dtype=torch.float32
        ) # [E,4], net_out edge의 net_delay label

        # set PyG Data
        graph=Data(
            x=node_ft,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_label=edge_label
        )
        graph.graph_id=sample_name
        return graph
