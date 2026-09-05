import numpy as np
import torch
from torch_geometric.data import Data

class TimingGraph:
    @staticmethod
    def graph_build(
            layout_sample_name:str,
            nodes:np.ndarray,
            net_edges:np.ndarray,
            pin_positions:dict[str,list]
        ):
        """
        graph type: bidirectional directed graph
        graph_id: layout_sample_name
        node: pin instance
        edge: 
            net_out (실제 net_edge): source pin instance -> sink pin instance 
            net_in (양방향 지정을 위한 실제 net_edge의 역방향 edge): sink pin instance -> source pin instance
            net delay label은 net_out edge_index에 대해서만 지정

        Input:
            layout_sample_name: str
            nodes: np.ndarray
            net_edges: np.ndarray
            pin_positions: dict[str,list]
        
        Return:
            graph: PyG Data
        """
        # node_ft: [N,4]
        node_ft=torch.tensor(
            [
                pin_positions.get(
                    str(node).replace("\\", ""), # layout 이름에서 \ 제거하여 position dict의 key 형식과 맞춤
                    [0.0,0.0,0.0,0.0] # position_dict에 없는 pin의 경우 임시로 0.0으로 feature 채움
                )[:4]
                for node in nodes
            ],
            dtype=torch.float32
        )

        # net_out_edge_index (source -> sink): [2,net_out_E]
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

        # net_in_edge_index (sink -> source): [2,net_in_E]
        net_in_edge_index=torch.stack(
            [dst,src],
            dim=0
        )

        # 전체 bidirectional edge
        edge_index=torch.cat(
            [net_out_edge_index,net_in_edge_index],
            dim=1
        )

        # net_out edge에 대한 4가지 net delay label: [net_out_E,4]
        net_delay_label=torch.tensor(
            net_edges[:,2:],
            dtype=torch.float32
        ) 

        # set PyG Data
        graph=Data(
            x=node_ft,
            edge_index=edge_index,
            net_out_edge_index=net_out_edge_index,
            net_in_edge_index=net_in_edge_index,
            net_delay_label=net_delay_label,
            num_nodes=node_ft.size(0),
            num_edges=edge_index.size(1)
        )
        graph.graph_id=layout_sample_name
        return graph