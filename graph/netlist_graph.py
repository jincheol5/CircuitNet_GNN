import numpy as np
import torch
from torch_geometric.data import Data

class NetlistGraph:
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
                value: list, 앞의 4개 값은 GCell 격자 인덱스 [x_min, y_min, x_max, y_max]
        Return: PyG Data
            x: [N,4]
            edge_index: [2,E]
            edge_attr: [E,1]
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

        # edge_index: [2,E]
        src=torch.tensor(
            net_edges[:,0],
            dtype=torch.long
        )
        dst=torch.tensor(
            net_edges[:,1],
            dtype=torch.long
        )
        edge_index=torch.stack([src,dst],dim=0) # net_out edge
        edge_label=torch.tensor(
            net_edges[:,2:],
            dtype=torch.float32
        ) # [E,4], net_out edge의 net_delay label
        edge_attr=torch.linalg.vector_norm(
            node_ft[edge_index[0]]-node_ft[edge_index[1]],
            dim=1,
            keepdim=True
        )  # [E,1], 두 노드 사이의 거리 값

        # src 기준 오름차순 정렬
        perm=edge_index[0].argsort(stable=True)
        edge_index=edge_index[:,perm]
        edge_label=edge_label[perm]
        edge_attr=edge_attr[perm]

        # set PyG Data
        graph=Data(
            x=node_ft, # [N,4]
            edge_index=edge_index, # [2,E]
            edge_attr=edge_attr, # [E,1]
            edge_label=edge_label # [E,4]
        )
        graph.graph_id=sample_name
        return graph
