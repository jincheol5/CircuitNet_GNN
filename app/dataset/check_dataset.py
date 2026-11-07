import argparse
import numpy as np
from utils import DataUtils

def main(**kwargs):
    match kwargs["app_num"]:
        case 1:
            """
            Check nodes, net_edges, pin_positions info.
            """
            data_type=kwargs["data_type"]
            head=5
            info=DataUtils.get_timing_features_info(
                data_type=data_type,
                head=head
            )
            print(f"File names (Head: {head}):")
            for file_name in info["file_names"]:
                print(file_name)
            print()
            print(f"Length of .npz files: {info['len_files']}")

        case 2:
            """
            Check nodes, net_edges, pin_positions file.
            """
            nodes_data=DataUtils.load_timing_features_data(
                file_name=kwargs["file_name"],
                data_type=f"nodes"
            )

            net_edges_data=DataUtils.load_timing_features_data(
                file_name=kwargs["file_name"],
                data_type=f"net_edges"
            )

            pin_positions_data=DataUtils.load_timing_features_data(
                file_name=kwargs["file_name"],
                data_type=f"pin_positions"
            )

            print(f"nodes data:")
            print(type(nodes_data))
            print(nodes_data.files) # .npz 파일에 저장된 배열들의 이름(Key)을 리스트로 반환하는 속성
            print(type(nodes_data["nodes"]))
            print(nodes_data["nodes"][:5])
            print(f"len of nodes: {len(nodes_data['nodes'])}",end=f"\n\n")

            print(f"net_edges data:")
            print(type(net_edges_data))
            print(net_edges_data.files)
            print(type(net_edges_data["net_edges"]))
            net_edges=net_edges_data["net_edges"]
            print(f"net_edges shape: {net_edges.shape}")
            print(net_edges[:5])
            print(f"len of net_edges: {len(net_edges_data['net_edges'])}",end=f"\n\n")

            print(f"pin_positions data:")
            print(type(pin_positions_data))
            print(pin_positions_data.files)
            print(type(pin_positions_data["pin_positions"])) # 0차원 object array: NumPy 배열이라는 껍데기 안에 Python 객체 하나를 통째로 넣어둔 형태
            print(pin_positions_data["pin_positions"].shape) # 0차원 object array
            print(type(pin_positions_data["pin_positions"].item())) # dict
            position_dict=pin_positions_data["pin_positions"].item()
            key,value=next(iter(position_dict.items()))
            print("Key Type:",type(key))
            print("Value Type:",type(value))
            print(f"key ex: {list(position_dict.keys())[:5]}")
            position_value_ex=position_dict[f"FE_OCPC10204_testmode_i/I"]
            print(f"pin_position value ex: {position_value_ex}",end="\n\n")

            print(f"Analysis:")
            nodes=nodes_data["nodes"]
            net_edges=net_edges_data["net_edges"]
            src=net_edges[:,0].astype(np.int64)
            dst=net_edges[:,1].astype(np.int64)
            used_nodes=np.unique(
                np.concatenate([src,dst])
            )
            print(f"전체 node 수: {len(nodes)}")
            print(f"net edge에 등장하는 node 수: {len(used_nodes)}")
            print(f"net edge 수: {len(net_edges)}",end="\n\n")

        case 3:
            """
            Check. node name and pin_positions key name 
            """
            nodes_data=DataUtils.load_timing_features_data(
                file_name=kwargs["file_name"],
                data_type=f"nodes"
            )
            nodes=nodes_data["nodes"]

            pin_positions_data=DataUtils.load_timing_features_data(
                file_name=kwargs["file_name"],
                data_type=f"pin_positions"
            )
            pin_positions=pin_positions_data["pin_positions"].item()

            # nodes의 처음 3개
            print(f"First 3 node names:")
            for name in nodes[:3]:
                print("Node:", repr(name))

            # pin_positions의 처음 3개 Key
            print(f"First 3 pin_positions key names:")
            for key in list(pin_positions.keys())[:3]:
                print("Key:", repr(key))

            for node_name in nodes[:3]:
                print(f"{node_name}: {node_name in pin_positions}") # Node 이름과 함께 pin_positions에 존재하는지 True/False로 출력

        case 4:
            """
            Check. net_edges 방향성 (역방향도 다 포함하고 있는지)
            """
            net_edges_data=DataUtils.load_timing_features_data(
                file_name=kwargs["file_name"],
                data_type=f"net_edges"
            )
            net_edges=net_edges_data["net_edges"]

            # (src, dst) 추출
            edges=set(map(tuple,net_edges[:,:2]))

            # 역방향 Edge 확인 (Self-loop 제외)
            reverse_edges=[
                (src,dst)
                for src,dst in edges
                if src!=dst and (dst,src) in edges
            ]

            print(f"Total Edges: {len(net_edges)}")
            print(f"Unique Edges: {len(edges)}")
            print(f"Reverse Edges: {len(reverse_edges)}")
            print(f"Has Reverse Edges: {len(reverse_edges) > 0}")
            print(f"Examples: {reverse_edges[:10]}")


if __name__=="__main__":
    """
    Execute APP
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--data_type",
        type=str,
        choices=["nodes","net_edges","pin_positions"],
        default=f"nodes"
    )
    parser.add_argument("--file_name",type=str,default=f"8-RISCY-a-1-c2-u0.7-m1-p8-f0")
    args=parser.parse_args()
    app_config={
        "app_num":args.app_num,
        "data_type":args.data_type,
        "file_name":args.file_name
    }
    main(**app_config)