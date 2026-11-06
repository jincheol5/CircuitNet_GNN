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
            print(nodes_data.files)
            print(type(nodes_data["nodes"]))
            print(nodes_data["nodes"][:5],end="\n\n")
            print(f"len of nodes: {len(nodes_data['nodes'])}")

            print(f"net_edges data:")
            print(type(net_edges_data))
            print(net_edges_data.files)
            print(type(net_edges_data["net_edges"]))
            print(net_edges_data["net_edges"][:5],end="\n\n")
            print(f"len of net_edges: {len(net_edges_data['net_edges'])}")

            print(f"pin_positions data:")
            print(type(pin_positions_data))
            print(pin_positions_data.files)
            print(type(pin_positions_data["pin_positions"])) # 0차원 object array: NumPy 배열이라는 껍데기 안에 Python 객체 하나를 통째로 넣어둔 형태
            print(pin_positions_data["pin_positions"].shape) # 0차원 object array
            print(type(pin_positions_data["pin_positions"].item())) # dict
            position_dict=pin_positions_data["pin_positions"].item()
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