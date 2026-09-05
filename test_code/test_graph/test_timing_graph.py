import argparse
import numpy as np
from utils import DataUtils
from graph import TimingGraph

def test_fn(**kwargs):
    ### Load dataset
    nodes_data=DataUtils.load_timing_features_data(
        file_name=kwargs["layout_sample_name"],
        data_type=f"nodes"
    )
    net_edges_data=DataUtils.load_timing_features_data(
        file_name=kwargs["layout_sample_name"],
        data_type=f"net_edges"
    )
    pin_positions_data=DataUtils.load_timing_features_data(
        file_name=kwargs["layout_sample_name"],
        data_type=f"pin_positions"
    )
    nodes=nodes_data["nodes"]
    net_edges=net_edges_data["net_edges"]
    pin_positions=pin_positions_data["pin_positions"].item()

    ### Convert to PyG Data
    data=TimingGraph.graph_build(
        layout_sample_name=kwargs["layout_sample_name"],
        nodes=nodes,
        net_edges=net_edges,
        pin_positions=pin_positions
    )

    print(f"Complete to convert PyG Data")

if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--layout_sample_name",type=str,default=f"8-RISCY-a-1-c2-u0.7-m1-p8-f0")
    args=parser.parse_args()
    test_config={
        "layout_sample_name":args.layout_sample_name
    }
    test_fn(**test_config)