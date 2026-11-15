import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import DataUtils,EMPTY_SAMPLE
from graph import NetlistGraph

BASE_PATH=os.path.join("..","data")
def main(**kwargs):
    match kwargs["app_num"]:
        case 1:
            """
            Save timing_features graph to PyG Data
            """
            nodes_dir=Path(BASE_PATH)/"CircuitNet-N28"/"timing_features"/"timing_features_post_route"/"nodes" 
            sample_names=[
                path.stem
                for path in nodes_dir.glob("*.npz")
            ]

            # empty sample name 제외
            empty_samples=set(EMPTY_SAMPLE)
            sample_names=[
                name for name in sample_names
                if name not in empty_samples
            ]

            failed_layouts=[]
            for sample_name in tqdm(sample_names,desc=f"Save graph..."):
                graph_path=Path(BASE_PATH)/"CircuitNet_GNN"/"graph"/f"{sample_name}.pt"
                
                # 이미 .pt 저장되어 있으면 넘어가기
                if graph_path.exists():
                    continue

                # graph 저장
                try:
                    nodes_data=DataUtils.load_timing_features_data(
                        file_name=sample_name,
                        data_type=f"nodes"
                    )
                    net_edges_data=DataUtils.load_timing_features_data(
                        file_name=sample_name,
                        data_type=f"net_edges"
                    )
                    pin_positions_data=DataUtils.load_timing_features_data(
                        file_name=sample_name,
                        data_type=f"pin_positions"
                    )
                    nodes=nodes_data["nodes"]
                    net_edges=net_edges_data["net_edges"]
                    pin_positions=pin_positions_data["pin_positions"].item()

                    # node pin names에서 EDA escape 문자 제거 전처리
                    nodes=np.array([
                        str(node).replace("\\", "")
                        for node in nodes
                    ])

                    data=NetlistGraph.build_graph(
                        sample_name=sample_name,
                        nodes=nodes,
                        net_edges=net_edges,
                        pin_positions=pin_positions
                    )
                    DataUtils.save_PyG_Data(
                        file_name=sample_name,
                        data=data
                    )
                except Exception as e:
                    failed_layouts.append({
                        "sample_name":sample_name,
                        "error":str(e)
                    })
                    tqdm.write(f"[ERROR] {sample_name}: {e}")

            print(f"Complete to save graphs!")
            if failed_layouts:
                print(f"\nFailed layouts: {len(failed_layouts)}")
                for failed in failed_layouts:
                    print(
                        f"- {failed['layout_sample_name']}: "
                        f"{failed['error']}"
                    )
            else:
                print("No failed layouts.")

if __name__=="__main__":
    """
    Execute APP
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--app_num",type=int,default=1)
    args=parser.parse_args()
    app_config={
        "app_num":args.app_num
    }
    main(**app_config)