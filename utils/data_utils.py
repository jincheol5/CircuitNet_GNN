import os
import numpy as np
import torch
from typing import Literal
from torch_geometric.data import Data

BASE_PATH=os.path.join("..","data","CircuitNet-N28")

class DataUtils:
    @staticmethod
    def get_base_path():
        return BASE_PATH

    @staticmethod
    def get_timing_features_info(
            data_type:Literal[
                "nodes",
                "net_edges",
                "pin_positions"
            ],
            head:int=10
        ):
        if data_type in ("nodes","net_edges"):
            dir_path=os.path.join(BASE_PATH,"timing_features","timing_features_post_route",data_type)
        else: # pin_positions
            dir_path=os.path.join(BASE_PATH,"timing_features",data_type)
        npz_files=[
            f for f in os.listdir(dir_path) 
            if f.endswith('.npz') and os.path.isfile(os.path.join(dir_path,f))
        ]
        return {
            "file_names":npz_files[:head],
            "len_files":len(npz_files)
        }

    @staticmethod
    def load_timing_features_data(
            file_name:str,
            data_type:Literal[
                "nodes",
                "net_edges",
                "pin_positions"
            ]
        ):
        """
        Net Delay Prediciton
        """
        if data_type in ("nodes","net_edges"):
            file_path=os.path.join(BASE_PATH,"timing_features","timing_features_post_route",data_type,f"{file_name}.npz")
        else: # pin_positions
            file_path=os.path.join(BASE_PATH,"timing_features",data_type,f"{file_name}.npz")
        data=np.load(file_path,allow_pickle=True)
        return data

    @staticmethod
    def save_PyG_Data(
            file_name:str,
            data:Data
        ):
        """
        """
        file_path=os.path.join(BASE_PATH,"timing_features","graph",f"{file_name}.pt")
        os.makedirs(
            os.path.dirname(file_path),
            exist_ok=True # 이미 폴더가 있어도 에러 X
        )
        torch.save(data,file_path)