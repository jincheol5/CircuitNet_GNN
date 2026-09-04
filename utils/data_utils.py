import os
import numpy as np
from typing import Literal


BASE_PATH=os.path.join("..","data","CircuitNet-N28")

class DataUtils:
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
        file_path=os.path.join(BASE_PATH,"timing_features",data_type,f"{file_name}.npz")
        data=np.load(file_path,allow_pickle=True)
        return data
        