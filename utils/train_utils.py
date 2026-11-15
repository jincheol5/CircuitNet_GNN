import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class NetlistDataset(Dataset):
    """
    하나의 Netlist Graph를 Source Pin 단위의 Star Graph로 분할하는 Dataset.

    각 Source Pin과 연결된 모든 Sink Pin 및 Edge를 하나의 Data 객체로 구성하며,
    PyG DataLoader를 통해 여러 Star Graph를 Mini-batch로 처리할 수 있도록 함.

    조건:
        - Edge는 Source -> Sink 방향으로 구성됨
        - Edge는 Source Node ID 기준으로 정렬되어 있음
        - Source와 Sink Node ID 집합은 서로 겹치지 않음
        - 각 Sink는 하나의 Source에만 연결됨
    """
    def __init__(self,graph:Data):
        super().__init__()
        self.graph=graph

        # Source별 Edge 개수
        self.sources,counts=torch.unique_consecutive(
            graph.edge_index[0],
            return_counts=True
        )

        # Source별 Edge 시작/종료 위치, [N_source+1,] 인덱싱을 위해 N_source+1 크기
        self.ptr=torch.cat([
            counts.new_zeros(1),
            counts.cumsum(dim=0)
        ]) 

    def __len__(self):
        return len(self.sources)

    def __getitem__(self,idx):
        start=self.ptr[idx].item()
        end=self.ptr[idx+1].item()

        # Source -> Sink Edge 추출
        sub_edge_index=self.graph.edge_index[:,start:end]

        # Local Node ID 변환
        node_ids,local_edge_index=torch.unique(
            sub_edge_index,
            sorted=True,
            return_inverse=True
        )
        return Data(
            x=self.graph.x[node_ids], # [sub_E+1,4], source pin + source와 연결된 sink pins
            edge_index=local_edge_index, # [2,sub_E]
            edge_attr=self.graph.edge_attr[start:end], # [sub_E,1]
            edge_label=self.graph.edge_label[start:end] # [sub_E,4]
        )

class EarlyStopper:
    def __init__(self,
            patience:int=1
        ):
        self.patience=patience
        self.patience_count=0
        self.best_loss=np.inf
        self.best_state=None
        self.early_stop=False
    def __call__(self,
            val_loss:float,
            model:torch.nn.Module
        ):
        # val_loss가 NaN, Inf이면 즉시 early stop
        if not np.isfinite(val_loss): 
            print("Loss is NaN or Inf!")
            self.early_stop=True
            if self.best_state is not None:
                model.load_state_dict(self.best_state)
            return model

        # 첫 번째 validation에서는 비교할 이전 best가 없으므로 현재 loss와 모델을 그대로 best로 저장
        if self.best_state is None: 
            self.best_loss=val_loss
            self.best_state={
                key: value.detach().clone()
                for key,value in model.state_dict().items()
            }
            return model

        # val_loss가 개선 되지 않은 경우
        if self.best_loss<=val_loss: 
            self.patience_count+=1
            if self.patience<=self.patience_count:
                self.early_stop=True
                model.load_state_dict(self.best_state)
            return model

        # val_loss가 개선 된 경우
        self.patience_count=0
        self.best_loss=val_loss
        self.best_state={
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        return model

class TrainUtils:
    @staticmethod
    def set_seed(seed:int):
        os.environ["PYTHONHASHSEED"]=str(seed) # Python의 문자열 등에 사용하는 해시 시드를 설정
        random.seed(seed) # Python 표준 random 모듈의 난수 시드를 고정
        np.random.seed(seed) # NumPy의 기존 전역 난수 생성기 시드를 고정
        torch.manual_seed(seed) # PyTorch의 CPU 및 CUDA 등 장치의 난수 시드를 설정
        torch.backends.cudnn.deterministic=True # cuDNN이 사용하는 알고리즘을 결정적인 알고리즘으로 제한
        torch.backends.cudnn.benchmark=False # cuDNN이 여러 알고리즘의 속도를 측정해 실행 알고리즘을 선택하는 기능 끔
