import os
import mlflow
import torch.nn as nn
from pathlib import Path

BASE_PATH=os.path.join("..","data","CircuitNet_GNN") # 상대 경로
os.makedirs(BASE_PATH,exist_ok=True)
class MLFlowUtils:
    """
    1. Tracking Store & Artifact Store 설정
    2. Experiment 생성 or 선택
    3. Run 생성
    4. 파라미터 기록
    5. 학습 및 Metric 기록
    6. 모델 저장
    7. Run 종료
    """
    @staticmethod
    def setup(experiment_name:str):
        ### Tracking DB 경로 설정
        # sqlite:/// -> 상대 경로, sqlite://// -> 절대 경로 
        tracking_path=os.path.join(BASE_PATH,"tracking")
        os.makedirs(tracking_path,exist_ok=True)
        tracking_db_path=f"sqlite:///{tracking_path}/experiments.db" 
        mlflow.set_tracking_uri(tracking_db_path)

        ### Artifact 저장 경로
        # Artifact 저장 경로는 Experiment를 생성할 때 설정.
        # MLflow에서는 Tracking Store와 Artifact Store가 분리되어 있지만,
        # 어느 Artifact Store를 사용할지는 Tracking Store에 저장된 Experiment 정보가 결정함
        # Artifact 저장 경로는 Experiment의 메타데이터로서 Tracking Store에 기록됨
        # Tracking Store에는 Artifact 저장 경로 정보가 저장되고, Artifact Store에는 실제 모델 가중치 파일이 저장됨
        artifact_path=os.path.join(BASE_PATH,"artifact")
        os.makedirs(artifact_path,exist_ok=True)

        ### Experiment 조회 및 없을 시 생성
        # mlflow.set_tracking_uri()로 지정한 Tracking Store에서 Experiment를 조회
        experiment=mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                name=experiment_name,
                artifact_location=Path(artifact_path).resolve().as_uri() # Artifact 저장 경로 설정
            )

        ### Experiment 설정
        mlflow.set_experiment(experiment_name=experiment_name)

    @staticmethod
    def search_run_ids(
            experiment_name:str,
            model_name:str,
            dataset_name:str,
            seed:int,
            lr:float
        ):
        """
        최신 Run ID부터 정렬된 상태로 반환
        start_time DESC -> 최신 run부터 내림차순 정렬
        """
        runs=mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=(
                f"params.model_name = '{model_name}' AND "
                f"params.dataset_name = '{dataset_name}' AND "
                f"params.seed = '{seed}' AND "
                f"params.lr = '{lr}'"
            ),
            order_by=["start_time DESC"],
            output_format="pandas"
        ) # runs -> pd.DataFrame
        return runs["run_id"].tolist()

    @staticmethod
    def save_model(
            model:nn.Module,
            artifact_name:str
        ):
        model_info=mlflow.pytorch.log_model(
            pytorch_model=model,
            name=artifact_name,
            serialization_format="pickle"
        )
        print(f"Complete to save model!")
        print(f"Artifact Name: {artifact_name}")
        print(f"Run ID: {model_info.run_id}")
        print(f"Model ID: {model_info.model_id}")
        print(f"Model URI: {model_info.model_uri}",end="\n\n")

    @staticmethod
    def load_model(
            run_id:str,
            artifact_name:str
        ):
        """
        load model using run_id & artifact_name
        """
        model=mlflow.pytorch.load_model(
            model_uri=f"runs:/{run_id}/{artifact_name}"
        )
        return model