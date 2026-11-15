import numpy as np
import torch
import torch.nn as nn
import mlflow
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from utils import EarlyStopper

class ModelTrainer:
    @staticmethod
    def train(
            model:torch.nn.Module,
            train_loader:DataLoader,
            val_loader:DataLoader,
            **kwargs
        ):
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")

        if kwargs["optimizer"]=="adam":
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=kwargs["lr"]
            )
        else:
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=kwargs["lr"]
            )
        model=model.to(device)

        """
        Set Early Stopper
        """
        if kwargs["early_stop"]:
            early_stop=EarlyStopper(patience=kwargs["patience"])

        """
        model train
        """
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
            # epoch별 train loss 기록
            epoch_train_loss_list=[]

            model.train()
            for batch_data in tqdm(
                    train_loader,
                    desc=f"Training epoch: {epoch+1}..."
                ):
                batch_data=batch_data.to(device)

                ### forward -> [E,4]
                pred=model(
                    x=batch_data.x,
                    edge_index=batch_data.edge_index,
                    edge_attr=batch_data.edge_attr
                )

                ### compute loss
                criterion=nn.MSELoss()
                loss=criterion(pred,batch_data.edge_label)

                ### backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            """
            model validate
            """
            val_result=ModelTrainer.validate(
                model=model,
                val_loader=val_loader
            )
            print(f"Validate MSE Loss: {val_result['loss']}")

            """
            MLFlow tracking
            """
            if kwargs["use_mlflow"]:
                # epoch train loss 평균 계산 
                train_loss=np.mean(epoch_train_loss_list)
                mlflow.log_metrics({
                    "train_loss":train_loss,
                    "val_loss":val_result["loss"]
                },step=epoch+1)

            """
            Check Early Stop
            """
            val_loss=val_result["loss"]
            if np.isfinite(val_loss) and (
                    best_metrics is None or val_loss<best_metrics["best_val_loss"]
                ):
                best_metrics={
                    "best_epoch":epoch+1,
                    "best_val_loss":val_loss,
                }
            print(f"{epoch+1} epoch Validate Loss: {val_loss}")
            if kwargs["early_stop"]:
                pre_model=early_stop(
                    val_loss=val_loss,
                    model=model
                )
                if early_stop.early_stop:
                    model=pre_model
                    print(f"Early Stop in epoch {epoch+1}")
                    break

        if kwargs["early_stop"] and early_stop.best_state is not None:
            model.load_state_dict(early_stop.best_state)

        if kwargs["use_mlflow"]:
            if best_metrics is not None:
                mlflow.log_metrics(best_metrics,step=best_metrics["best_epoch"])

        return model

    @staticmethod
    def validate(
            model:torch.nn.Module,
            val_loader:DataLoader,
            **kwargs
        ):
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model=model.to(device)
        model.eval()

        """
        model validate
        """
        loss_list=[]
        with torch.no_grad():
            for batch_data in tqdm(
                    val_loader,
                    desc=f"Evaluating.."
                ):
                batch_data=batch_data.to(device)

                ### forward -> [net_out_E,4]
                pred=model(
                    x=batch_data.x,
                    edge_index=batch_data.edge_index,
                    net_out_edge_index=batch_data.net_out_edge_index
                ) 

                ### compute loss
                criterion=nn.L1Loss() # MAE Loss
                loss=criterion(pred,batch_data.net_delay_label)
                loss_list.append(loss)

        return torch.stack(loss_list).mean().item()

    @staticmethod
    def evaluate(
            model:torch.nn.Module,
            test_loader:DataLoader,
            **kwargs
        ):
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model=model.to(device)
        model.eval()

        """
        model validate
        """
        MSE_loss_list=[]
        MAE_loss_list=[]
        with torch.no_grad():
            for batch_data in tqdm(
                    test_loader,
                    desc=f"Evaluating.."
                ):
                batch_data=batch_data.to(device)

                ### forward -> [net_out_E,4]
                pred=model(
                    x=batch_data.x,
                    edge_index=batch_data.edge_index,
                    net_out_edge_index=batch_data.net_out_edge_index
                ) 

                ### compute loss
                MSE_criterion=nn.MSELoss() # MSE Loss
                MSE_loss=MSE_criterion(pred,batch_data.net_delay_label)
                MSE_loss_list.append(MSE_loss)

                MAE_criterion=nn.L1Loss() # MAE Loss
                MAE_loss=MAE_criterion(pred,batch_data.net_delay_label)
                MAE_loss_list.append(MAE_loss)
        return {
            "mse":torch.stack(MSE_loss_list).mean().item(),
            "mae":torch.stack(MAE_loss_list).mean().item()
        }