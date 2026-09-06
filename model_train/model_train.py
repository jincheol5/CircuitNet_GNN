import torch
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader

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
        model train
        """
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
            model.train()
            for batch_data in tqdm(
                    train_loader,
                    desc=f"Training epoch: {epoch+1}..."
                ):
                batch_data=batch_data.to(device)

                ### forward -> [net_out_E,4]
                pred=model(
                    x=batch_data.x,
                    edge_index=batch_data.edge_index,
                    net_out_edge_index=batch_data.net_out_edge_index
                ) 

                ### compute loss
                criterion=nn.MSELoss()
                loss=criterion(pred,batch_data.net_delay_label)

                ### backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            """
            model validate
            """
            val_loss=ModelTrainer.validate(
                model=model,
                val_loader=val_loader
            )
            print(f"Validate MAE Loss: {val_loss}")
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