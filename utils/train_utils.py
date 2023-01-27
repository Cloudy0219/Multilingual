import torch
from tqdm import tqdm
from utils.data_utils import get_data_loader
from torch import nn
import scipy


# def single_predict():
#     loss_list = []
#     y_pred_list = []
#     y_list = []
#     model_args["optimizer"].zero_grad()
#     train_type = None
#     if is_train:
#         data_loader = get_data_loader(train_data, batch_size=model_args["batch_size"])
#         model.train()
#         train_type = "train"
#     else:
#         data_loader = get_data_loader(
#             valid_data, batch_size=model_args["batch_size"], train=False
#         )
#         model.eval()
#         train_type = "valid"
#
#     with tqdm(data_loader, unit="batch") as tepoch:
#         tepoch.set_description(f"Epoch {curr_epoch} - {train_type}")
#         for step, batch in enumerate(tepoch):
#             model_args["optimizer"].zero_grad()
#             X = batch[0]
#             y = batch[1].float().to(DEVICE)
#             y_pred, _ = model(X)
#             y_pred_list.extend(y_pred.reshape(-1).tolist())
#             y_list.extend(y.tolist())
#             loss = model_args["criterion"](y_pred.reshape(-1), y)
#             loss_list.append(loss.item())
#             if is_train:
#                 loss.backward()
#                 nn.utils.clip_grad_norm_(model.parameters(), 2)
#                 model_args["optimizer"].step()
#                 model_args["scheduler"].step()
#                 with torch.cuda.device(DEVICE):
#                     torch.cuda.empty_cache()
#             tepoch.set_postfix(loss=sum(loss_list) / len(loss_list))
#     if is_train is False:
#         valid_data[f"y_pred_{curr_epoch}"] = y_pred_list
#     else:
#         train_data[f"y_pred_{curr_epoch}"] = y_pred_list
#     return sum(loss_list) / len(loss_list), y_pred_list, y_list
#
#     pass


class Trainer(nn.Module):
    def __init__(self, args, criterion, optimizer, scheduler, model, train_data, valid_data, device):
        super().__init__()
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.device = device

    def single_train_loop(self, curr_epoch):
        loss_list = []
        y_pred_list = []
        y_list = []
        data_loader = get_data_loader(self.train_data, batch_size=self.args.batch_size)
        self.optimizer.zero_grad()
        self.model.train()

        with tqdm(data_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {curr_epoch} - train")
            for step, batch in enumerate(tepoch):
                self.optimizer.zero_grad()
                X = batch[0]
                y = batch[1].float().to(self.device)
                y_pred, _ = self.model(X)
                y_pred_list.extend(y_pred.reshape(-1).tolist())
                y_list.extend(y.tolist())
                loss = self.criterion(y_pred.reshape(-1), y)
                loss_list.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 2)
                self.optimizer.step()
                self.scheduler.step()
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
                tepoch.set_postfix(loss=sum(loss_list) / len(loss_list))
        self.train_data[f"y_pred_{curr_epoch}"] = y_pred_list
        return sum(loss_list) / len(loss_list), y_pred_list, y_list

    def evaluate(self, curr_epoch):
        loss_list = []
        y_pred_list = []
        y_list = []
        data_loader = get_data_loader(
            self.valid_data, batch_size=self.args.batch_size, train=False
        )
        self.optimizer.zero_grad()
        self.model.eval()
        train_type = "valid"

        with tqdm(data_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {curr_epoch} - {train_type}")
            for step, batch in enumerate(tepoch):
                self.optimizer.zero_grad()
                X = batch[0]
                y = batch[1].float().to(self.device)
                y_pred, _ = self.model(X)
                y_pred_list.extend(y_pred.reshape(-1).tolist())
                y_list.extend(y.tolist())
                loss = self.criterion(y_pred.reshape(-1), y)
                loss_list.append(loss.item())
                tepoch.set_postfix(loss=sum(loss_list) / len(loss_list))

        self.valid_data[f"y_pred_{curr_epoch}"] = y_pred_list
        return sum(loss_list) / len(loss_list), y_pred_list, y_list


# def train_or_valid(model_args, curr_epoch, model, train_data, valid_data, DEVICE, is_train=True):
#     """
#     This fn. is used to train or validate the model
#     params:
#         model_args: a dict of model parameters
#         curr_epoch: Current value of the epoch
#         model: model to be trained
#         is_train: can be True or False depending on whether to train or validate
#
#     returns:
#         loss: sum of the loss across all tokens
#
#     """
#     loss_list = []
#     y_pred_list = []
#     y_list = []
#     model_args["optimizer"].zero_grad()
#     train_type = None
#     if is_train:
#         data_loader = get_data_loader(train_data, batch_size=model_args["batch_size"])
#         model.train()
#         train_type = "train"
#     else:
#         data_loader = get_data_loader(
#             valid_data, batch_size=model_args["batch_size"], train=False
#         )
#         model.eval()
#         train_type = "valid"
#
#     with tqdm(data_loader, unit="batch") as tepoch:
#         tepoch.set_description(f"Epoch {curr_epoch} - {train_type}")
#         for step, batch in enumerate(tepoch):
#             model_args["optimizer"].zero_grad()
#             X = batch[0]
#             y = batch[1].float().to(DEVICE)
#             y_pred, _ = model(X)
#             y_pred_list.extend(y_pred.reshape(-1).tolist())
#             y_list.extend(y.tolist())
#             loss = model_args["criterion"](y_pred.reshape(-1), y)
#             loss_list.append(loss.item())
#             if is_train:
#                 loss.backward()
#                 nn.utils.clip_grad_norm_(model.parameters(), 2)
#                 model_args["optimizer"].step()
#                 model_args["scheduler"].step()
#                 with torch.cuda.device(DEVICE):
#                     torch.cuda.empty_cache()
#             tepoch.set_postfix(loss=sum(loss_list) / len(loss_list))
#     if is_train is False:
#         valid_data[f"y_pred_{curr_epoch}"] = y_pred_list
#     else:
#         train_data[f"y_pred_{curr_epoch}"] = y_pred_list
#     return sum(loss_list) / len(loss_list), y_pred_list, y_list


# # Defining parameters for training the model
# def get_model_args():
#     # returns a dict - {param: value}
#     return {
#         "batch_size": 64,
#         "epoch": 8,
#         "learning_rate": 0.0001,
#         "model_name": "cardiffnlp/twitter-xlm-roberta-base",
#     }








