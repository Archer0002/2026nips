from functools import partial

import torch
import numpy as np
# from sklearn.model_selection import train_test_split
from sklift.metrics import qini_auc_score, uplift_auc_score
# from dragonnet.model import DragonNetBase, dragonnet_loss, tarreg_loss, EarlyStopper
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import kendalltau
import os
from models.model_utils import test_model
import matplotlib.pyplot as plt

class DragonNetBase(nn.Module):
    """
    Base Dragonnet model.

    Parameters
    ----------
    input_dim: int
        input dimension for convariates
    shared_hidden: int
        layer size for hidden shared representation layers
    outcome_hidden: int
        layer size for conditional outcome layers
    """
    def __init__(self, input_dim, h_dim=200, outcome_hidden=100, num_layers=5):
        super(DragonNetBase, self).__init__()
        self.x_net = []
        self.x_net.append(nn.Linear(input_dim, h_dim))
        for i in range(num_layers - 3):
            self.x_net.append(nn.ELU())
            self.x_net.append(nn.Linear(h_dim, h_dim))
        self.x_net = nn.Sequential(*self.x_net)

        self.treat_out = nn.Linear(in_features=h_dim, out_features=1)

        self.y0_fc1 = nn.Linear(in_features=h_dim, out_features=outcome_hidden)
        self.y0_fc2 = nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden)
        self.y0_out = nn.Linear(in_features=outcome_hidden, out_features=1)

        self.y1_fc1 = nn.Linear(in_features=h_dim, out_features=outcome_hidden)
        self.y1_fc2 = nn.Linear(in_features=outcome_hidden, out_features=outcome_hidden)
        self.y1_out = nn.Linear(in_features=outcome_hidden, out_features=1)

        self.epsilon = nn.Linear(in_features=1, out_features=1)
        torch.nn.init.xavier_normal_(self.epsilon.weight)

    def forward(self, inputs):
        """
        forward method to train model.

        Parameters
        ----------
        inputs: torch.Tensor
            covariates

        Returns
        -------
        y0: torch.Tensor
            outcome under control
        y1: torch.Tensor
            outcome under treatment
        t_pred: torch.Tensor
            predicted treatment
        eps: torch.Tensor
            trainable epsilon parameter
        """
        z = self.x_net(inputs)
        t_pred = torch.sigmoid(self.treat_out(z))

        y0 = F.elu(self.y0_fc1(z))
        y0 = F.elu(self.y0_fc2(y0))
        y0 = self.y0_out(y0)

        y1 = F.elu(self.y1_fc1(z))
        y1 = F.elu(self.y1_fc2(y1))
        y1 = self.y1_out(y1)

        eps = self.epsilon(torch.ones_like(t_pred)[:, 0:1])

        return y0, y1, t_pred, eps


def dragonnet_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, alpha=1.0):
    """
    Generic loss function for dragonnet

    Parameters
    ----------
    y_true: torch.Tensor
        Actual target variable
    t_true: torch.Tensor
        Actual treatment variable
    t_pred: torch.Tensor
        Predicted treatment
    y0_pred: torch.Tensor
        Predicted target variable under control
    y1_pred: torch.Tensor
        Predicted target variable under treatment
    eps: torch.Tensor
        Trainable epsilon parameter
    alpha: float
        loss component weighting hyperparameter between 0 and 1
    Returns
    -------
    loss: torch.Tensor
    """
    t_pred = (t_pred + 0.01) / 1.02
    loss_t = torch.sum(F.binary_cross_entropy(t_pred, t_true))

    loss0 = torch.sum((1. - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))
    loss_y = loss0 + loss1

    loss = loss_y + alpha * loss_t
    # print(f'loss_y:{loss_y}, loss_t:{loss_t}')
    return loss

def outcome_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, alpha=1.0):

    loss0 = torch.sum((1. - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))
    loss_y = loss0 + loss1

    loss = loss_y

    return loss

def tarreg_loss(y_true, t_true, t_pred, y0_pred, y1_pred, eps, alpha=0.1, beta=0.1):
    """
    Targeted regularisation loss function for dragonnet

    Parameters
    ----------
    y_true: torch.Tensor
        Actual target variable
    t_true: torch.Tensor
        Actual treatment variable
    t_pred: torch.Tensor
        Predicted treatment
    y0_pred: torch.Tensor
        Predicted target variable under control
    y1_pred: torch.Tensor
        Predicted target variable under treatment
    eps: torch.Tensor
        Trainable epsilon parameter
    alpha: float
        loss component weighting hyperparameter between 0 and 1
    beta: float
        targeted regularization hyperparameter between 0 and 1
    Returns
    -------
    loss: torch.Tensor
    """
    vanilla_loss = dragonnet_loss(y_true, t_true, t_pred, y0_pred, y1_pred, alpha)
    t_pred = (t_pred + 0.01) / 1.02

    y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

    h = (t_true / t_pred) - ((1 - t_true) / (1 - t_pred))

    y_pert = y_pred + eps * h
    targeted_regularization = torch.sum((y_true - y_pert)**2)
    
    # final
    loss = vanilla_loss + beta * targeted_regularization
    # loss = vanilla_loss 

    # print(f'vanilla_loss:{vanilla_loss}, targeted_regularization:{targeted_regularization}')
    return loss


class EarlyStopper:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class DragonNet:
    """
    Main class for the Dragonnet model

    Parameters
    ----------
    input_dim: int
        input dimension for convariates
    shared_hidden: int, default=200
        layer size for hidden shared representation layers
    outcome_hidden: int, default=100
        layer size for conditional outcome layers
    alpha: float, default=1.0
        loss component weighting hyperparameter between 0 and 1
    beta: float, default=1.0
        targeted regularization hyperparameter between 0 and 1
    epochs: int, default=200
        Number training epochs
    batch_size: int, default=64
        Training batch size
    learning_rate: float, default=1e-3
        Learning rate
    data_loader_num_workers: int, default=4
        Number of workers for data loader
    loss_type: str, {'tarreg', 'default'}, default='tarreg'
        Loss function to use
    """

    def __init__(
        self,
        input_dim,
        shared_hidden=200,
        outcome_hidden=100,
        num_layers=5,
        alpha=1.0,
        beta=1.0,
        epochs=200,
        batch_size=2000,
        learning_rate=1e-5,
        data_loader_num_workers=4,
        # loss_type="default",
        loss_type="tarreg",

    ):

        self.model = DragonNetBase(input_dim, shared_hidden, outcome_hidden, num_layers)
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = data_loader_num_workers
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_dataloader = None
        self.valid_dataloader = None
        if loss_type == "tarreg":
            self.loss_f = partial(tarreg_loss, alpha=alpha, beta=beta)
        elif loss_type == "default":
            self.loss_f = partial(dragonnet_loss, alpha=alpha)


    def parameters(self):
        return self.model.parameters()
    def state_dict(self):
        return self.model.state_dict()
    def train(self):
        return self.model.train()
    def eval(self):
        return self.model.eval()
    def load_state_dict(self, param):
        return self.model.load_state_dict(param)

    def train_model(self, opt, train_dataloader, valid_data, test_data, args, exp, best_model_path=None):
        # self.train_dataloader = train_dataloader
        # self.valid_dataloader = valid_dataloader
        # self.optim = opt
        # self.test_cates = test_cates
        self.model.train()
        # self.create_dataloaders(x, y, t, valid_perc)
        early_stopper = EarlyStopper(patience=5, min_delta=0)
        best_val_value = -1.0

        test_cate_epochs = np.zeros((args.epochs, test_data.shape[0]))

        for epoch in range(args.epochs):
            for batch, train_data in enumerate(train_dataloader):
                train_data = train_data.to(torch.float32)
                X = train_data[:,:args.x_dim]
                t = train_data[:,args.x_dim]
                y = train_data[:,args.x_dim+1]
                y0_pred, y1_pred, t_pred, eps = self.model(X)

                loss = tarreg_loss(y.squeeze(), t.squeeze(), t_pred.squeeze(),
                                y0_pred.squeeze(), y1_pred.squeeze(), eps.squeeze(),
                                args.alpha, args.beta)

                opt.zero_grad()
                loss.backward()
                opt.step()
            if (epoch+1) % 1 == 0:
                self.model.eval()
                if not torch.is_tensor(valid_data):
                    valid_data = torch.tensor(valid_data, dtype=torch.float32)
                else:
                    valid_data = valid_data.to(torch.float32)

                valid_x = valid_data[:,:args.x_dim]
                valid_t = valid_data[:,args.x_dim]
                valid_y = valid_data[:,args.x_dim+1]

                if not torch.is_tensor(test_data):
                    test_data = torch.tensor(test_data, dtype=torch.float32)
                else:
                    test_data = test_data.to(torch.float32)
                test_x = test_data[:,:args.x_dim]

                with torch.no_grad():
                    if args.data == 'synthetic':
                        test_y0, test_y1 = self.predict(test_x)
                        test_cate = test_y1 - test_y0
                        test_cate_epochs[epoch] = test_cate.squeeze().detach().numpy()
                        pred_tau, pehe, pred_relative_uplift, pred_sep_qini,\
                        pred_joint_uplift_score, pred_joint_qini_score, pred_pu_score,\
                        true_relative_uplift, true_sep_qini, true_joint_uplift,\
                        true_joint_qini, true_pu_score, true_gains, pred_gains = test_model(self, args, valid_data, best_model_path, is_valid=True)
                    elif args.data == 'criteo' or args.data == 'lzd':
                        pred_tau, pred_relative_uplift, pred_sep_qini,\
                        pred_joint_uplift_score, pred_joint_qini_score, pred_pu_score = test_model(self, args, valid_data, best_model_path, is_valid=True)

                if args.valid_metric == 'qini':
                    valid_score = pred_joint_qini_score
                elif args.valid_metric == 'pu':
                    valid_score = pred_pu_score

                if (epoch+1) % 1 == 0:
                    print(f"{args.model_name} epoch: {epoch} --- train_loss: {loss} --- valid_score: {valid_score}")
                if valid_score > best_val_value:
                    best_val_value = valid_score
                    torch.save(self.state_dict(), os.path.join(best_model_path, 'best_'+str(args.model_name)+'.pth'))
                if early_stopper.early_stop(valid_score):
                    print(f"{args.model_name} best model epoch: {epoch} --- train_loss: {loss} --- valid_score: {best_val_value}")
                    return best_val_value
                self.train()
        # 如果是测50个ecpoh的变化趋势，返回这个数组
        # return test_cate_epochs
        #否则，返回最好的valid_score
        return best_val_value

    # def validate_step(self, args):
    #     """
    #     Calculates validation loss

    #     Returns
    #     -------
    #     valid_loss: torch.Tensor
    #         validation loss
    #     """
    #     valid_loss = []
    #     with torch.no_grad():
    #         for batch, valid_data in enumerate(self.valid_dataloader):
    #             valid_data = valid_data.to(torch.float32)
    #             valid_X = valid_data[:,:args.x_dim]
    #             valid_t = valid_data[:,args.x_dim:args.x_dim+1]
    #             valid_y = valid_data[:,args.x_dim+1:args.x_dim+2]
    #             y0_pred, y1_pred, t_pred, eps = self.predict(valid_X)
    #             # loss = outcome_loss(valid_y.squeeze(), valid_t.squeeze(), t_pred.squeeze(), y0_pred.squeeze(), y1_pred.squeeze(), eps.squeeze())
    #             loss = tarreg_loss(valid_y.squeeze(), valid_t.squeeze(), t_pred.squeeze(), y0_pred.squeeze(), y1_pred.squeeze(), eps.squeeze())

    #             valid_loss.append(loss)
    #     return torch.Tensor(valid_loss).mean()

    def predict(self, x):
        """
        Function used to predict on covariates.

        Parameters
        ----------
        x: torch.Tensor or numpy.array
            covariates

        Returns
        -------
        y0_pred: torch.Tensor
            outcome under control
        y1_pred: torch.Tensor
            outcome under treatment
        t_pred: torch.Tensor
            predicted treatment
        eps: torch.Tensor
            trainable epsilon parameter
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            y0_pred, y1_pred, t_pred, eps = self.model(x)
        return y0_pred, y1_pred