import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.earlystop import EarlyStopper
# from utils import tau2score
import os
from sklift.metrics import qini_auc_score, uplift_auc_score
# from utils import kendalltau, principled_uplift_auc_score, relative_uplift_auc_score, sep_qini_auc_score, get_true_gain
import numpy as np
# import matplotlib.pyplot as plt
from models.model_utils import test_model


class EUEN(nn.Module):

    def __init__(self, input_dim, h_dim, num_layers) -> None:
        super().__init__()
        self.layer_cfg = [h_dim,h_dim//2,h_dim//4,h_dim//8,1]

        self.t0_net = []
        self.t0_net.append(nn.Linear(input_dim, self.layer_cfg[0]))
        for i in range(num_layers - len(self.layer_cfg)):
            self.t0_net.append(nn.ELU())
            self.t0_net.append(nn.Linear(h_dim, h_dim))
        for i in range(1, len(self.layer_cfg)):
            self.t0_net.append(nn.ELU())
            self.t0_net.append(nn.Linear(self.layer_cfg[i-1], self.layer_cfg[i]))
        self.t0_net = nn.Sequential(*self.t0_net)

        self.t1_net = []
        self.t1_net.append(nn.Linear(input_dim, self.layer_cfg[0]))
        for i in range(num_layers - len(self.layer_cfg)):
            self.t1_net.append(nn.ELU())
            self.t1_net.append(nn.Linear(h_dim, h_dim))
        for i in range(1, len(self.layer_cfg)):
            self.t1_net.append(nn.ELU())
            self.t1_net.append(nn.Linear(self.layer_cfg[i-1], self.layer_cfg[i]))
        self.t1_net = nn.Sequential(*self.t1_net)

        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
    def forward(self, x, t):
        out_t0 = self.t0_net(x).squeeze()
        tau = self.t1_net(x).squeeze()
        
        out_t0_fix = out_t0.detach()
        out_t1 = tau + out_t0_fix

        return (1-t.squeeze())*out_t0 + t.squeeze()*out_t1

    def criterion(self, out, y, w=1.0):
        loss = self.mse(out.squeeze(), y.squeeze()).mean()
        return loss
    
    def predict(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x,dtype=torch.float32)
        t0 = torch.zeros((x.shape[0],1))
        t1 = torch.ones((x.shape[0],1))

        with torch.no_grad():
            y_0 = self.forward(x, t0)
            tau = self.forward(x, t1)

        return y_0, y_0+tau
    
    
    def train_model(self, opt, train_dataloader, valid_data, test_data, args, exp=None, best_model_path=None):
        self.train()
        w = 1.0

        early_stopper = EarlyStopper(patience=5, min_delta=0)
        test_cate_epochs = np.zeros((args.epochs, test_data.shape[0]))

        best_val_value = -1.0
        for epoch in range(args.epochs):
            for i, sample in enumerate(train_dataloader):
                sample = sample.to(torch.float32)
                opt.zero_grad()
                x = sample[:, :args.x_dim]
                t = sample[:, args.x_dim]
                y = sample[:, args.x_dim+1]
                out = self(x, t)
                loss = self.criterion(out, y, w)
                # opt.zero_grad()
                loss.backward()
                opt.step()

            if (epoch+1) % 1 == 0:
                self.eval()
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
                    out = self(valid_x, valid_t)
                    valid_loss = self.criterion(out,valid_y)

                    if args.data == 'synthetic':
                        test_y0, test_y1 = self.predict(test_x)
                        test_cate = test_y1 - test_y0
                        # test_cate_epochs[epoch] = test_cate.squeeze().detach().numpy()
                        pred_tau, pehe, pred_relative_uplift, pred_sep_qini,\
                        pred_joint_uplift_score, pred_joint_qini_score, pred_pu_score,\
                        true_relative_uplift, true_sep_qini, true_joint_uplift,\
                        true_joint_qini, true_pu_score, true_gains, pred_gains = test_model(self, args, valid_data, best_model_path, is_valid=True)
                    elif args.data == 'criteo' or args.data =='lzd':
                        pred_tau, pred_relative_uplift, pred_sep_qini,\
                        pred_joint_uplift_score, pred_joint_qini_score, pred_pu_score = test_model(self, args, valid_data, best_model_path, is_valid=True)

                if args.valid_metric == 'qini':
                    valid_score = pred_joint_qini_score
                elif args.valid_metric == 'pu':
                    valid_score = pred_pu_score

                if (epoch+1) % 1 == 0:
                    print(f"{args.model_name} epoch: {epoch} --- train_loss: {loss} --- valid_loss: {valid_loss} --- pred_tau: {pred_tau.mean()} --- valid_score: {valid_score}")
                if valid_score > best_val_value:
                    best_val_value = valid_score
                    torch.save(self.state_dict(), os.path.join(best_model_path, 'best_'+str(args.model_name)+'.pth'))
                if early_stopper.early_stop(valid_score):
                    print(f"{args.model_name} best model epoch: {epoch} --- train_loss: {loss} --- valid_score: {valid_score}")
                    return best_val_value
                self.train()
        # 如果是测50个ecpoh的变化趋势，返回这个数组
        # return test_cate_epochs
        #否则，返回最好的valid_score
        return best_val_value