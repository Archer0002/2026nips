
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from models.earlystop import EarlyStopper
from sklift.metrics import qini_auc_score
import matplotlib.pyplot as plt
from utils import kendalltau
# from utils import tau2score
import os
from models.model_utils import test_model
class TONetv2(nn.Module):

    def __init__(self, input_dim, h_dim, num_layers) -> None:
        super().__init__()
        self.layer_cfg = [h_dim,h_dim//2,h_dim//2,h_dim//4]

        self.x_net = []
        self.x_net.append(nn.Linear(input_dim, self.layer_cfg[0]))
        for i in range(num_layers - len(self.layer_cfg)-1):
            self.x_net.append(nn.ELU())
            self.x_net.append(nn.Linear(h_dim, h_dim))
        self.x_net.append(nn.ELU())
        self.x_net = nn.Sequential(*self.x_net)

        self.h_net = []
        self.h_net.append(nn.Linear(self.layer_cfg[0]+1, self.layer_cfg[1]))
        self.h_net.append(nn.ELU())
        self.h_net.append(nn.Linear(self.layer_cfg[1], self.layer_cfg[2]))
        self.h_net.append(nn.ELU())
        self.h_net = nn.Sequential(*self.h_net)

        self.ht_net = []
        self.ht_net.append(nn.Linear(self.layer_cfg[2], self.layer_cfg[3]))
        self.ht_net.append(nn.ELU())
        self.ht_net.append(nn.Linear(self.layer_cfg[3], 1))
        self.ht_net = nn.Sequential(*self.ht_net)

        self.hy_net = []
        self.hy_net.append(nn.Linear(self.layer_cfg[2], self.layer_cfg[3]))
        self.hy_net.append(nn.ELU())
        self.hy_net.append(nn.Linear(self.layer_cfg[3], 1))
        self.hy_net = nn.Sequential(*self.hy_net)

        self.e_net = []
        self.e_net.append(nn.Linear(self.layer_cfg[0], self.layer_cfg[3]))
        self.e_net.append(nn.ELU())
        self.e_net.append(nn.Linear(self.layer_cfg[3], 1))
        self.e_net = nn.Sequential(*self.e_net)

        self.hy_bce = nn.BCELoss()
        self.ht_bce = nn.BCELoss()
        self.e_bce = nn.BCELoss()
        self.tr_bce = nn.BCELoss()

        self.epsilon = nn.Linear(in_features=1, out_features=1)
        torch.nn.init.xavier_normal_(self.epsilon.weight)


    def forward(self, x, t):
        x_rep = self.x_net(x)
        e_hat = self.e_net(x_rep)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        xt_rep = torch.cat([x_rep, t], dim=1)
        h_rep = self.h_net(xt_rep)
        t_recon = self.ht_net(h_rep)
        y_out = self.hy_net(h_rep)

        e_hat = torch.sigmoid(e_hat)
        t_recon = torch.sigmoid(t_recon)
        y_hat = torch.sigmoid(y_out)
        eps = self.epsilon(torch.ones_like(e_hat)[:, :1])

        return y_hat, t_recon, e_hat, eps, y_out

    def criterion(self, y_hat, t_recon, e_hat, eps, y_out, y, t, alpha, w=1.0):
        # loss = (w*(out.squeeze()-y.squeeze())**2).mean()
        hy_loss = self.hy_bce(y_hat.squeeze(), y.squeeze()).mean()
        ht_loss = self.ht_bce(t_recon.squeeze(), t.squeeze()).mean()
        e_loss = self.e_bce(e_hat.squeeze(), t.squeeze()).mean()
        
        to_loss = hy_loss + ht_loss + e_loss
        
        e_hat = ((e_hat+ 0.01)/1.02).squeeze()

        h = (t.squeeze() / e_hat) - ((1 - t.squeeze()) / (1 - e_hat))

        y_pert = torch.sigmoid(y_out.squeeze() + eps.squeeze() * h.squeeze())
        # 这个设计需要思考一下
        targeted_regularization = self.tr_bce(y_pert.squeeze(), y.squeeze()).mean()
        return to_loss + alpha * targeted_regularization
    def predict(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        t0 = torch.zeros((x.shape[0],1))
        t1 = torch.ones((x.shape[0],1))

        with torch.no_grad():
            y_0, t_0, e_0,eps, y_out = self.forward(x, t0)
            y_1, t_1, e_1,eps, y_out = self.forward(x, t1)

        return y_0, y_1
    
    def train_model(self, opt, train_dataloader, valid_data, test_data, args, exp=None, best_model_path=None):
        self.train()
        w = 1.0

        early_stopper = EarlyStopper(patience=5, min_delta=0)
        best_val_value = -1.0
        test_cate_epochs = np.zeros((args.epochs, test_data.shape[0]))
        for epoch in range(args.epochs):
            for i, sample in enumerate(train_dataloader):
                sample = sample.to(torch.float32)
                opt.zero_grad()
                x = sample[:, :args.x_dim]
                t = sample[:, args.x_dim]
                y = sample[:, args.x_dim+1]
                y_hat, t_hat, e_hat, eps, y_out = self(x,t)
                loss = self.criterion(y_hat, t_hat, e_hat, eps, y_out, y, t, args.alpha, w)
                loss.backward()
                opt.step()

            if (epoch+1) % 1 == 0:
                self.eval()
                if not torch.is_tensor(valid_data):
                    valid_data = torch.tensor(valid_data, dtype=torch.float32)
                else:
                    valid_data = valid_data.to(torch.float32)
                valid_xt = valid_data[:,:args.x_dim+1]
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
                        # test_cate_epochs[epoch] = test_cate.squeeze().detach().numpy()
                        pred_tau, pehe, pred_relative_uplift, pred_sep_qini,\
                        pred_joint_uplift_score, pred_joint_qini_score, pred_pu_score,\
                        true_relative_uplift, true_sep_qini, true_joint_uplift,\
                        true_joint_qini, true_pu_score, true_gains, pred_gains = test_model(self, args, valid_data, best_model_path, is_valid=True)
                    elif args.data == 'criteo':
                        pred_tau, pred_relative_uplift, pred_sep_qini,\
                        pred_joint_uplift_score, pred_joint_qini_score, pred_pu_score = test_model(self, args, valid_data, best_model_path, is_valid=True)

                if args.valid_metric == 'qini':
                    valid_score = pred_joint_qini_score
                elif args.valid_metric == 'pu':
                    valid_score = pred_pu_score

                if (epoch+1) % 1 == 0:
                    print(f"{args.model_name} epoch: {epoch} --- train_loss: {loss} --- valid_score: {valid_score} --- qini_score: {pred_joint_qini_score}")
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
            
