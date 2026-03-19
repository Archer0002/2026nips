import torch
import torch.nn as nn
import math
import sys
import numpy as np
from torch.utils.data import DataLoader
from models.earlystop import EarlyStopper
# from utils import tau2score
import os
from models.model_utils import test_model
import matplotlib.pyplot as plt

def init_weights(m):
    if isinstance(m, nn.Linear):
        stdv = 1 / math.sqrt(m.weight.size(1))
        torch.nn.init.normal_(m.weight, mean=0.0, std=stdv)
        # torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

def sigmod2(y):
    # y = torch.clamp(0.995 / (1.0 + torch.exp(-y)) + 0.0025, 0, 1)
    # y = torch.clamp(y, -16, 16)
    y=torch.sigmoid(y)
    # y = 0.995 / (1.0 + torch.exp(-y)) + 0.0025

    return y

def safe_sqrt(x):
    ''' Numerically safe version of Pytoch sqrt '''
    return torch.sqrt(torch.clip(x, 1e-9, 1e+9))

class ShareNetwork(nn.Module):
    def __init__(self, input_dim, share_dim, base_dim, args, device):
        super(ShareNetwork, self).__init__()
        if args.BatchNorm1d:
            print("use BatchNorm1d")
            self.DNN = nn.Sequential(

                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(share_dim, share_dim),
                # nn.BatchNorm1d(share_dim),
                nn.ELU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(share_dim, base_dim),
                # nn.BatchNorm1d(base_dim),
                nn.ELU(),
                nn.Dropout(p=args.dropout)
            )
        else:
            print("No BatchNorm1d")
            self.DNN = nn.Sequential(
                nn.Linear(input_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(share_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(share_dim, base_dim),
                nn.ELU(),
            )

        self.DNN.apply(init_weights)
        self.args = args
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        h_rep = self.DNN(x)
        if self.args.normalization == "divide":
            h_rep_norm = h_rep / safe_sqrt(torch.sum(torch.square(h_rep), dim=1, keepdim=True))
        else:
            h_rep_norm = 1.0 * h_rep
        return h_rep_norm


class BaseModel(nn.Module):
    def __init__(self, base_dim, args):
        super(BaseModel, self).__init__()
        self.DNN = nn.Sequential(
            nn.Linear(base_dim, base_dim//2),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(base_dim//2, base_dim//4),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=args.dropout),
            # nn.Linear(base_dim, base_dim),
            # # nn.BatchNorm1d(base_dim),
            # nn.ELU(),
            # nn.Dropout(p=args.dropout)
        )
        self.DNN.apply(init_weights)

    def forward(self, x):
        logits = self.DNN(x)
        return logits

class BaseModel4MetaLearner(nn.Module):
    def __init__(self, input_dim, base_dim, args, device):
        super(BaseModel4MetaLearner, self).__init__()
        self.DNN = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, base_dim//2),
            nn.ELU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(base_dim//2, base_dim//4),
            # nn.BatchNorm1d(share_dim),
            # nn.ELU(),
            # nn.Dropout(p=args.dropout),
            # nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(share_dim),
            # nn.ELU(),
            # nn.Dropout(p=args.dropout),
            # nn.Linear(base_dim, 1),
            # nn.ELU()
            # nn.BatchNorm1d(base_dim),
        )
        self.DNN.apply(init_weights)
        self.args = args
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        logit = self.DNN(x)
        return logit


class PrpsyNetwork(nn.Module):
    """propensity network"""
    def __init__(self, base_dim, args):
        super(PrpsyNetwork, self).__init__()
        self.baseModel = BaseModel(base_dim, args)
        self.logitLayer = nn.Linear(base_dim//4, 1)
        self.sigmoid = nn.Sigmoid()
        self.logitLayer.apply(init_weights)

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        return p


class Mu0Network(nn.Module):
    def __init__(self, base_dim, args):
        super(Mu0Network, self).__init__()
        self.baseModel = BaseModel(base_dim, args)
        self.logitLayer = nn.Linear(base_dim//4, 1)
        self.logitLayer.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        # return self.relu(p)
        return p


class Mu1Network(nn.Module):
    def __init__(self, base_dim, args):
        super(Mu1Network, self).__init__()
        self.baseModel = BaseModel(base_dim, args)
        self.logitLayer = nn.Linear(base_dim//4, 1)
        self.logitLayer.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        # return self.relu(p)
        return p


class TauNetwork(nn.Module):
    """pseudo tau network"""
    def __init__(self, base_dim, args):
        super(TauNetwork, self).__init__()
        self.baseModel = BaseModel(base_dim, args)
        self.logitLayer = nn.Linear(base_dim//4, 1)
        self.logitLayer.apply(init_weights)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        tau_logit = self.logitLayer(inputs)
        # return self.tanh(p)
        return tau_logit

class DESCN(nn.Module):
    def __init__(self,  args, device):
        super(DESCN, self).__init__()
        # self.feature_extractor = feature_extractor
        self.shareNetwork = []
        self.shareNetwork.append(nn.Linear(args.x_dim, args.h_dim))
        for i in range(args.num_layers - 2):
            self.shareNetwork.append(nn.ELU())
            self.shareNetwork.append(nn.Linear(args.h_dim, args.h_dim))
        self.shareNetwork.append(nn.Linear(args.h_dim, args.h_dim//2))
        self.shareNetwork = nn.Sequential(*self.shareNetwork).to(device)
        # self.shareNetwork = ShareNetwork(args.x_dim, args.h_dim, args.h_dim//2, args, device).to(device)
        self.prpsy_network = PrpsyNetwork(args.h_dim//2, args=args).to(device)
        self.mu1_network = Mu1Network(args.h_dim//2, args=args).to(device)
        self.mu0_network = Mu0Network(args.h_dim//2, args=args).to(device)
        self.tau_network =TauNetwork(args.h_dim//2, args=args).to(device)
        self.args = args
        self.device = device
        self.to(device)

    def forward(self, inputs):
        shared_h = self.shareNetwork(inputs)

        # propensity output_logit
        p_prpsy_logit = self.prpsy_network(shared_h)

        # p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.05, 0.95)
        p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.001, 0.999)

        # logit for mu1, mu0
        mu1_logit = self.mu1_network(shared_h)
        mu0_logit = self.mu0_network(shared_h)

        # pseudo tau
        tau_logit = self.tau_network(shared_h)

        p_mu1 = sigmod2(mu1_logit)
        p_mu0 = sigmod2(mu0_logit)
        p_h1 = p_mu1 # Refer to the naming in TARnet/CFR
        p_h0 = p_mu0 # Refer to the naming in TARnet/CFR


        # entire space
        p_estr = torch.mul(p_prpsy, p_h1)
        p_i_prpsy = 1 - p_prpsy
        p_escr = torch.mul(p_i_prpsy, p_h0)

        return p_prpsy_logit.squeeze(), p_estr.squeeze(), p_escr.squeeze(), tau_logit.squeeze(),\
            mu1_logit.squeeze(), mu0_logit.squeeze(), p_prpsy.squeeze(), p_mu1.squeeze(), p_mu0.squeeze(),\
            p_h1.squeeze(), p_h0.squeeze(), shared_h.squeeze()

    def predict(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x,dtype=torch.float32)

        with torch.no_grad():
            p_prpsy_logit, p_estr, p_escr, tau_logit, mu1_logit, mu0_logit,\
            p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h = self.forward(x)

        return p_h0, p_h1

    def criterion(self, t, y,p_prpsy_logit, p_estr, p_escr, p_tau_logit, p_mu1_logit, p_mu0_logit, p_prpsy,\
                p_mu1, p_mu0, p_h1, p_h0, shared_h, args):
        e_labels = torch.zeros_like(t)

        p_t = torch.mean(t).item()
        if args.reweight_sample:
            w_t = t / (2 * p_t)
            w_c = (1 - t) / (2 * (1 - p_t))
            sample_weight = w_t + w_c
        else:
            sample_weight = torch.ones_like(t)
            p_t = 0.5

        sample_weight = sample_weight[~e_labels.bool()]
        loss_w_fn = nn.BCELoss(weight=sample_weight)
        loss_fn = nn.BCELoss()
        loss_mse = nn.MSELoss()
        loss_with_logit_fn = nn.BCEWithLogitsLoss()  # for logit
        loss_w_with_logit_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(1 / (2 * p_t)))  # for propensity loss

        # loss for propensity
        prpsy_loss = args.prpsy_w * loss_w_with_logit_fn(p_prpsy_logit[~e_labels.bool()],
                                                        t[~e_labels.bool()])
        # loss for ESTR, ESCR
        estr_loss = args.escvr1_w * loss_w_fn(p_estr[~e_labels.bool()],
                                                (y * t)[~e_labels.bool()])
        escr_loss = args.escvr0_w * loss_w_fn(p_escr[~e_labels.bool()],
                                                (y * (1 - t))[~e_labels.bool()])

        #loss for TR, CR
        tr_loss = args.h1_w * loss_fn(p_h1[t.bool()],
                                        y[t.bool()])  # * (1 / (2 * p_t))
        cr_loss = args.h0_w * loss_fn(p_h0[~t.bool()],
                                        y[~t.bool()])  # * (1 / (2 * (1 - p_t)))


        #loss for cross TR: mu1_prime, cross CR: mu0_prime
        cross_tr_loss = args.mu1hat_w * loss_fn(torch.sigmoid(p_mu0_logit + p_tau_logit)[t.bool()],
                                                y[t.bool()])
        cross_cr_loss = args.mu0hat_w * loss_fn(torch.sigmoid(p_mu1_logit - p_tau_logit)[~t.bool()],
                                                y[~t.bool()])

        total_loss = prpsy_loss + estr_loss + escr_loss + tr_loss + cr_loss + cross_tr_loss + cross_cr_loss
        return total_loss

    def train_model(self, opt, train_dataloader, valid_data, test_data, args, exp=None, best_model_path=None):
        self.train()

        early_stopper = EarlyStopper(patience=1, min_delta=0)
        best_val_value = -1.0
        test_cate_epochs = np.zeros((args.epochs, test_data.shape[0]))

        for epoch in range(args.epochs):
            for i, sample in enumerate(train_dataloader):
                
                sample = sample.to(torch.float32)
                x = sample[:, :args.x_dim]
                t = sample[:, args.x_dim]
                y = sample[:, args.x_dim+1]
                ''' Compute sample reweighting '''

                opt.zero_grad()
                p_prpsy_logit, p_estr, p_escr, p_tau_logit, p_mu1_logit, p_mu0_logit, p_prpsy,\
                p_mu1, p_mu0, p_h1, p_h0, shared_h = self(x)
                total_loss = self.criterion(t,y,p_prpsy_logit, p_estr, p_escr, p_tau_logit, p_mu1_logit,\
                                            p_mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h, args)
                # Backpropagation
                total_loss.backward()
                # total_loss.backward(retain_graph=True)
                # Update parameters
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
                    valid_p_prpsy_logit, valid_p_estr, valid_p_escr, valid_p_tau_logit, valid_p_mu1_logit,\
                    valid_p_mu0_logit, valid_p_prpsy, valid_p_mu1, valid_p_mu0, valid_p_h1, valid_p_h0,\
                    valid_shared_h = self(valid_x)
                    valid_loss = self.criterion(valid_t, valid_y,valid_p_prpsy_logit, valid_p_estr, valid_p_escr,\
                    valid_p_tau_logit, valid_p_mu1_logit,valid_p_mu0_logit, valid_p_prpsy, valid_p_mu1, valid_p_mu0,\
                    valid_p_h1, valid_p_h0,valid_shared_h, args)

                    if args.data == 'synthetic':
                        test_y0, test_y1 = self.predict(test_x)
                        test_cate = test_y1 - test_y0
                        # test_cate_epochs[epoch] = test_cate.squeeze().detach().numpy()
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
                    print(f"{args.model_name} epoch: {epoch} --- train_loss: {total_loss} --- valid_loss: {valid_loss} --- pred_tau: {pred_tau.mean()} --- valid_score: {valid_score}")
                if valid_score > best_val_value:
                    best_val_value = valid_score
                    torch.save(self.state_dict(), os.path.join(best_model_path, 'best_'+str(args.model_name)+'.pth'))
                if early_stopper.early_stop(valid_score):
                    print(f"{args.model_name} best model epoch: {epoch} --- train_loss: {total_loss} --- valid_score: {valid_score}")
                    return best_val_value
                self.train()
        # 如果是测50个ecpoh的变化趋势，返回这个数组
        # return test_cate_epochs
        #否则，返回最好的valid_score
        return best_val_value