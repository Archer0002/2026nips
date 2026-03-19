import numpy as np

import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from models.model_utils import test_model
from models.earlystop import EarlyStopper
import os

class EFIN(nn.Module):
    """
    EFIN class -- a explicit feature interaction network with two heads.
    """
    def __init__(self, input_dim, hc_dim, hu_dim, num_layers, is_self=None, act_type='elu'):
        super(EFIN, self).__init__()
        self.nums_feature = input_dim
        self.is_self = is_self

        # interaction attention
        self.att_embed_1 = nn.Linear(hu_dim, hu_dim, bias=False)
        self.att_embed_2 = nn.Linear(hu_dim, hu_dim)
        self.att_embed_3 = nn.Linear(hu_dim, 1, bias=False)

        # self-attention
        self.softmax = nn.Softmax(dim=-1)
        self.Q_w = nn.Linear(in_features=hu_dim, out_features=hu_dim, bias=True)
        self.K_w = nn.Linear(in_features=hu_dim, out_features=hu_dim, bias=True)
        self.V_w = nn.Linear(in_features=hu_dim, out_features=hu_dim, bias=True)

        # representation parts for X
        self.x_rep = nn.Embedding(input_dim, hu_dim)

        # representation parts for T
        self.t_rep = nn.Linear(1, hu_dim)

        self.layer_cfg = [hc_dim,hc_dim,hc_dim//2,hc_dim//2,hc_dim//4]
        '''control net'''
        self.c_fc = []
        self.c_fc.append(nn.Linear(input_dim * hu_dim, hc_dim))
        for i in range(num_layers - 5):
            self.c_fc.append(nn.ELU())
            self.c_fc.append(nn.Linear(hc_dim, hc_dim))
        for i in range(1, len(self.layer_cfg)):
            self.c_fc.append(nn.ELU())
            self.c_fc.append(nn.Linear(self.layer_cfg[i-1], self.layer_cfg[i]))
        self.c_fc = nn.Sequential(*self.c_fc)
        out_dim = hc_dim // 4

        self.c_logit = nn.Linear(out_dim, 1)
        self.c_tau = nn.Linear(out_dim, 1)

        '''uplift net'''
        self.layer_cfg = [hu_dim,hu_dim,hu_dim//2,hu_dim//2,hu_dim//4]
        self.u_fc = []
        self.u_fc.append(nn.Linear(hu_dim, hu_dim))
        for i in range(num_layers - 5):
            self.u_fc.append(nn.ELU())
            self.u_fc.append(nn.Linear(hu_dim, hu_dim))
        for i in range(1, len(self.layer_cfg)):
            self.u_fc.append(nn.ELU())
            self.u_fc.append(nn.Linear(self.layer_cfg[i-1], self.layer_cfg[i]))
        self.u_fc = nn.Sequential(*self.u_fc)
        out_dim = hu_dim // 4
        if self.is_self:
            self.u_fc4 = nn.Linear(hu_dim // 4, hu_dim // 8)
            out_dim = hu_dim // 8
        self.t_logit = nn.Linear(out_dim, 1)
        self.u_tau = nn.Linear(out_dim, 1)

        # activation function
        if act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'tanh':
            self.act = nn.Tanh()
        elif act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'elu':
            self.act = nn.ELU()
        else:
            raise RuntimeError('unknown act_type {0}'.format(act_type))

    def self_attn(self, q, k, v):
        Q, K, V = self.Q_w(q), self.K_w(k), self.V_w(v)
        attn_weights = Q.matmul(torch.transpose(K, 1, 2)) / (K.shape[-1] ** 0.5)
        attn_weights = self.softmax(torch.sigmoid(attn_weights))

        outputs = attn_weights.matmul(V)

        return outputs, attn_weights

    def interaction_attn(self, t, x):
        attention = []
        for i in range(self.nums_feature):
            temp = self.att_embed_3(torch.relu(
                torch.sigmoid(self.att_embed_1(t)) + torch.sigmoid(self.att_embed_2(x[:, i, :]))))
            attention.append(temp)
        attention = torch.squeeze(torch.stack(attention, 1), 2)
        # print('interaction attention', attention)
        attention = torch.softmax(attention, 1)
        # print('mean interaction attention', torch.mean(attention, 0))

        outputs = torch.squeeze(torch.matmul(torch.unsqueeze(attention, 1), x), 1)
        return outputs, attention

    def forward(self, feature_list, is_treat):
        t_true = torch.unsqueeze(is_treat, 1)

        x_rep = feature_list.unsqueeze(2) * self.x_rep.weight.unsqueeze(0)

        # control net
        dims = x_rep.size()
        _x_rep = x_rep / torch.linalg.norm(x_rep, dim=1, keepdim=True)
        xx, xx_weight = self.self_attn(_x_rep, _x_rep, _x_rep)

        _x_rep = torch.reshape(xx, (dims[0], dims[1] * dims[2]))

        # c_last = self.act(self.c_fc4(self.act(self.c_fc3(self.act(self.c_fc2(self.act(self.c_fc1(_x_rep))))))))
        # if self.is_self:
        #     c_last = self.act(self.c_fc5(c_last))
        c_last = self.act(self.c_fc(_x_rep))
        c_logit = self.c_logit(c_last)
        c_tau = self.c_tau(c_last)
        c_prob = torch.sigmoid(c_logit)

        # uplift net
        t_rep = self.t_rep(torch.ones_like(t_true))

        xt, xt_weight = self.interaction_attn(t_rep, x_rep)

        # u_last = self.act(self.u_fc3(self.act(self.u_fc2(self.act(self.u_fc1(xt))))))
        # if self.is_self:
        #     u_last = self.act(self.u_fc4(u_last))
        u_last = self.act(self.u_fc(xt))
        t_logit = self.t_logit(u_last)
        u_tau = self.u_tau(u_last)
        t_prob = torch.sigmoid(t_logit)

        return c_logit, c_prob, c_tau, t_logit, t_prob, u_tau

    def criterion(self, feature_list, is_treat, label_list):
        # Model outputs
        c_logit, c_prob, c_tau, t_logit, t_prob, u_tau = self.forward(feature_list, is_treat)

        # regression
        c_logit_fix = c_logit.detach()
        uc = c_logit
        ut = (c_logit_fix + u_tau)

        y_true = torch.unsqueeze(label_list, 1)
        t_true = torch.unsqueeze(is_treat, 1)

        # response loss
        bce = torch.nn.BCEWithLogitsLoss(reduction='mean')

        temp = torch.square((1 - t_true) * uc + t_true * ut - y_true)
        loss1 = torch.mean(temp)
        loss2 = bce(t_logit, 1 - t_true)
        loss = loss1 + loss2

        return loss
    
    def predict(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x,dtype=torch.float32)
        t0 = torch.zeros((x.shape[0]))
        # t1 = torch.ones((x.shape[0],1))

        with torch.no_grad():
            c_logit, c_prob, c_tau, t_logit, t_prob, u_tau = self.forward(x, t0.squeeze())
        #    c_logit, c_prob, c_tau, t_logit, t_prob, u_tau = self.forward(x, t1)

        # return c_prob, t_prob
        return c_logit, c_logit + u_tau
    
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
                loss = self.criterion(x,t,y)
                loss.backward()
                opt.step()

            if (epoch+1) % 1 == 0:
                self.eval()
                if not torch.is_tensor(valid_data):
                    valid_data = torch.tensor(valid_data, dtype=torch.float32)
                else:
                    valid_data = valid_data.to(torch.float32)

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
                    elif args.data == 'criteo' or args.data =='lzd':
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
                    print(f"{args.model_name} best model epoch: {epoch} --- train_loss: {loss} --- valid_score: {valid_score}")
                    return best_val_value
                self.train()
        # 如果是测50个ecpoh的变化趋势，返回这个数组
        # return test_cate_epochs
        #否则，返回最好的valid_score
        return best_val_value
    