# cfrnet
from data import data_loader
from sklearn.model_selection import KFold
import numpy as np
from utils import log, setup_seed, get_true_gain, get_true_gain_auc
# from utils import principled_uplift_auc_score, relative_uplift_auc_score, sep_qini_auc_score
# from sklift.metrics import qini_auc_score, uplift_auc_score

import matplotlib.pyplot as plt
from models.SLearner import SLearner
from models.TLearner import TLearner
from models.TARNet import TARNet
from models.CFR import CFR
from models.DragonNet import DragonNet
from models.EUEN import EUEN
from models.DESCN import DESCN
from models.EFIN import EFIN
from models.TONet import TONet
from models.TONet_v2 import TONetv2
from models.PTONet import PTONet
# from models.TONet_qini import TONet_qini
from models.TONet_pu import TONet_pu
import os
from argparse import ArgumentParser
import torch
# from scipy.stats import kendalltau
from models.model_utils import test_model
import time



def create_save_path(args):
    
    save_path = os.path.join(args.save_dir, args.data, args.model_name, str(args.valid_metric),
                str(args.lr)+'lr', str(args.h_dim)+'hdim', str(args.train_bs)+'bs', 
                str(args.epochs)+'e', str(args.num_layers)+'l')
    if args.model_name == 'dragonnet' or args.model_name == 'tonet':
        save_path = os.path.join(save_path, str(args.alpha)+'alpha', str(args.beta)+'beta')
    elif args.model_name == 'descn':
        save_path = os.path.join(save_path, str(args.dropout)+'dropout', str(args.prpsy_w)+'prpsy_w',
                    str(args.escvr1_w)+'escvr1_w',str(args.escvr0_w)+'escvr0_w',str(args.h1_w)+'h1_w',
                    str(args.h0_w)+'h0_w',str(args.mu0hat_w)+'mu0hat_w',str(args.mu1hat_w)+'mu1hat_w')
    # elif args.model_name == 'tonet_qini':
    #     save_path = os.path.join(save_path, str(args.alpha)+'alpha', str(args.beta)+'beta', str(args.gamma)+'gamma')
    elif args.model_name == 'tonet_pu':
        save_path = os.path.join(save_path, str(args.alpha)+'alpha', str(args.beta)+'beta', str(args.gamma)+'gamma')
    elif args.model_name == 'tonetv2':
        save_path = os.path.join(save_path, str(args.alpha)+'alpha')
    elif args.model_name == 'ptonet':
        save_path = os.path.join(save_path, str(args.alpha)+'alpha', str(args.beta)+'beta')
    save_path = os.path.join(save_path, args.model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return save_path

def create_model(args):
    if args.model_name == 'slearner':
        model = SLearner(args.x_dim+1, args.h_dim, args.num_layers)
    elif args.model_name == 'tlearner':
        model = TLearner(args.x_dim, args.h_dim, args.num_layers)
    elif args.model_name == 'tarnet':
        model = TARNet(args.x_dim, args.h_dim, args.num_layers)
    elif args.model_name == 'cfrnet':
        model = CFR(args.x_dim, args.out_dim, args)
    elif args.model_name == 'dragonnet':
        model = DragonNet(args.x_dim, args.h_dim, args.h_dim//2, args.num_layers, args.alpha, args.beta, 
                          args.epochs, args.train_bs, args.lr)
    elif args.model_name == 'euen':
        model = EUEN(args.x_dim, args.h_dim, args.num_layers)
    elif args.model_name == 'descn':
        device = 'cpu'
        model = DESCN(args, device)
        model = model.to(device)
    elif args.model_name == 'efin':
        model = EFIN(args.x_dim, args.h_dim, args.h_dim, args.num_layers)
    elif args.model_name == 'tonet':
        model = TONet(args.x_dim+1, args.h_dim, args.num_layers)
    elif args.model_name == 'tonetv2':
        model = TONetv2(args.x_dim, args.h_dim, args.num_layers)
    # elif args.model_name == 'tonet_qini':
    #     model = TONet_qini(args.x_dim+1, args.h_dim, args.num_layers)
    elif args.model_name == 'tonet_pu':
        model = TONet_pu(args.x_dim+1, args.h_dim, args.num_layers)
    elif args.model_name == 'ptonet':
        model = PTONet(args.x_dim, args.h_dim, args.num_layers)
    return model
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_exp', type=int, default=0)
    parser.add_argument('--root_path', type=str, default='/data/zhuminqin/PrincipleUplift')

    parser.add_argument('--data', type=str, default='synthetic')

    parser.add_argument('--save_dir', type=str, default='/data/zhuminqin/PrincipleUplift/log/2025-5-6')
    # parser.add_argument('--k_folds', type=int, default=0)
    # [256, 512, 1024, 2048]
    parser.add_argument('--train_bs', type=int, default=512)
    # parser.add_argument('--valid_bs', type=int, default=2048)
    # [1e-1, 1e-2, 1e-3, 1e-4]
    parser.add_argument('--lr', type=float, default=1e-2)
    # [16,32,64]
    parser.add_argument('--h_dim', type=int, default=32)

    parser.add_argument('--out_dim', type=int, default=1)
    # [5,6,7]
    parser.add_argument('--num_layers', type=int, default=5)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--BatchNorm1d', type=bool, default=True)
    parser.add_argument('--normalization', type=str, default='divide')
    parser.add_argument('--reweight_sample', type=bool, default=True)

    parser.add_argument('--prpsy_w', type=float, default=1)
    parser.add_argument('--escvr1_w', type=float, default=1)
    parser.add_argument('--escvr0_w', type=float, default=1)
    parser.add_argument('--h1_w', type=float, default=0)
    parser.add_argument('--h0_w', type=float, default=0)
    parser.add_argument('--mu0hat_w', type=float, default=1)
    parser.add_argument('--mu1hat_w', type=float, default=1)

    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--gamma', type=float, default=1)
    # [0.1, 0.5, 1, 5, 10]
    parser.add_argument('--alpha', type=float, default=0.5)
    # [0.1, 0.5, 1, 5, 10]
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--ipm_type', type=str, default='mmd_lin')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--model_name', type=str, default='ptonet')
    parser.add_argument('--valid_metric', type=str, default='pu')

    parser.add_argument('--x_dim', type=int, default=10, help="the dimension of covariates")

    args = parser.parse_args()

    exp = args.num_exp

    setup_seed(exp)
    start_time = time.time()
    # data_loader.data_split(args.root_path, args.data, is_valid=True)
    train_data, valid_data, test_data = data_loader.load_data(args.root_path, args.data, is_valid=True)


    trainloader = torch.utils.data.DataLoader(
                    train_data, 
                    batch_size=args.train_bs, shuffle=True)
    
    # validloader = torch.utils.data.DataLoader(
    #                 valid_data,
    #                 batch_size=args.valid_bs)
    best_val_metric = -1.0

    best_train_bs = None
    best_lr = None
    best_h_dim = None
    best_num_layers = None
    best_model_path = None

    train_bs_set = [512]
    lr_set = [1e-1]
    h_dim_set = [16]
    num_layers_set = [5]
    # train_bs_set = [2048]
    # lr_set = [1e-2]
    # h_dim_set = [64]
    # num_layers_set = [5]
    # alpha_set = [0.1, 0.5, 1, 5, 10]
    # beta_set = [0.1, 0.5, 1, 5, 10]
    # gamma_set = [0.1, 0.5, 1, 5, 10]
    # gamma_set = [2.0]
    for train_bs in train_bs_set:
        for lr in lr_set:
            for h_dim in h_dim_set:
                for num_layers in num_layers_set:
                    args.train_bs = train_bs
                    args.lr = lr
                    args.h_dim = h_dim
                    args.num_layers = num_layers

                    save_path = create_save_path(args)

                    model = create_model(args)
                    opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)
                    val_metric = model.train_model(opt, trainloader, valid_data, test_data, args, exp, save_path)
                    # valid qini
                    if val_metric > best_val_metric:
                        best_val_metric = val_metric
                        best_model_path = os.path.join(save_path, 'best_'+str(args.model_name)+'.pth')
                        best_train_bs = train_bs
                        best_lr = lr
                        best_h_dim = h_dim
                        best_num_layers = num_layers
 
    args.train_bs = best_train_bs
    args.lr = best_lr
    args.h_dim = best_h_dim
    args.num_layers = best_num_layers
    model = create_model(args)
    model.eval()
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_X = test_data[:,:args.x_dim]
    test_t = test_data[:,args.x_dim:args.x_dim+1]
    param = torch.load(best_model_path)
    model.load_state_dict(param)
    if args.data.startswith('synthetic'):
        pred_tau, pehe, pred_relative_uplift, pred_sep_qini,\
        pred_joint_uplift_score, pred_joint_qini_score, pred_pu_score,\
        true_relative_uplift, true_sep_qini, true_joint_uplift, true_joint_qini,\
        true_pu_score, true_gains, pred_gains = test_model(model, args=args, test_data=test_data,save_path=best_model_path)
        print(f'best_model_path:{best_model_path}')
        print(f'pehe:{pehe}')
        print(f'pred_relative_uplift:{pred_relative_uplift}')
        print(f'pred_sep_qini:{pred_sep_qini}')
        print(f'pred_joint_uplift_score:{pred_joint_uplift_score}')
        print(f'pred_joint_qini_score:{pred_joint_qini_score}')
        print(f'pred_pu_score:{pred_pu_score}')
        print(f'true_relative_uplift:{true_relative_uplift}')
        print(f'true_sep_qini:{true_sep_qini}')
        print(f'true_joint_uplift:{true_joint_uplift}')
        print(f'true_joint_qini:{true_joint_qini}')
        print(f'true_pu_score:{true_pu_score}')
        print(f'true_gain_top10:{true_gains[0]}, true_gain_top20:{true_gains[1]}, true_gain_top30:{true_gains[2]}, true_gain_top40:{true_gains[3]}, true_gain_top50:{true_gains[4]}')
        print(f'pred_gain_top10:{pred_gains[0]}, pred_gain_top20:{pred_gains[1]}, pred_gain_top30:{pred_gains[2]}, pred_gain_top40:{pred_gains[3]}, pred_gain_top50:{pred_gains[4]}')

        log_save_path = os.path.join(args.save_dir, args.data, args.model_name)
        log(log_save_path, 'best_model_path:'+str(best_model_path)+' pehe:'+str(pehe)+' pred_relative_uplift:'+str(pred_relative_uplift)+' pred_sep_qini:'+str(pred_sep_qini)\
            +' pred_joint_uplift_score:'+str(pred_joint_uplift_score)\
            +' pred_joint_qini_score:'+str(pred_joint_qini_score)\
            +' pred_pu_score:'+str(pred_pu_score)\
            +' true_relative_uplift:'+str(true_relative_uplift)\
            +' true_sep_qini:'+str(true_sep_qini)\
            +' true_joint_uplift:'+str(true_joint_uplift)\
            +' true_joint_qini:'+str(true_joint_qini)\
            +' true_pu_score:'+str(true_pu_score)\
            +' true_gain_top1:'+str(true_gains[0])+' true_gain_top2:'+str(true_gains[1])+' true_gain_top3:'+str(true_gains[2])+' true_gain_top4:'+str(true_gains[3])+' true_gain_top5:'+str(true_gains[4])\
            +' pred_gain_top1:'+str(pred_gains[0])+' pred_gain_top2:'+str(pred_gains[1])+' pred_gain_top3:'+str(pred_gains[2])+' pred_gain_top4:'+str(pred_gains[3])+' pred_gain_top5:'+str(pred_gains[4])\
            +'\n',
            args.model_name+'.txt')
    elif args.data == 'criteo':
        pred_tau, pred_relative_uplift, pred_sep_qini,\
        pred_joint_uplift_score, pred_joint_qini_score, pred_pu_score = test_model(model, args=args, test_data=test_data,save_path=best_model_path)
        print(f'best_model_path:{best_model_path}')
        print(f'pred_relative_uplift:{pred_relative_uplift}')
        print(f'pred_sep_qini:{pred_sep_qini}')
        print(f'pred joint uplift_score:{pred_joint_uplift_score}')
        print(f'pred joint qini_score:{pred_joint_qini_score}')
        print(f'pred_pu_score:{pred_pu_score}')
        log_save_path = os.path.join(args.save_dir, args.data, args.model_name)
        log(log_save_path, 'best_model_path'+str(best_model_path)+' pred_relative_uplift:'+str(pred_relative_uplift)+' pred_sep_qini:'+str(pred_sep_qini)\
            +' pred joint uplift_score:'+str(pred_joint_uplift_score)\
            +' pred joint qini_score:'+str(pred_joint_qini_score)\
            +' pred_pu_score:'+str(pred_pu_score)\
            +'\n',
            args.model_name+'.txt')
    # log(log_save_path, '\n', args.model_name+'.txt')
    end_time = time.time()
    print(f'time:{end_time-start_time}')