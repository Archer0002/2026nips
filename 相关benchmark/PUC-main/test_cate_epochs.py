# import the module

from data import data_loader
from sklearn.model_selection import KFold
import numpy as np
from utils import log, setup_seed, get_true_gain, get_true_gain_auc
from utils import principled_uplift_auc_score, relative_uplift_auc_score, sep_qini_auc_score
from sklift.metrics import qini_auc_score, uplift_auc_score

import matplotlib.pyplot as plt
from models.SLearner import SLearner
from models.TLearner import TLearner
from models.TARNet import TARNet
from models.DUMN import DUMN
from models.DUMNv2 import DUMNv2
from models.PSN import PSN
from models.CFR import CFR
from models.DragonNet import DragonNet
from models.EUEN import EUEN
from models.DESCN import DESCN
from models.EFIN import EFIN
import os
from argparse import ArgumentParser
import torch
from scipy.stats import kendalltau

def create_save_path(args):
    
    save_path = os.path.join(args.save_dir, args.data, args.model_name, str(args.valid_metric),
                str(args.lr)+'lr', str(args.h_dim)+'hdim', str(args.train_bs)+'bs', 
                str(args.epochs)+'e')
    if args.model_name == 'dragonnet':
        save_path = os.path.join(save_path, str(args.alpha)+'alpha', str(args.beta)+'beta')
    elif args.model_name == 'descn':
        save_path = os.path.join(save_path, str(args.dropout)+'dropout', str(args.prpsy_w)+'prpsy_w',
                    str(args.escvr1_w)+'escvr1_w',str(args.escvr0_w)+'escvr0_w',str(args.h1_w)+'h1_w',
                    str(args.h0_w)+'h0_w',str(args.mu0hat_w)+'mu0hat_w',str(args.mu1hat_w)+'mu1hat_w')

    save_path = os.path.join(save_path, args.model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return save_path

def create_model(args):
    if args.model_name == 'slearner':
        model = SLearner(args.x_dim+1, args.h_dim)
    elif args.model_name == 'tlearner':
        model = TLearner(args.x_dim, args.h_dim)
    elif args.model_name == 'tarnet':
        model = TARNet(args.x_dim, args.h_dim)
    elif args.model_name == 'cfrnet':
        model = CFR(args.x_dim, args.out_dim, args)
    elif args.model_name == 'dumn':
        model = DUMN(args.x_dim, args.h_dim)
    elif args.model_name == 'dumnv2':
        model = DUMNv2(args.x_dim, args.h_dim)
    elif args.model_name == 'psn':
        model = PSN(args.x_dim, args.h_dim)
    elif args.model_name == 'dragonnet':
        model = DragonNet(args.x_dim, args.h_dim, args.h_dim//2, args.alpha, args.beta, 
                          args.epochs, args.train_bs, args.lr)
    elif args.model_name == 'euen':
        model = EUEN(args.x_dim, args.h_dim)
    elif args.model_name == 'descn':
        device = 'cpu'
        model = DESCN(args, device)
        model = model.to(device)
    elif args.model_name == 'efin':
        model = EFIN(args.x_dim, args.h_dim, args.h_dim)
    return model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_exp', type=int, default=0)

    parser.add_argument('--root_path', type=str, default='D:\code\ecrank\principle-uplift-v1')

    parser.add_argument('--data', type=str, default='syntheticv8')

    parser.add_argument('--save_dir', type=str, default=r'D:\\code\\ecrank\\principle-uplift-v1\\log\\2024-5-15')
    # parser.add_argument('--k_folds', type=int, default=0)
    # [256, 512, 1024,2048]
    parser.add_argument('--train_bs', type=int, default=1024)
    # parser.add_argument('--valid_bs', type=int, default=2048)
    # [1e-1, 1e-2, 1e-3, 1e-4]
    # 1e-1 直接不收敛，不用跑
    parser.add_argument('--lr', type=float, default=1e-2)
    # [16,32,64]
    parser.add_argument('--h_dim', type=int, default=16)

    parser.add_argument('--out_dim', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=7)
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
    parser.add_argument('--alpha', type=float, default=1)
    # [0.1, 0.5, 1, 5, 10]
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--ipm_type', type=str, default='mmd_lin')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--model_name', type=str, default='slearner')
    parser.add_argument('--valid_metric', type=str, default='qini')

    parser.add_argument('--x_dim', type=int, default=10, help="the dimension of covariates")

    args = parser.parse_args()

    save_path = create_save_path(args)
    exp = args.num_exp

    setup_seed(0)
    # data_loader.data_split(args.root_path, args.data, is_valid=True)
    train_data, valid_data, test_data = data_loader.load_data(args.root_path, args.data, is_valid=True)


    trainloader = torch.utils.data.DataLoader(
                    train_data, 
                    batch_size=args.train_bs, shuffle=True)
    
    # validloader = torch.utils.data.DataLoader(
    #                 valid_data,
    #                 batch_size=args.valid_bs)
    model = create_model(args)
    opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    test_cate_epochs = model.train_model(opt, trainloader, valid_data, test_data, args, exp, save_path)


    # 按照每个指标，对model进行排序，取前k个model，计算 10%，20%, 50% 的gain
    percentage = 1

    # desc_test_cate_idxs = np.zeros((args.epochs,test_data.shape[0]))

    test_sep_uplifts = np.zeros((args.epochs,))
    test_sep_qinis = np.zeros((args.epochs,))
    test_joint_uplifts = np.zeros((args.epochs,))
    test_joint_qinis = np.zeros((args.epochs,))
    test_pu_scores = np.zeros((args.epochs,))
    test_pred_gains = np.zeros((5,args.epochs))
    test_pred_gains_auc = np.zeros((5,args.epochs))

    for i, cates in enumerate(test_cate_epochs):

        # 对每个epoch的cate，计算uplift和qini
        test_sep_uplifts[i] = relative_uplift_auc_score(test_data[:,args.x_dim+1], cates.squeeze(), test_data[:,args.x_dim])
        test_sep_qinis[i] = sep_qini_auc_score(test_data[:,args.x_dim+1], cates.squeeze(), test_data[:,args.x_dim])
        test_joint_uplifts[i] = uplift_auc_score(test_data[:,args.x_dim+1], cates.squeeze(), test_data[:,args.x_dim])
        test_joint_qinis[i] = qini_auc_score(test_data[:,args.x_dim+1], cates.squeeze(), test_data[:,args.x_dim])
        test_pu_scores[i] = principled_uplift_auc_score(test_data[:,args.x_dim+1], cates.squeeze(), test_data[:,args.x_dim])
        # for j in range(5):
        #     test_pred_gains[j,i] = get_true_gain(test_data, cates, (j+1)*0.1)
        #     test_pred_gains_auc[j,i] = get_true_gain_auc(test_data, cates, (j+1)*0.1)
        test_pred_gains[0,i] = get_true_gain(test_data, cates, 0.1)
        test_pred_gains[1,i] = get_true_gain(test_data, cates, 0.3)
        test_pred_gains[2,i] = get_true_gain(test_data, cates, 0.5)
        test_pred_gains[3,i] = get_true_gain(test_data, cates, 0.7)
        test_pred_gains[4,i] = get_true_gain(test_data, cates, 1.0)
        test_pred_gains_auc[0,i] = get_true_gain_auc(test_data, cates, 0.1)
        test_pred_gains_auc[1,i] = get_true_gain_auc(test_data, cates, 0.3)
        test_pred_gains_auc[2,i] = get_true_gain_auc(test_data, cates, 0.5)
        test_pred_gains_auc[3,i] = get_true_gain_auc(test_data, cates, 0.7)
        test_pred_gains_auc[4,i] = get_true_gain_auc(test_data, cates, 1.0)

    desc_sep_uplift_idx = np.argsort(test_sep_uplifts, kind="mergesort", axis=0)[::-1]
    desc_sep_qini_idx = np.argsort(test_sep_qinis, kind="mergesort", axis=0)[::-1]
    desc_joint_uplift_idx = np.argsort(test_joint_uplifts, kind="mergesort", axis=0)[::-1]
    desc_joint_qini_idx = np.argsort(test_joint_qinis, kind="mergesort", axis=0)[::-1]
    desc_pu_score_idx = np.argsort(test_pu_scores, kind="mergesort", axis=0)[::-1]

    # percentage = 1

    gain_sep_uplift_100n = get_true_gain_auc(test_data, test_cate_epochs[desc_sep_uplift_idx[0]], percentage)
    gain_sep_qini_100n = get_true_gain_auc(test_data, test_cate_epochs[desc_sep_qini_idx[0]], percentage)
    gain_joint_uplift_100n = get_true_gain_auc(test_data, test_cate_epochs[desc_joint_uplift_idx[0]], percentage)
    gain_joint_qini_100n = get_true_gain_auc(test_data, test_cate_epochs[desc_joint_qini_idx[0]], percentage)
    gain_pu_score_100n = get_true_gain_auc(test_data, test_cate_epochs[desc_pu_score_idx[0]], percentage)
    print('model num: 1')
    print(f'desc_sep_uplift_idx:{desc_sep_uplift_idx[0]}, desc_sep_qini_idx:{desc_sep_qini_idx[0]}, desc_joint_uplift_idx:{desc_joint_uplift_idx[0]}, desc_joint_qini_idx:{desc_joint_qini_idx[0]}, desc_pu_score_idx:{desc_pu_score_idx[0]}')
    print(f'gain_sep_uplift_100n:{gain_sep_uplift_100n}, gain_sep_qini_100n:{gain_sep_qini_100n}, gain_joint_uplift_100n:{gain_joint_uplift_100n}, gain_joint_qini_100n:{gain_joint_qini_100n}, gain_pu_score_100n:{gain_pu_score_100n}')

    # 按照每个指标，对model进行排序，取前5个model，计算 5%， 10%，50% 的gain
    model_num = 5
    gain_sep_uplift_100n = 0
    gain_sep_qini_100n = 0
    gain_joint_uplift_100n = 0
    gain_joint_qini_100n = 0
    gain_pu_score_100n = 0
    for i in range(model_num):
        gain_sep_uplift_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_sep_uplift_idx[i]], percentage)
        gain_sep_qini_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_sep_qini_idx[i]], percentage)
        gain_joint_uplift_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_joint_uplift_idx[i]], percentage)
        gain_joint_qini_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_joint_qini_idx[i]], percentage)
        gain_pu_score_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_pu_score_idx[i]], percentage)

    print(f'model num: {model_num}')
    print(f'desc_sep_uplift_idx:{desc_sep_uplift_idx[:model_num]}, desc_sep_qini_idx:{desc_sep_qini_idx[:model_num]}, desc_joint_uplift_idx:{desc_joint_uplift_idx[:model_num]}, desc_joint_qini_idx:{desc_joint_qini_idx[:model_num]}, desc_pu_score_idx:{desc_pu_score_idx[:model_num]}')
    print(f'gain_sep_uplift_100n:{gain_sep_uplift_100n}, gain_sep_qini_100n:{gain_sep_qini_100n}, gain_joint_uplift_100n:{gain_joint_uplift_100n}, gain_joint_qini_100n:{gain_joint_qini_100n}, gain_pu_score_100n:{gain_pu_score_100n}')

    # 按照每个指标，对model进行排序，取前10个model，计算 5%， 10%，20% 的gain
    model_num = 10
    gain_sep_uplift_100n = 0
    gain_sep_qini_100n = 0
    gain_joint_uplift_100n = 0
    gain_joint_qini_100n = 0
    gain_pu_score_100n = 0
    for i in range(model_num):
        gain_sep_uplift_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_sep_uplift_idx[i]], percentage)
        gain_sep_qini_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_sep_qini_idx[i]], percentage)
        gain_joint_uplift_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_joint_uplift_idx[i]], percentage)
        gain_joint_qini_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_joint_qini_idx[i]], percentage)
        gain_pu_score_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_pu_score_idx[i]], percentage)

    print(f'model num: {model_num}')
    print(f'desc_sep_uplift_idx:{desc_sep_uplift_idx[:model_num]}, desc_sep_qini_idx:{desc_sep_qini_idx[:model_num]}, desc_joint_uplift_idx:{desc_joint_uplift_idx[:model_num]}, desc_joint_qini_idx:{desc_joint_qini_idx[:model_num]}, desc_pu_score_idx:{desc_pu_score_idx[:model_num]}')
    print(f'gain_sep_uplift_100n:{gain_sep_uplift_100n}, gain_sep_qini_100n:{gain_sep_qini_100n}, gain_joint_uplift_100n:{gain_joint_uplift_100n}, gain_joint_qini_100n:{gain_joint_qini_100n}, gain_pu_score_100n:{gain_pu_score_100n}')


    # 按照每个指标，对model进行排序，取前10个model，计算 5%， 10%，20% 的gain
    model_num = 20
    gain_sep_uplift_100n = 0
    gain_sep_qini_100n = 0
    gain_joint_uplift_100n = 0
    gain_joint_qini_100n = 0
    gain_pu_score_100n = 0
    for i in range(model_num):
        gain_sep_uplift_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_sep_uplift_idx[i]], percentage)
        gain_sep_qini_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_sep_qini_idx[i]], percentage)
        gain_joint_uplift_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_joint_uplift_idx[i]], percentage)
        gain_joint_qini_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_joint_qini_idx[i]], percentage)
        gain_pu_score_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_pu_score_idx[i]], percentage)

    print(f'model num: {model_num}')
    print(f'desc_sep_uplift_idx:{desc_sep_uplift_idx[:model_num]}, desc_sep_qini_idx:{desc_sep_qini_idx[:model_num]}, desc_joint_uplift_idx:{desc_joint_uplift_idx[:model_num]}, desc_joint_qini_idx:{desc_joint_qini_idx[:model_num]}, desc_pu_score_idx:{desc_pu_score_idx[:model_num]}')
    print(f'gain_sep_uplift_100n:{gain_sep_uplift_100n}, gain_sep_qini_100n:{gain_sep_qini_100n}, gain_joint_uplift_100n:{gain_joint_uplift_100n}, gain_joint_qini_100n:{gain_joint_qini_100n}, gain_pu_score_100n:{gain_pu_score_100n}')


    # 按照每个指标，对model进行排序，取前10个model，计算 5%， 10%，20% 的gain
    model_num = 30
    gain_sep_uplift_100n = 0
    gain_sep_qini_100n = 0
    gain_joint_uplift_100n = 0
    gain_joint_qini_100n = 0
    gain_pu_score_100n = 0
    for i in range(model_num):
        gain_sep_uplift_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_sep_uplift_idx[i]], percentage)
        gain_sep_qini_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_sep_qini_idx[i]], percentage)
        gain_joint_uplift_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_joint_uplift_idx[i]], percentage)
        gain_joint_qini_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_joint_qini_idx[i]], percentage)
        gain_pu_score_100n += get_true_gain_auc(test_data, test_cate_epochs[desc_pu_score_idx[i]], percentage)

    print(f'model num: {model_num}')
    print(f'desc_sep_uplift_idx:{desc_sep_uplift_idx[:model_num]}, desc_sep_qini_idx:{desc_sep_qini_idx[:model_num]}, desc_joint_uplift_idx:{desc_joint_uplift_idx[:model_num]}, desc_joint_qini_idx:{desc_joint_qini_idx[:model_num]}, desc_pu_score_idx:{desc_pu_score_idx[:model_num]}')
    print(f'gain_sep_uplift_100n:{gain_sep_uplift_100n}, gain_sep_qini_100n:{gain_sep_qini_100n}, gain_joint_uplift_100n:{gain_joint_uplift_100n}, gain_joint_qini_100n:{gain_joint_qini_100n}, gain_pu_score_100n:{gain_pu_score_100n}')

    # test
    model.eval()
    test_data = torch.tensor(test_data, dtype=torch.float32)

    test_X = test_data[:,:args.x_dim]
    test_t = test_data[:,args.x_dim:args.x_dim+1]
    param = torch.load(os.path.join(save_path, 'best_'+str(args.model_name)+'.pth'))
    model.load_state_dict(param)
    pred_tau, qini_score, uplift_score, pr_qini, sdr_qini, pr_sdr_qini, pr_uplift, sdr_uplift, pr_sdr_uplift,\
    true_tau_qini, true_pr_qini, true_sdr_qini,\
    true_tau_uplift, true_pr_uplift, true_sdr_uplift,\
    pehe, acc, persuadable_acc, sleepingdog_acc,\
    pred_cate_kendall, true_cate_kendall,\
    pred_pr_kendall, true_pr_kendall,\
    pred_sdr_kendall, true_sdr_kendall,\
    true_principle_rank_kendall, true_principle_rank_qini, true_principle_rank_uplift,\
    pred_principle_rank_kendall, pred_principle_rank_qini, pred_principle_rank_uplift,\
    pred_pu_score, true_pu_score, pred_relative_uplift, true_relative_uplift,\
    pred_sep_qini, true_sep_qini, true_gains, pred_gains = model.test_model(args=args, test_data=test_data,save_path=save_path)
    sep_uplift_gain_kendalls = []
    sep_uplift_gain_p_values = []
    sep_qini_gain_kendalls = []
    sep_qini_gain_p_values = []
    joint_uplift_gain_kendalls = []
    joint_uplift_gain_p_values = []
    joint_qini_gain_kendalls = []
    joint_qini_gain_p_values = []
    pu_scores_gain_kendalls = []
    pu_scores_gain_p_values = []

    for i in range(test_pred_gains_auc.shape[0]):
        kendall, p_value = kendalltau(test_sep_uplifts, test_pred_gains_auc[i])
        sep_uplift_gain_kendalls.append(kendall)
        sep_uplift_gain_p_values.append(p_value)
        kendall, p_value = kendalltau(test_sep_qinis, test_pred_gains_auc[i])
        sep_qini_gain_kendalls.append(kendall)
        sep_qini_gain_p_values.append(p_value)
        kendall, p_value = kendalltau(test_joint_uplifts, test_pred_gains_auc[i])
        joint_uplift_gain_kendalls.append(kendall)
        joint_uplift_gain_p_values.append(p_value)
        kendall, p_value = kendalltau(test_joint_qinis, test_pred_gains_auc[i])
        joint_qini_gain_kendalls.append(kendall)
        joint_qini_gain_p_values.append(p_value)
        kendall, p_value = kendalltau(test_pu_scores, test_pred_gains_auc[i])
        pu_scores_gain_kendalls.append(kendall)
        pu_scores_gain_p_values.append(p_value)
    

    # sep_uplifts_gain10_kendall, sep_uplifts_gain10_p_value = kendalltau(model.sep_uplifts, model.gain10)
    # sep_uplifts_gain20_kendall, sep_uplifts_gain20_p_value = kendalltau(model.sep_uplifts, model.gain20)
    # sep_uplifts_gain30_kendall, sep_uplifts_gain30_p_value = kendalltau(model.sep_uplifts, model.gain30)
    # sep_uplifts_gain40_kendall, sep_uplifts_gain40_p_value = kendalltau(model.sep_uplifts, model.gain40)
    # sep_uplifts_gain50_kendall, sep_uplifts_gain50_p_value = kendalltau(model.sep_uplifts, model.gain50)
    # sep_qinis_gain10_kendall, sep_qinis_gain10_p_value = kendalltau(model.sep_qinis, model.gain10)
    # sep_qinis_gain20_kendall, sep_qinis_gain20_p_value = kendalltau(model.sep_qinis, model.gain20)
    # sep_qinis_gain30_kendall, sep_qinis_gain30_p_value = kendalltau(model.sep_qinis, model.gain30)
    # sep_qinis_gain40_kendall, sep_qinis_gain40_p_value = kendalltau(model.sep_qinis, model.gain40)
    # sep_qinis_gain50_kendall, sep_qinis_gain50_p_value = kendalltau(model.sep_qinis, model.gain50)
    # joint_uplifts_gain10_kendall, joint_uplifts_gain10_p_value = kendalltau(model.joint_uplifts, model.gain10)
    # joint_uplifts_gain20_kendall, joint_uplifts_gain20_p_value = kendalltau(model.joint_uplifts, model.gain20)
    # joint_uplifts_gain30_kendall, joint_uplifts_gain30_p_value = kendalltau(model.joint_uplifts, model.gain30)
    # joint_uplifts_gain40_kendall, joint_uplifts_gain40_p_value = kendalltau(model.joint_uplifts, model.gain40)
    # joint_uplifts_gain50_kendall, joint_uplifts_gain50_p_value = kendalltau(model.joint_uplifts, model.gain50)
    # joint_qinis_gain10_kendall, joint_qinis_gain10_p_value = kendalltau(model.joint_qinis, model.gain10)
    # joint_qinis_gain20_kendall, joint_qinis_gain20_p_value = kendalltau(model.joint_qinis, model.gain20)
    # joint_qinis_gain30_kendall, joint_qinis_gain30_p_value = kendalltau(model.joint_qinis, model.gain30)
    # joint_qinis_gain40_kendall, joint_qinis_gain40_p_value = kendalltau(model.joint_qinis, model.gain40)
    # joint_qinis_gain50_kendall, joint_qinis_gain50_p_value = kendalltau(model.joint_qinis, model.gain50)
    # pu_scores_gain10_kendall, pu_scores_gain10_p_value = kendalltau(model.pu_scores, model.gain10)
    # pu_scores_gain20_kendall, pu_scores_gain20_p_value = kendalltau(model.pu_scores, model.gain20)
    # pu_scores_gain30_kendall, pu_scores_gain30_p_value = kendalltau(model.pu_scores, model.gain30)
    # pu_scores_gain40_kendall, pu_scores_gain40_p_value = kendalltau(model.pu_scores, model.gain40)
    # pu_scores_gain50_kendall, pu_scores_gain50_p_value = kendalltau(model.pu_scores, model.gain50)

    print(f'sep_uplifts_gain10_kendall:{sep_uplift_gain_kendalls[0]}, sep_uplifts_gain20_kendall:{sep_uplift_gain_kendalls[1]}, sep_uplifts_gain30_kendall:{sep_uplift_gain_kendalls[2]}, sep_uplifts_gain40_kendall:{sep_uplift_gain_kendalls[3]}, sep_uplifts_gain50_kendall:{sep_uplift_gain_kendalls[4]}')
    print(f'sep_uplifts_gain10_p_value:{sep_uplift_gain_p_values[0]}, sep_uplifts_gain20_p_value:{sep_uplift_gain_p_values[1]}, sep_uplifts_gain30_p_value:{sep_uplift_gain_p_values[2]}, sep_uplifts_gain40_p_value:{sep_uplift_gain_p_values[3]}, sep_uplifts_gain50_p_value:{sep_uplift_gain_p_values[4]}')
    print(f'sep_qinis_gain10_kendall:{sep_qini_gain_kendalls[0]}, sep_qinis_gain20_kendall:{sep_qini_gain_kendalls[1]}, sep_qinis_gain30_kendall:{sep_qini_gain_kendalls[2]}, sep_qinis_gain40_kendall:{sep_qini_gain_kendalls[3]}, sep_qinis_gain50_kendall:{sep_qini_gain_kendalls[4]}')
    print(f'sep_qinis_gain10_p_value:{sep_qini_gain_p_values[0]}, sep_qinis_gain20_p_value:{sep_qini_gain_p_values[1]}, sep_qinis_gain30_p_value:{sep_qini_gain_p_values[2]}, sep_qinis_gain40_p_value:{sep_qini_gain_p_values[3]}, sep_qinis_gain50_p_value:{sep_qini_gain_p_values[4]}')
    print(f'joint_uplifts_gain10_kendall:{joint_uplift_gain_kendalls[0]}, joint_uplifts_gain20_kendall:{joint_uplift_gain_kendalls[1]}, joint_uplifts_gain30_kendall:{joint_uplift_gain_kendalls[2]}, joint_uplifts_gain40_kendall:{joint_uplift_gain_kendalls[3]}, joint_uplifts_gain50_kendall:{joint_uplift_gain_kendalls[4]}')
    print(f'joint_uplifts_gain10_p_value:{joint_uplift_gain_p_values[0]}, joint_uplifts_gain20_p_value:{joint_uplift_gain_p_values[1]}, joint_uplifts_gain30_p_value:{joint_uplift_gain_p_values[2]}, joint_uplifts_gain40_p_value:{joint_uplift_gain_p_values[3]}, joint_uplifts_gain50_p_value:{joint_uplift_gain_p_values[4]}')
    print(f'joint_qinis_gain10_kendall:{joint_qini_gain_kendalls[0]}, joint_qinis_gain20_kendall:{joint_qini_gain_kendalls[1]}, joint_qinis_gain30_kendall:{joint_qini_gain_kendalls[2]}, joint_qinis_gain40_kendall:{joint_qini_gain_kendalls[3]}, joint_qinis_gain50_kendall:{joint_qini_gain_kendalls[4]}')
    print(f'joint_qinis_gain10_p_value:{joint_qini_gain_p_values[0]}, joint_qinis_gain20_p_value:{joint_qini_gain_p_values[1]}, joint_qinis_gain30_p_value:{joint_qini_gain_p_values[2]}, joint_qinis_gain40_p_value:{joint_qini_gain_p_values[3]}, joint_qinis_gain50_p_value:{joint_qini_gain_p_values[4]}')
    print(f'pu_scores_gain10_kendall:{pu_scores_gain_kendalls[0]}, pu_scores_gain20_kendall:{pu_scores_gain_kendalls[1]}, pu_scores_gain30_kendall:{pu_scores_gain_kendalls[2]}, pu_scores_gain40_kendall:{pu_scores_gain_kendalls[3]}, pu_scores_gain50_kendall:{pu_scores_gain_kendalls[4]}')
    print(f'pu_scores_gain10_p_value:{pu_scores_gain_p_values[0]}, pu_scores_gain20_p_value:{pu_scores_gain_p_values[1]}, pu_scores_gain30_p_value:{pu_scores_gain_p_values[2]}, pu_scores_gain40_p_value:{pu_scores_gain_p_values[3]}, pu_scores_gain50_p_value:{pu_scores_gain_p_values[4]}')
    # print(f'sep_uplifts_gain10_kendall:{sep_uplifts_gain10_kendall}, sep_uplifts_gain20_kendall:{sep_uplifts_gain20_kendall}, sep_uplifts_gain30_kendall:{sep_uplifts_gain30_kendall}, sep_uplifts_gain40_kendall:{sep_uplifts_gain40_kendall}')
    # print(f'sep_qinis_gain10_kendall:{sep_qinis_gain10_kendall}, sep_qinis_gain20_kendall:{sep_qinis_gain20_kendall}, sep_qinis_gain30_kendall:{sep_qinis_gain30_kendall}, sep_qinis_gain40_kendall:{sep_qinis_gain40_kendall}')
    # print(f'joint_uplifts_gain10_kendall:{joint_uplifts_gain10_kendall}, joint_uplifts_gain20_kendall:{joint_uplifts_gain20_kendall}, joint_uplifts_gain30_kendall:{joint_uplifts_gain30_kendall}, joint_uplifts_gain40_kendall:{joint_uplifts_gain40_kendall}')
    # print(f'joint_qinis_gain10_kendall:{joint_qinis_gain10_kendall}, joint_qinis_gain20_kendall:{joint_qinis_gain20_kendall}, joint_qinis_gain30_kendall:{joint_qinis_gain30_kendall}, joint_qinis_gain40_kendall:{joint_qinis_gain40_kendall}')
    # print(f'pu_scores_gain10_kendall:{pu_scores_gain10_kendall}, pu_scores_gain20_kendall:{pu_scores_gain20_kendall}, pu_scores_gain30_kendall:{pu_scores_gain30_kendall}, pu_scores_gain40_kendall:{pu_scores_gain40_kendall}')


    # 画各个指标训练100次的折线图
    plt.plot(range(args.epochs), test_sep_uplifts, label='sep_uplifts')
    plt.savefig(os.path.join(save_path, 'sep_uplifts.png'))
    plt.plot(range(args.epochs), test_sep_qinis, label='sep_qinis')
    plt.savefig(os.path.join(save_path, 'sep_qinis.png'))
    plt.plot(range(args.epochs), test_joint_uplifts, label='joint_uplifts')
    plt.savefig(os.path.join(save_path, 'joint_uplifts.png'))
    plt.plot(range(args.epochs), test_joint_qinis, label='joint_qinis')
    plt.savefig(os.path.join(save_path, 'joint_qinis.png'))
    plt.plot(range(args.epochs), test_pu_scores, label='pu_scores')
    plt.savefig(os.path.join(save_path, 'pu_scores.png'))
    # plt.plot(range(args.epochs), model.pehes, label='pehes')
    # plt.savefig(os.path.join(save_path, 'pehes.png'))
    plt.plot(range(args.epochs), test_pred_gains_auc[0]/(test_data.shape[0]*0.1), label='gain10')
    plt.savefig(os.path.join(save_path, 'gain10.png'))
    plt.plot(range(args.epochs), test_pred_gains_auc[1]/(test_data.shape[0]*0.2), label='gain20')
    plt.savefig(os.path.join(save_path, 'gain20.png'))
    plt.plot(range(args.epochs), test_pred_gains_auc[2]/(test_data.shape[0]*0.3), label='gain30')
    plt.savefig(os.path.join(save_path, 'gain30.png'))
    plt.plot(range(args.epochs), test_pred_gains_auc[3]/(test_data.shape[0]*0.4), label='gain40')
    plt.savefig(os.path.join(save_path, 'gain40.png'))
    plt.plot(range(args.epochs), test_pred_gains_auc[4]/(test_data.shape[0]*0.5), label='gain50')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'gain50.png'))

    print(f'pred_relative_uplift:{pred_relative_uplift}, true_relative_uplift:{true_relative_uplift}')
    print(f'pred_sep_qini:{pred_sep_qini}, true_sep_qini:{true_sep_qini}')
    print(f' true_joint_uplift:{true_tau_uplift} true_pr_uplift:{true_pr_uplift} true_sdr_uplift:{true_sdr_uplift}')
    print(f' true_joint_qini:{true_tau_qini} true_pr_qini:{true_pr_qini} true_sdr_qini:{true_sdr_qini}')
    print(f' joint uplift_score:{uplift_score}, joint qini_score:{qini_score}, pr_qini:{pr_qini}, sdr_qini:{sdr_qini},pr_sdr_qini:{pr_sdr_qini}, pr_uplift:{pr_uplift}, sdr_uplift:{sdr_uplift}, pr_sdr_uplift:{pr_sdr_uplift}')
    print(f'pred_pu_score:{pred_pu_score}, true_pu_score:{true_pu_score}')
    print(f'pehe:{pehe} acc:{acc} persuadable_acc:{persuadable_acc} sleepingdog_acc:{sleepingdog_acc}')
    print(f'pred_cate_kendall:{pred_cate_kendall} true_cate_kendall:{true_cate_kendall}')
    print(f'true_gain_top10:{true_gains[0]}, true_gain_top20:{true_gains[1]}, true_gain_top30:{true_gains[2]}, true_gain_top40:{true_gains[3]}, true_gain_top50:{true_gains[4]}')
    print(f'pred_gain_top10:{pred_gains[0]}, pred_gain_top20:{pred_gains[1]}, pred_gain_top30:{pred_gains[2]}, pred_gain_top40:{pred_gains[3]}, pred_gain_top50:{pred_gains[4]}')
    # print(f'pred_pr_kendall:{pred_pr_kendall} true_pr_kendall:{true_pr_kendall}')
    # print(f'pred_sdr_kendall:{pred_sdr_kendall} true_sdr_kendall:{true_sdr_kendall}')
    # print(f'pred_principle_kendall:{pred_principle_rank_kendall} true_principle_kendall:{true_principle_rank_kendall}')
    # print(f'pred_principle_uplift:{pred_principle_rank_uplift} true_principle_uplift:{true_principle_rank_uplift}')
    # print(f'pred_principle_qini:{pred_principle_rank_qini} true_principle_qini:{true_principle_rank_qini}')
    log(save_path, 'joint qini_score:'+str(qini_score)+' pr_qini:'+str(pr_qini)+' sdr_qini:'+str(sdr_qini)\
        +' joint uplift_score:'+str(uplift_score)+' pr_uplift:'+str(pr_uplift)+' sdr_uplift:'+str(sdr_uplift)\
        +' true_joint_uplift:'+str(true_tau_uplift)+' true_pr_uplift:'+str(true_pr_uplift)+' true_sdr_uplift:'+str(true_sdr_uplift)\
        +' true_joint_qini:'+str(true_tau_qini)+' true_pr_qini'+str(true_pr_qini)+' true_sdr_qini:'+str(true_sdr_qini)\
        +' pehe'+str(pehe)+' acc'+str(acc)+' persuadable_acc'+str(persuadable_acc)+' sleepingdog_acc'+str(sleepingdog_acc)\
        # +' pred_cate_kendall'+str(pred_cate_kendall)+' true_cate_kendall'+str(true_cate_kendall)\
        # +' pred_pr_kendall'+str(pred_pr_kendall)+' true_pr_kendall'+str(true_pr_kendall)\
        # +' pred_sdr_kendall'+str(pred_sdr_kendall)+' true_sdr_kendall'+str(true_sdr_kendall)\
        # +' pred_principle_kendall'+str(pred_principle_rank_kendall)+' true_principle_kendall'+str(true_principle_rank_kendall)\
        # +' pred_principle_qini'+str(pred_principle_rank_qini)+' true_principle_qini'+str(true_principle_rank_qini)\
        # +' pred_principle_uplift'+str(pred_principle_rank_uplift)+' true_principle_uplift'+str(true_principle_rank_uplift)\
        +' pred_pu_score:'+str(pred_pu_score)+' true_pu_score:'+str(true_pu_score)\
        +' pred_relative_uplift:'+str(pred_relative_uplift)+' true_relative_uplift:'+str(true_relative_uplift)\
        +' pred_sep_qini:'+str(pred_sep_qini)+' true_sep_qini:'+str(true_sep_qini)\
        +' true_gain_top10:'+str(true_gains[0])+' true_gain_top20:'+str(true_gains[1])+' true_gain_top30:'+str(true_gains[2])+' true_gain_top40:'+str(true_gains[3])+' true_gain_top50:'+str(true_gains[4])\
        +' pred_gain_top10:'+str(pred_gains[0])+' pred_gain_top20:'+str(pred_gains[1])+' pred_gain_top30:'+str(pred_gains[2])+' pred_gain_top40:'+str(pred_gains[3])+' pred_gain_top50:'+str(pred_gains[4])\
        +' sep_uplifts_gain10_kendall:'+str(sep_uplift_gain_kendalls[0])+' sep_uplifts_gain20_kendall'+str(sep_uplift_gain_kendalls[1])\
        +' sep_uplifts_gain30_kendall:'+str(sep_uplift_gain_kendalls[2])+' sep_uplifts_gain40_kendall:'+str(sep_uplift_gain_kendalls[3])+' sep_uplifts_gain50_kendall:'+str(sep_uplift_gain_kendalls[4])\
        +' sep_qinis_gain10_kendall:'+str(sep_qini_gain_kendalls[0])+' sep_qinis_gain20_kendall:'+str(sep_qini_gain_kendalls[1])\
        +' sep_qinis_gain30_kendall:'+str(sep_qini_gain_kendalls[2])+' sep_qinis_gain40_kendall:'+str(sep_qini_gain_kendalls[3])+' sep_qinis_gain50_kendall:'+str(sep_qini_gain_kendalls[4])\
        +' joint_uplifts_gain10_kendall:'+str(joint_uplift_gain_kendalls[0])+' joint_uplifts_gain20_kendall:'+str(joint_uplift_gain_kendalls[1])\
        +' joint_uplifts_gain30_kendall:'+str(joint_uplift_gain_kendalls[2])+' joint_uplifts_gain40_kendall:'+str(joint_uplift_gain_kendalls[3])+' joint_uplifts_gain50_kendall:'+str(joint_uplift_gain_kendalls[4])\
        +' joint_qinis_gain10_kendall:'+str(joint_qini_gain_kendalls[0])+' joint_qinis_gain20_kendall:'+str(joint_qini_gain_kendalls[1])\
        +' joint_qinis_gain30_kendall:'+str(joint_qini_gain_kendalls[2])+' joint_qinis_gain40_kendall:'+str(joint_qini_gain_kendalls[3])+' joint_qinis_gain50_kendall:'+str(joint_qini_gain_kendalls[4])\
        +' pu_scores_gain10_kendall:'+str(pu_scores_gain_kendalls[0])+' pu_scores_gain20_kendall:'+str(pu_scores_gain_kendalls[1])\
        +' pu_scores_gain30_kendall:'+str(pu_scores_gain_kendalls[2])+' pu_scores_gain40_kendall:'+str(pu_scores_gain_kendalls[3])+' pu_scores_gain50_kendall:'+str(pu_scores_gain_kendalls[4])
        +'\n',
        args.model_name+'.txt')
    log(save_path, '\n', args.model_name+'.txt')