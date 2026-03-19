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
# from models.TONet_qini import TONet_qini
from models.TONet_pu import TONet_pu
import os
from argparse import ArgumentParser
import torch
# from scipy.stats import kendalltau
from models.model_utils import test_model
import time

import subprocess
import sys
import os
import signal

# 子进程列表
subprocesses = []

def close_subprocesses(signum, frame):
    print("Received interrupt signal. Closing subprocesses...")
    for proc in subprocesses:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

    sys.exit(0)
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_exp', type=int, default=0)
    # parser.add_argument('--k_folds', type=int, default=0)
    # [256, 512, 1024, 2048]
    parser.add_argument('--train_bs', type=int, default=2048)
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
    parser.add_argument('--alpha', type=float, default=0.1)
    # [0.1, 0.5, 1, 5, 10]
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--ipm_type', type=str, default='mmd_lin')
    args = parser.parse_args()

    exp = 71

    signal.signal(signal.SIGINT, close_subprocesses)

    start_time = time.time()


    train_bs_set = [512, 1024, 2048]
    lr_set = [1e-2,1e-3,1e-4]
    h_dim_set = [16,32,64]
    num_layers_set = [5,6,7]
    # alpha_set = [0.1, 0.5, 1, 5, 10]
    # beta_set = [0.1, 0.5, 1, 5, 10]
    # gamma_set = [0.1, 0.5, 1, 5, 10]
    # gamma_set = [2.0]
    for train_bs in train_bs_set:
        for lr in lr_set:
            for h_dim in h_dim_set:
                for num_layers in num_layers_set:
                    # for alpha in alpha_set:
            # for beta in beta_set:
                    #     for gamma in gamma_set:
                    exp = exp + 1

                    command = f'taskset -c {exp} python main_synthetic_ptonet.py --train_bs={train_bs} --lr={lr} --num_layers={num_layers}' 
                    # command = f'taskset -c {exp} python main_criteo_ptonet.py --alpha={alpha} --beta={beta} --num_layers={num_layers}' 
                    # command = f'taskset -c {exp} python main_lzd_ptonet.py --train_bs={train_bs} --lr={lr} --num_layers={num_layers} --h_dim={h_dim}' 

                    proc = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
                    subprocesses.append(proc)

    print(f'exp: {exp}')    
    print('all subprocesses are running...')
    
    for proc in subprocesses:
        proc.wait()  # 等待子进程结束

    print('all subprocesses have finished.')

    sys.exit(0)