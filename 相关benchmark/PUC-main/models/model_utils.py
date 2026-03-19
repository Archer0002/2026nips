from sklift.metrics import qini_auc_score, uplift_auc_score
from sklift.viz import plot_qini_curve, plot_uplift_curve
from utils import principled_uplift_auc_score, relative_uplift_auc_score, sep_qini_auc_score, get_true_gain, get_true_gain_auc
import torch

def test_model(model, args, test_data, save_path, is_valid=False):
        if is_valid:
            fig_prefix = 'valid'
        else:
            fig_prefix = 'test'
        p_y1_t0, p_y1_t1 = model.predict(test_data[:, :args.x_dim])
        pred_tau = p_y1_t1 - p_y1_t0
        # p_y0_t1 = 1-p_y1_t1
        # p_y0_t0 = 1-p_y1_t0
        # # persuadable risk
        # pr = torch.min(p_y0_t0, p_y1_t1)
        # # sleeping dog risk
        # sdr = torch.min(p_y1_t0, p_y0_t1)
        # # sure thing risk
        # str = torch.min(p_y1_t0, p_y1_t1)
        # # lost cause risk
        # lcr = torch.min(p_y0_t0, p_y0_t1)
        # pr_sdr = pr-sdr

        pred_relative_uplift = relative_uplift_auc_score(test_data[:,args.x_dim+1], pred_tau.squeeze(), test_data[:,args.x_dim])
        pred_sep_qini = sep_qini_auc_score(test_data[:,args.x_dim+1], pred_tau.squeeze(), test_data[:,args.x_dim])
        pred_joint_uplift_score = uplift_auc_score(test_data[:,args.x_dim+1], pred_tau.squeeze(), test_data[:,args.x_dim])
        pred_joint_qini_score = qini_auc_score(test_data[:,args.x_dim+1], pred_tau.squeeze(), test_data[:,args.x_dim])
        pred_pu_score = principled_uplift_auc_score(test_data[:,args.x_dim+1], pred_tau.squeeze(), test_data[:,args.x_dim])

        if args.data == 'criteo' or args.data == 'lzd':
            return pred_tau, pred_relative_uplift, pred_sep_qini, pred_joint_uplift_score, pred_joint_qini_score, pred_pu_score
        elif args.data.startswith('synthetic'):
            true_tau = test_data[:,args.x_dim+2]
            true_rank_score = true_tau

            true_relative_uplift = relative_uplift_auc_score(test_data[:,args.x_dim+1], true_rank_score.squeeze(), test_data[:,args.x_dim])
            true_sep_qini = sep_qini_auc_score(test_data[:,args.x_dim+1], true_rank_score.squeeze(), test_data[:,args.x_dim])
            true_joint_uplift = uplift_auc_score(test_data[:,args.x_dim+1], true_rank_score.squeeze(), test_data[:,args.x_dim])
            true_joint_qini = qini_auc_score(test_data[:,args.x_dim+1], true_rank_score.squeeze(), test_data[:,args.x_dim])
            true_pu_score = principled_uplift_auc_score(test_data[:,args.x_dim+1], true_rank_score.squeeze(), test_data[:,args.x_dim])
            
            true_gains = []
            for i in range(1,6):
                true_gain = get_true_gain_auc(test_data, true_rank_score.squeeze(), 0.2*i)
                # true_gain = get_true_gain(test_data, true_rank_score.squeeze(), 0.1*i)
                true_gains.append(true_gain)

            pred_gains = []
            for i in range(1,6):
                pred_gain = get_true_gain_auc(test_data, pred_tau.squeeze(), 0.2*i)
                pred_gains.append(pred_gain)

            pehe = torch.sqrt(torch.mean((pred_tau.squeeze()-true_rank_score.squeeze())**2))

            return pred_tau, pehe, pred_relative_uplift, pred_sep_qini, pred_joint_uplift_score, pred_joint_qini_score, pred_pu_score,\
            true_relative_uplift, true_sep_qini, true_joint_uplift, true_joint_qini, true_pu_score, true_gains, pred_gains
