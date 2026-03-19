import sys
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import math
from models.earlystop import EarlyStopper
import os
from models.model_utils import test_model


def mmd_lin(Xt, Xc, p):
    mean_treated = torch.mean(Xt)
    mean_control = torch.mean(Xc)
    
    mmd = torch.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control).sum()

    return mmd


def mmd_rbf(Xt, Xc, p, sig=0.1):
    sig = torch.tensor(sig)
    Kcc = torch.exp(-torch.cdist(Xc, Xc, 2.0001) / torch.sqrt(sig))
    Kct = torch.exp(-torch.cdist(Xc, Xt, 2.0001) / torch.sqrt(sig))
    Ktt = torch.exp(-torch.cdist(Xt, Xt, 2.0001) / torch.sqrt(sig))

    m = Xc.shape[0]
    n = Xt.shape[0]

    mmd = (1 - p) ** 2 / (m *(m-1)) * (Kcc.sum() - m)
    mmd += p ** 2 / (n * (n-1)) * (Ktt.sum() - n)
    mmd -= 2 * p * (1 - p) / (m * n) * Kct.sum()
    mmd *= 4

    return mmd


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        num_layers,
        hidden_dim,
        out_dim,
        activation=nn.ELU(inplace=True),
        dropout=0.2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dropout = dropout

        nonlin = True
        if self.activation is None:
            nonlin = False

        layers = []
        for i in range(num_layers - 1):
            layers.extend(
                self._layer(
                    hidden_dim if i > 0 else in_dim,
                    hidden_dim,
                    nonlin,
                )
            )
        layers.extend(self._layer(hidden_dim, out_dim, False))

        self.regression = nn.Sequential(*layers)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def _layer(self, in_dim, out_dim, activation=True):
        if activation:
            return [
                nn.Linear(in_dim, out_dim),
                self.activation,
                nn.Dropout(self.dropout),
            ]
        else:
            return [
                nn.Linear(in_dim, out_dim),
            ]

    def forward(self, x):
        out = self.regression(x)
        return  out

 

def get_score(model, x_test, y_test, t_test):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    N = len(x_test)

    # MSE
    _ypred = model.forward(x_test, t_test)
    mse = mean_squared_error(y_test, _ypred)

    # treatment index
    t_idx = np.where(t_test.to("cpu").detach().numpy().copy() == 1)[0]
    c_idx = np.where(t_test.to("cpu").detach().numpy().copy() == 0)[0]

    # ATE & ATT
    _t0 = torch.FloatTensor([0 for _ in range(N)]).reshape([-1, 1])
    _t1 = torch.FloatTensor([1 for _ in range(N)]).reshape([-1, 1])

    # _cate = model.forward(x_test, _t1) - model.forward(x_test, _t0)
    _cate_t = y_test - model.forward(x_test, _t0)
    _cate_c = model.forward(x_test, _t1) - y_test
    _cate = torch.cat([_cate_c[c_idx], _cate_t[t_idx]])
    
    _ate = np.mean(_cate.to("cpu").detach().numpy().copy())
    _att = np.mean(_cate_t[t_idx].to("cpu").detach().numpy().copy())

    return {"ATE": _ate, "ATT": _att, "RMSE": np.sqrt(mse)}


class Base(nn.Module):
    def __init__(self, args):
        super(Base, self).__init__()
        self.args = args
        self.criterion = nn.MSELoss(reduction='none')
        self.mse = mean_squared_error

    def fit(
        self,
        dataloader,
        x_train,
        y_train,
        t_train,
        x_test,
        y_test,
        t_test,
        logger,
    ):
        losses = []
        ipm_result = []
        logger.debug("within sample, out of sample")
        logger.debug("[Train MSE, IPM], [RMSE, ATT, ATE], [RMSE, ATT, ATE]")
        for epoch in range(self.args.epochs):
            epoch_loss = 0
            epoch_ipm = []
            n = 0
            for (x, y, z) in dataloader:

                x = x.to(device=torch.device("cpu"))
                y = y.to(device=torch.device("cpu"))
                z = z.to(device=torch.device("cpu"))
                self.optimizer.zero_grad()

                x_rep = self.repnet(x)

                _t_id = np.where((z.cpu().detach().numpy() == 1).all(axis=1))[0]
                _c_id = np.where((z.cpu().detach().numpy() == 0).all(axis=1))[0]

                y_hat_treated = self.outnet_treated(x_rep[_t_id])
                y_hat_control = self.outnet_control(x_rep[_c_id])

                _index = np.argsort(np.concatenate([_t_id, _c_id], 0))

                y_hat = torch.cat([y_hat_treated, y_hat_control])[_index]


                loss = self.criterion(y_hat, y.reshape([-1, 1]))
                # sample weight
                p_t = np.mean(z.cpu().detach().numpy())
                w_t = z/(2*p_t)
                # 这里写错了吧，我给改了
                w_c = (1-z)/(2*(1-p_t))
                sample_weight = w_t + w_c
                if (p_t == 1) or (p_t ==0):
                    sample_weight = 1
                
                loss =torch.mean((loss * sample_weight))

                if self.args.alpha > 0.0:    
                    if self.args.ipm_type == "mmd_rbf":
                       ipm = mmd_rbf(
                            x_rep[_t_id],
                            x_rep[_c_id],
                            p=len(_t_id) / (len(_t_id) + len(_c_id)),
                        )
                    elif self.args.ipm_type == "mmd_lin":
                        ipm = mmd_lin(
                            x_rep[_t_id],
                            x_rep[_c_id],
                            p=len(_t_id) / (len(_t_id) + len(_c_id))
                        )
                    else:
                        logger.debug(f'{self.args.ipm_type} : TODO!!! Not implemented yet!')
                        sys.exit()

                    loss += ipm * self.args.alpha
                    epoch_ipm.append(ipm.cpu().detach().numpy())

                
                mse = self.mse(
                    y_hat.detach().cpu().numpy(),
                    y.reshape([-1, 1]).detach().cpu().numpy(),
                )
                
                loss.backward()

                self.optimizer.step()
                epoch_loss += mse * y.shape[0]
                n += y.shape[0]

            self.scheduler.step()
            epoch_loss = epoch_loss / n
            losses.append(epoch_loss)
            if self.args.alpha > 0:
                ipm_result.append(np.mean(epoch_ipm))

            if epoch % 100 == 0:
                with torch.no_grad():
                    within_result = get_score(self, x_train, y_train, t_train)
                    outof_result = get_score(self, x_test, y_test, t_test)
                logger.debug(
                    "[Epoch: %d] [%.3f, %.3f], [%.3f, %.3f, %.3f], [%.3f, %.3f, %.3f] "
                    % (
                        epoch,
                        epoch_loss,
                        ipm if self.args.alpha > 0 else -1,
                        within_result["RMSE"],
                        within_result["ATT"],
                        within_result["ATE"],
                        outof_result["RMSE"],
                        outof_result["ATT"],
                        outof_result["ATE"],
                    )
                )

        return within_result, outof_result, losses, ipm_result


class CFR(Base):
    def __init__(self, in_dim, out_dim, args):
        super().__init__(args)
        repnet_layers = args.num_layers-3
        self.repnet = MLP(
            num_layers=repnet_layers,
            in_dim=in_dim,
            hidden_dim=args.h_dim,
            out_dim=args.out_dim,
            activation=nn.ELU(inplace=True),
            dropout=args.dropout,
        )

        self.outnet_treated = MLP(
            in_dim=args.out_dim, out_dim=out_dim, num_layers=3, hidden_dim=args.h_dim, dropout=args.dropout
        )
        self.outnet_control = MLP(
            in_dim=args.out_dim, out_dim=out_dim, num_layers=3, hidden_dim=args.h_dim, dropout=args.dropout
        )

        self.params = (
            list(self.repnet.parameters())
            + list(self.outnet_treated.parameters())
            + list(self.outnet_control.parameters())
        )

        self.optimizer = optim.Adam(
            params=self.params, lr=args.lr, weight_decay=args.wd
        )
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=args.gamma)
    def predict(self, x):
        self.eval()
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        x_rep = self.repnet(x)
        y_hat_treated = self.outnet_treated(x_rep)
        y_hat_control = self.outnet_control(x_rep)
        return y_hat_control.detach(), y_hat_treated.detach()
    def forward(self, x, t):
        x_rep = self.repnet(x)
        # pytorch 获取 t==1 的索引
        _t_id = torch.where(t.squeeze()==1)[0]
        _c_id = torch.where(t.squeeze()==0)[0]

        y_hat_treated = self.outnet_treated(x_rep[_t_id])
        y_hat_control = self.outnet_control(x_rep[_c_id])

        _index = torch.argsort(torch.cat([_t_id, _c_id], 0))
        y_hat = torch.cat([y_hat_treated, y_hat_control])[_index]

        return y_hat, x_rep, _t_id, _c_id
    
    def train_model(self, opt, train_dataloader, valid_data, test_data, args, exp=None, best_model_path=None, logger=None):
        self.train()
        w = 1.0
        test_cate_epochs = np.zeros((args.epochs, test_data.shape[0]))
        early_stopper = EarlyStopper(patience=5, min_delta=0)
        best_val_value = -1.0
        for epoch in range(args.epochs):
            for i, sample in enumerate(train_dataloader):
                sample = sample.to(torch.float32)
                opt.zero_grad()
                x = sample[:, :args.x_dim]
                t = sample[:, args.x_dim]
                y = sample[:, args.x_dim+1]

                out, x_rep, _t_id, _c_id = self(x, t)
                loss = self.criterion(out, y.reshape([-1, 1]))

                # sample weight
                p_t = t.mean()
                w_t = t/(2*p_t)
                # 这里写错了吧，我给改了
                w_c = (1-t)/(2*(1-p_t))
                sample_weight = w_t + w_c
                if (p_t == 1) or (p_t ==0):
                    sample_weight = 1
                
                loss =torch.mean((loss.squeeze() * sample_weight.squeeze()))

                if self.args.alpha > 0.0:    
                    if self.args.ipm_type == "mmd_rbf":
                       ipm = mmd_rbf(
                            x_rep[_t_id],
                            x_rep[_c_id],
                            p=len(_t_id) / (len(_t_id) + len(_c_id)),
                        )
                    elif self.args.ipm_type == "mmd_lin":
                        ipm = mmd_lin(
                            x_rep[_t_id],
                            x_rep[_c_id],
                            p=len(_t_id) / (len(_t_id) + len(_c_id))
                        )
                    else:
                        logger.debug(f'{self.args.ipm_type} : TODO!!! Not implemented yet!')
                        sys.exit()

                    loss += ipm * self.args.alpha
                # opt.zero_grad()
                loss.backward()
                opt.step()

            if (epoch) % 1 == 0:
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
                    out, x_rep, _t_id, _c_id = self(valid_x, valid_t)
                    valid_loss = self.criterion(out,valid_y.reshape([-1,1])).mean()
                    
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


# # Adapted from https://github.com/gpeyre/SinkhornAutoDiff
# class SinkhornDistance(nn.Module):
#     r"""
#     Given two empirical measures each with :math:`P_1` locations
#     :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
#     outputs an approximation of the regularized OT cost for point clouds.
#     Args:
#         eps (float): regularization coefficient
#         max_iter (int): maximum number of Sinkhorn iterations
#         reduction (string, optional): Specifies the reduction to apply to the output:
#             'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
#             'mean': the sum of the output will be divided by the number of
#             elements in the output, 'sum': the output will be summed. Default: 'none'
#     Shape:
#         - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
#         - Output: :math:`(N)` or :math:`()`, depending on `reduction`
#     """

#     def __init__(self, eps, max_iter, reduction='none', device='cpu'):
#         super(SinkhornDistance, self).__init__()
#         self.eps = eps
#         self.max_iter = max_iter
#         self.reduction = reduction
#         self.device = device

#     def forward(self, x, y):
#         # The Sinkhorn algorithm takes as input three variables :
#         C = self._cost_matrix(x, y)  # Wasserstein cost function
#         x_points = x.shape[-2]
#         y_points = y.shape[-2]
#         if x.dim() == 2:
#             batch_size = 1
#         else:
#             batch_size = x.shape[0]

#         # both marginals are fixed with equal weights
#         mu = torch.empty(batch_size, x_points, dtype=torch.float,
#                          requires_grad=False).fill_(1.0 / x_points).squeeze()
#         nu = torch.empty(batch_size, y_points, dtype=torch.float,
#                          requires_grad=False).fill_(1.0 / y_points).squeeze()
#         mu = mu.to(self.device)
#         nu = nu.to(self.device)
#         u = torch.zeros_like(mu)
#         v = torch.zeros_like(nu)
#         # To check if algorithm terminates because of threshold
#         # or max iterations reached
#         actual_nits = 0
#         # Stopping criterion
#         thresh = 1e-1

#         # Sinkhorn iterations
#         for i in range(self.max_iter):
#             u1 = u  # useful to check the update
#             u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
#             v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
#             err = (u - u1).abs().sum(-1).mean()

#             actual_nits += 1
#             if err.item() < thresh:
#                 break

#         U, V = u, v
#         # Transport plan pi = diag(a)*K*diag(b)
#         pi = torch.exp(self.M(C, U, V))
#         # Sinkhorn distance
#         cost = torch.sum(pi * C, dim=(-2, -1))

#         if self.reduction == 'mean':
#             cost = cost.mean()
#         elif self.reduction == 'sum':
#             cost = cost.sum()

#         return cost, pi, C

#     def M(self, C, u, v):
#         "Modified cost for logarithmic updates"
#         "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
#         return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

#     @staticmethod
#     def _cost_matrix(x, y, p=2):
#         "Returns the matrix of $|x_i-y_j|^p$."
#         x_col = x.unsqueeze(-2)
#         y_lin = y.unsqueeze(-3)
#         C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
#         return C

#     @staticmethod
#     def ave(u, u1, tau):
#         "Barycenter subroutine, used by kinetic acceleration through extrapolation."
#         return tau * u + (1 - tau) * u1
