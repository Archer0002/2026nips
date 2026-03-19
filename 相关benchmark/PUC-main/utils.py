import os
import torch
import numpy as np
import random
import pandas as pd
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.extmath import stable_cumsum
from sklearn.metrics import auc
from sklearn.utils import check_matplotlib_support
import matplotlib.pyplot as plt

def log(save_path, txt, file_name='file.txt'):
    with open(os.path.join(save_path, file_name), 'a+') as f:
        f.write(txt)
def setup_seed(seed):
     torch.manual_seed(seed)
     if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

# 定义函数 reform_eval_df，用于整理评估数据的 DataFrame
def reform_eval_df(label, pred_score, treatment, label_col='label'):
    # 创建一个新的 DataFrame，指定列名
    new_df = pd.DataFrame(columns=['treatment', 'pred_score', label_col])
    
    # 将输入数组的值分配给相应的列
    new_df['treatment'] = treatment  # .squeeze(1)
    new_df['pred_score'] = pred_score  # .squeeze(1)
    new_df[label_col] = label  # .squeeze(1)
    # 按照 'pred_score' 列对 DataFrame 进行排序

    # 如果 'user_id' 不在列名中，添加 'user_id' 列并赋予递增的数值
    if 'user_id' not in new_df.columns:
        new_df['user_id'] = np.arange(len(new_df))

    # 创建一个新的 DataFrame 'df_eval'，选择特定的列并重命名它们
    df_eval = new_df[['user_id', 'treatment', label_col, 'pred_score']]
    df_eval.columns = ['user_id', 'treatment', 'label', 'pred_score']
    df_eval['treatment'] = df_eval['treatment'].astype('int')
    return df_eval

# 定义函数 get_eval_score，用于计算评估分数
def get_eval_score(df_eval, num_bucket):
    # 创建一个新列 'bin'，通过将 'pred_score' 列分桶成指定数量的区间
    df_eval['bin'] = pd.qcut(df_eval['pred_score'], num_bucket, duplicates='drop', labels=False)
    
    # 先按照score进行分组，然后按照treatment进行分组，计算组内label的均值
    res = df_eval.groupby(['bin', 'treatment'])['label'].mean()
    res = res.reset_index()

    # 分别提取两种处理组合的结果
    t0 = res[res.treatment.isin([0, '0'])]
    t1 = res[res.treatment.isin([1, '1'])]

    # 将两种处理组合的结果合并在 'bin' 列上
    comp = pd.merge(t0, t1, on=['bin'])

    # 计算 uplift，即处理组合 1 的 'label' 均值减去处理组合 0 的 'label' 均值
    comp['uplift'] = comp['label_y'] - comp['label_x']

    # 计算两侧两个的桶的 uplift 差异
    cate_diff = comp[comp.bin == num_bucket - 1]['uplift'].values - comp[comp.bin == 0]['uplift'].values

    # 计算 'score' 和 'uplift' 之间的 Kendall 相关性
    kendall = comp[['bin', 'uplift']].corr(method='kendall').values[0][1]

    return kendall, comp['bin'], comp['uplift']

def kendalltau(y, pred_tau, treatment, label_col='label', num_bucket=20):
    df_eval = reform_eval_df(y, pred_tau, treatment, label_col=label_col)
    return get_eval_score(df_eval, num_bucket)

def check_is_binary(array):
    """Checker if array consists of int or float binary values 0 (0.) and 1 (1.)

    Args:
        array (1d array-like): Array to check.
    """

    if not np.all(np.unique(array) == np.array([0, 1])):
        raise ValueError(f"Input array is not binary. "
                         f"Array should contain only int or float binary values 0 (or 0.) and 1 (or 1.). "
                         f"Got values {np.unique(array)}.")

def principled_uplift_curve(y_true, uplift, treatment):
    """Compute Qini curve.

    For computing the area under the Qini Curve, see :func:`.qini_auc_score`.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.

    Returns:
        array (shape = [>2]), array (shape = [>2]): Points on a curve.

    See also:
        :func:`.uplift_curve`: Compute the area under the Qini curve.

        :func:`.perfect_qini_curve`: Compute the perfect Qini curve.

        :func:`.plot_qini_curves`: Plot Qini curves from predictions..

        :func:`.uplift_curve`: Compute Uplift curve.

    References:
        Nicholas J Radcliffe. (2007). Using control groups to target on predicted lift:
        Building and assessing uplift model. Direct Marketing Analytics Journal, (3):14–21, 2007.

        Devriendt, F., Guns, T., & Verbeke, W. (2020). Learning to rank for uplift modeling. ArXiv, abs/2002.05897.
    """

    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    # [::-1] 是反转序列的切片操作。这里是将 uplift 按照从大到小的顺序排列
    desc_score_indices = np.argsort(uplift, kind="mergesort", axis=0)[::-1]

    y_true = y_true[desc_score_indices]
    treatment = treatment[desc_score_indices]
    uplift = uplift[desc_score_indices]

    n_y1_t1, n_y0_t0, n_y1_t0, n_y0_t1 = np.zeros_like(y_true), np.zeros_like(y_true), np.zeros_like(y_true), np.zeros_like(y_true)
    # 获取 treatment 为 1 且 y_true 为 1 的索引
    n_y1_t1[np.where((y_true == 1) & (treatment == 1))] = 1
    n_y0_t0[np.where((y_true == 0) & (treatment == 0))] = 1
    n_y1_t0[np.where((y_true == 1) & (treatment == 0))] = 1
    n_y0_t1[np.where((y_true == 0) & (treatment == 1))] = 1

    cum_y1_t1 = stable_cumsum(n_y1_t1)
    cum_y0_t0 = stable_cumsum(n_y0_t0)
    cum_y1_t0 = stable_cumsum(n_y1_t0)
    cum_y0_t1 = stable_cumsum(n_y0_t1)
    cum_all = stable_cumsum(np.ones_like(treatment))
    cum_t = stable_cumsum(treatment)
    cum_c = stable_cumsum(cum_all - treatment)
    sum_t = np.sum(treatment)
    sum_c = treatment.shape[0]- sum_t
    k = np.array(range(1, len(y_true) + 1))
    
    # 1
    # curve_values = np.minimum(np.divide(cum_y1_t1, cum_t, out=np.zeros_like(cum_y1_t1), where=cum_t != 0),
    #     np.divide(cum_y0_t0, cum_c, out=np.zeros_like(cum_y0_t0), where=cum_c != 0)) -\
    # np.minimum(np.divide(cum_y1_t0, cum_c, out=np.zeros_like(cum_y1_t0), where=cum_c != 0), 
    #     np.divide(cum_y0_t1, cum_t, out=np.zeros_like(cum_y0_t1), where=cum_t != 0))

    # 2
    # curve_values = np.minimum(np.divide(cum_y1_t1, sum_t, out=np.zeros_like(cum_y1_t1), where=sum_t != 0),
    #     np.divide(cum_y0_t0, sum_c, out=np.zeros_like(cum_y0_t0), where=sum_c != 0)) -\
    # np.minimum(np.divide(cum_y1_t0, sum_c, out=np.zeros_like(cum_y1_t0), where=sum_c != 0), 
    #     np.divide(cum_y0_t1, sum_t, out=np.zeros_like(cum_y0_t1), where=sum_t != 0))

    # 3
    # curve_values = np.minimum(np.divide(cum_y1_t1, sum_t, out=np.zeros_like(cum_y1_t1), where=sum_t != 0),
    #     np.divide(cum_y0_t0, sum_c, out=np.zeros_like(cum_y0_t0), where=sum_c != 0)) * (cum_y1_t1+ cum_y0_t0) -\
    # np.minimum(np.divide(cum_y1_t0, sum_c, out=np.zeros_like(cum_y1_t0), where=sum_c != 0), 
    #     np.divide(cum_y0_t1, sum_t, out=np.zeros_like(cum_y0_t1), where=sum_t != 0))* (cum_y0_t1+ cum_y1_t0)
    
    # 4
    # curve_values = np.minimum(np.divide(cum_y1_t1, sum_t*k, out=np.zeros_like(cum_y1_t1), where=sum_t != 0),
    #     np.divide(cum_y0_t0, sum_c*k, out=np.zeros_like(cum_y0_t0), where=sum_c != 0)) -\
    # np.minimum(np.divide(cum_y1_t0, sum_c*k, out=np.zeros_like(cum_y1_t0), where=sum_c != 0), 
    #     np.divide(cum_y0_t1, sum_t*k, out=np.zeros_like(cum_y0_t1), where=sum_t != 0))
    
    # 5
    # curve_values = np.maximum(np.divide(cum_y1_t1, sum_t, out=np.zeros_like(cum_y1_t1), where=sum_t != 0),
    #     np.divide(cum_y0_t0, sum_c, out=np.zeros_like(cum_y0_t0), where=sum_c != 0)) -\
    # np.minimum(np.divide(cum_y1_t0, sum_c, out=np.zeros_like(cum_y1_t0), where=sum_c != 0), 
    #     np.divide(cum_y0_t1, sum_t, out=np.zeros_like(cum_y0_t1), where=sum_t != 0))

    # 6
    # curve_values = np.maximum(np.divide(cum_y1_t1, sum_t, out=np.zeros_like(cum_y1_t1), where=sum_t != 0),
    #     np.divide(cum_y0_t0, sum_c, out=np.zeros_like(cum_y0_t0), where=sum_c != 0)) -\
    # np.maximum(np.divide(cum_y1_t0, sum_c, out=np.zeros_like(cum_y1_t0), where=sum_c != 0), 
    #     np.divide(cum_y0_t1, sum_t, out=np.zeros_like(cum_y0_t1), where=sum_t != 0))
    
    # 7
    # curve_values = np.minimum(np.divide(cum_y1_t1, sum_t, out=np.zeros_like(cum_y1_t1), where=sum_t != 0),
    #     np.divide(cum_y0_t0, sum_c, out=np.zeros_like(cum_y0_t0), where=sum_c != 0)) -\
    # np.maximum(np.divide(cum_y1_t0, sum_c, out=np.zeros_like(cum_y1_t0), where=sum_c != 0), 
    #     np.divide(cum_y0_t1, sum_t, out=np.zeros_like(cum_y0_t1), where=sum_t != 0))

    # 8
    # curve_values = np.divide(cum_y1_t1, sum_t, out=np.zeros_like(cum_y1_t1), where=sum_t != 0)+\
    #     np.divide(cum_y0_t0, sum_c, out=np.zeros_like(cum_y0_t0), where=sum_c != 0) -\
    #     np.divide(cum_y1_t0, sum_c, out=np.zeros_like(cum_y1_t0), where=sum_c != 0) -\
    #     np.divide(cum_y0_t1, sum_t, out=np.zeros_like(cum_y0_t1), where=sum_t != 0)
    # 9
    # curve_values = np.divide(cum_y1_t1, cum_t, out=np.zeros_like(cum_y1_t1), where=cum_t != 0)+\
    #     np.divide(cum_y0_t0, cum_c, out=np.zeros_like(cum_y0_t0), where=cum_c != 0) -\
    #     np.divide(cum_y1_t0, cum_c, out=np.zeros_like(cum_y1_t0), where=cum_c != 0) -\
    #     np.divide(cum_y0_t1, cum_t, out=np.zeros_like(cum_y0_t1), where=cum_t != 0)
    # 10
    curve_values = cum_y1_t1 + cum_y0_t0 - cum_y1_t0 - cum_y0_t1   
    # 11
    # curve_values = np.minimum(cum_y1_t1, cum_y0_t0) - np.minimum(cum_y1_t0, cum_y0_t1)
    # 12
    # curve_values = np.divide(cum_y1_t1, sum_t, out=np.zeros_like(cum_y1_t1), where=sum_t != 0)+\
    # np.divide(cum_y0_t0, sum_c, out=np.zeros_like(cum_y0_t0), where=sum_c != 0)
    
    # y_true_ctrl, y_true_trmnt = y_true.copy(), y_true.copy()

    # y_true_ctrl[treatment == 1] = 0
    # y_true_trmnt[treatment == 0] = 0

    # distinct_value_indices = np.where(np.diff(uplift))[0]
    # threshold_indices = np.r_[distinct_value_indices, uplift.size - 1]

    # 删去相同值后的treatment组总数
    # num_trmnt = stable_cumsum(treatment)[threshold_indices]
    # y_trmnt = stable_cumsum(y_true_trmnt)[threshold_indices]

    # num_all = threshold_indices + 1
    # # 删去相同值后，control组的总数
    # num_ctrl = num_all - num_trmnt
    # y_ctrl = stable_cumsum(y_true_ctrl)[threshold_indices]

    # curve_values = y_trmnt - y_ctrl * np.divide(num_trmnt, num_ctrl, out=np.zeros_like(num_trmnt), where=num_ctrl != 0)


    return curve_values


def max_principled_uplift_curve(y_true, treatment):
    """Compute the maximum principled Qini curve, not the optimal.
       The optimal is identifiable from observed data.

    For computing the area under the Qini Curve, see :func:`.qini_auc_score`.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        treatment (1d array-like): Treatment labels.
        negative_effect (bool): If True, optimum Qini Curve contains the negative effects
            (negative uplift because of campaign). Otherwise, optimum Qini Curve will not
            contain the negative effects.
    Returns:
        array (shape = [>2]), array (shape = [>2]): Points on a curve.

    See also:
        :func:`.qini_curve`: Compute Qini curve.

        :func:`.qini_auc_score`: Compute the area under the Qini curve.

        :func:`.plot_qini_curves`: Plot Qini curves from predictions..
    """

    check_consistent_length(y_true, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)

    y_true, treatment = np.array(y_true), np.array(treatment)

    perfect_uplift = np.zeros_like(y_true)
    perfect_uplift[treatment == y_true] = 1
    # perfect_uplift[treatment != y_true] = -1
    # express an ideal uplift curve through y_true and treatment

    y_perfect = principled_uplift_curve(y_true, perfect_uplift, treatment)

    return y_perfect


def principled_uplift_auc_score(y_true, uplift, treatment):
    """Compute normalized Area Under the Qini curve (aka Qini coefficient) from prediction scores.

    By computing the area under the Qini curve, the curve information is summarized in one number.
    For binary outcomes the ratio of the actual uplift gains curve above the diagonal to that of
    the optimum Qini curve.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        negative_effect (bool): If True, optimum Qini Curve contains the negative effects
            (negative uplift because of campaign). Otherwise, optimum Qini Curve will not contain the negative effects.

            .. versionadded:: 0.2.0

    Returns:
        float: Qini coefficient.

    See also:
        :func:`.qini_curve`: Compute Qini curve.

        :func:`.perfect_qini_curve`: Compute the perfect (optimum) Qini curve.

        :func:`.plot_qini_curves`: Plot Qini curves from predictions..

        :func:`.uplift_auc_score`: Compute normalized Area Under the Uplift curve from prediction scores.

    References:
        Nicholas J Radcliffe. (2007). Using control groups to target on predicted lift:
        Building and assessing uplift model. Direct Marketing Analytics Journal, (3):14–21, 2007.
    """

    # TODO: Add Continuous Outcomes
    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    # if not isinstance(negative_effect, bool):
    #     raise TypeError(f'Negative_effects flag should be bool, got: {type(negative_effect)}')
    x_actual = np.array(range(len(y_true)))
    y_actual = principled_uplift_curve(y_true, uplift, treatment)
    y_perfect = max_principled_uplift_curve(y_true, treatment)
    x_baseline, y_baseline = np.array([0, x_actual[-1]]), np.array([0, y_perfect[-1]])

    auc_score_baseline = auc(x_baseline, y_baseline)
    auc_score_perfect = auc(x_actual, y_perfect) - auc_score_baseline
    auc_score_actual = auc(x_actual, y_actual) - auc_score_baseline

    # return auc_score_actual
    return auc_score_actual / auc_score_perfect

def plot_principled_uplift_curve(y_true, uplift, treatment,
                    random=True, perfect=True, negative_effect=True, ax=None, name=None, title="AUC", **kwargs):
    """Plot Qini curves from predictions.

    Args:
        y_true (1d array-like): Ground truth (correct) binary labels.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        random (bool): Draw a random curve. Default is True.
        perfect (bool): Draw a perfect curve. Default is True.
        negative_effect (bool): If True, optimum Qini Curve contains the negative effects
            (negative uplift because of campaign). Otherwise, optimum Qini Curve will not
            contain the negative effects. Default is True.
        ax (object): The graph on which the function will be built. Default is None.
        name (string): The name of the function. Default is None.

    Returns:
        Object that stores computed values.

    Example::

        from sklift.viz import plot_qini_curve


        qini_disp = plot_qini_curve(
            y_test, uplift_predicted, trmnt_test,
            perfect=True, name='Model name'
        );

        qini_disp.figure_.suptitle("Qini curve");
    """
    check_matplotlib_support('plot_qini_curve')
    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)

    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)
    y_actual = principled_uplift_curve(y_true, uplift, treatment)
    x_actual = np.array(range(len(y_actual)))

    if perfect:
        y_perfect = max_principled_uplift_curve(y_true, treatment)
    else:
        y_perfect = None

    if random:
        y_baseline = x_actual * y_perfect[-1] / len(y_true)
    else:
        y_baseline = None
    # print(y_baseline)


    auc = principled_uplift_auc_score(y_true, uplift, treatment)
    line_kwargs = {}
    if auc is not None and name is not None:
        line_kwargs["label"] = f"{name} ({title} = {auc:0.2f})"
    elif auc is not None:
        line_kwargs["label"] = f"{title} = {auc:0.2f}"
    elif name is not None:
        line_kwargs["label"] = name

    line_kwargs.update(**kwargs)

    if y_actual[0] != 0 or y_baseline[0] != 0 or y_perfect[0] != 0:
    # Add an extra threshold position if necessary
    # to make sure that the curve starts at (0, 0)
        y_actual = np.r_[0, y_actual]
        y_baseline = np.r_[0, y_baseline]
        y_perfect = np.r_[0, y_perfect]
        x_actual = np.r_[0, x_actual]

    # if y_baseline[-1] != y_actual[-1] or y_perfect[-1] != y_actual[-1]:
    #     print('y_perfect[-1] != y_actual[-1]')
    #     y_actual = np.r_[y_actual,y_baseline[-1]]
    #     y_baseline = np.r_[y_baseline,y_baseline[-1]]
    #     y_perfect = np.r_[y_perfect,y_baseline[-1]]
    #     x_actual = np.r_[x_actual,x_actual[-1]+1]
    if ax is None:
        fig, ax = plt.subplots()

        line_, = ax.plot(x_actual, y_actual, **line_kwargs)

    if random:
        
        ax.plot(x_actual, y_baseline, label="Random")
        ax.fill_between(x_actual, y_actual, y_baseline, alpha=0.2)

    if perfect:
        ax.plot(x_actual, y_perfect, label="Perfect")

    ax.set_xlabel('Number targeted')
    ax.set_ylabel('Number of incremental outcome')

    if random == perfect:
        variance = False
    else:
        variance = True

    if len(ax.lines) > 4:
        ax.lines.pop(len(ax.lines) - 1)
        if variance == False:
            ax.lines.pop(len(ax.lines) - 1)

    if "label" in line_kwargs:
        ax.legend(loc=u'upper left', bbox_to_anchor=(1, 1))

    return ax

# set p = n时，等价于sep-uplift-curve
def relative_uplift_curve(y_true, uplift, treatment):
    """Compute Uplift curve.

    For computing the area under the Uplift Curve, see :func:`.uplift_auc_score`.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.

    Returns:
        array (shape = [>2]), array (shape = [>2]): Points on a curve.

    See also:
        :func:`.uplift_auc_score`: Compute normalized Area Under the Uplift curve from prediction scores.

        :func:`.perfect_uplift_curve`: Compute the perfect Uplift curve.

        :func:`.plot_uplift_curve`: Plot Uplift curves from predictions.

        :func:`.qini_curve`: Compute Qini curve.

    References:
        Devriendt, F., Guns, T., & Verbeke, W. (2020). Learning to rank for uplift modeling. ArXiv, abs/2002.05897.
    """

    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)

    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    desc_score_indices = np.argsort(uplift, kind="mergesort")[::-1]
    y_true, uplift, treatment = y_true[desc_score_indices], uplift[desc_score_indices], treatment[desc_score_indices]

    y_true_ctrl, y_true_trmnt = y_true.copy(), y_true.copy()

    y_true_ctrl[treatment == 1] = 0
    y_true_trmnt[treatment == 0] = 0

    distinct_value_indices = np.where(np.diff(uplift))[0]
    threshold_indices = np.r_[distinct_value_indices, uplift.size - 1]

    num_trmnt = stable_cumsum(treatment)[threshold_indices]
    y_trmnt = stable_cumsum(y_true_trmnt)[threshold_indices]

    num_all = threshold_indices + 1

    num_ctrl = num_all - num_trmnt
    y_ctrl = stable_cumsum(y_true_ctrl)[threshold_indices]

    curve_values = (np.divide(y_trmnt, num_trmnt[-1], out=np.zeros_like(y_trmnt), where=num_trmnt != 0) -
                    np.divide(y_ctrl, num_ctrl[-1], out=np.zeros_like(y_ctrl), where=num_ctrl != 0))

    if num_all.size == 0 or curve_values[0] != 0 or num_all[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        num_all = np.r_[0, num_all]
        curve_values = np.r_[0, curve_values]

    return num_all, curve_values

def perfect_relative_uplift_curve(y_true, treatment):
    """Compute the perfect (optimum) Uplift curve.

    This is a function, given points on a curve.  For computing the
    area under the Uplift Curve, see :func:`.uplift_auc_score`.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        treatment (1d array-like): Treatment labels.

    Returns:
        array (shape = [>2]), array (shape = [>2]): Points on a curve.

    See also:
        :func:`.uplift_curve`: Compute the area under the Qini curve.

        :func:`.uplift_auc_score`: Compute normalized Area Under the Uplift curve from prediction scores.

        :func:`.plot_uplift_curve`: Plot Uplift curves from predictions.
    """

    check_consistent_length(y_true, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)

    y_true, treatment = np.array(y_true), np.array(treatment)

    # perfect_uplift = np.zeros_like(y_true)
    # perfect_uplift[(y_true == 1) & (treatment == 1)] = 1
    # perfect_uplift[(y_true == 0) & (treatment == 1)] = -1
    cr_num = np.sum((y_true == 1) & (treatment == 0))  # Control Responders
    tn_num = np.sum((y_true == 0) & (treatment == 1))  # Treated Non-Responders

    # express an ideal uplift curve through y_true and treatment
    summand = y_true if cr_num > tn_num else treatment
    perfect_uplift = 2 * (y_true == treatment) + summand

    return relative_uplift_curve(y_true, perfect_uplift, treatment)


def relative_uplift_auc_score(y_true, uplift, treatment):
    """Compute normalized Area Under the Uplift Curve from prediction scores.

    By computing the area under the Uplift curve, the curve information is summarized in one number.
    For binary outcomes the ratio of the actual uplift gains curve above the diagonal to that of
    the optimum Uplift Curve.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.

    Returns:
        float: Area Under the Uplift Curve.

    See also:
        :func:`.uplift_curve`: Compute Uplift curve.

        :func:`.perfect_uplift_curve`: Compute the perfect (optimum) Uplift curve.

        :func:`.plot_uplift_curve`: Plot Uplift curves from predictions.

        :func:`.qini_auc_score`: Compute normalized Area Under the Qini Curve from prediction scores.
    """

    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    x_actual, y_actual = relative_uplift_curve(y_true, uplift, treatment)
    x_perfect, y_perfect = perfect_relative_uplift_curve(y_true, treatment)
    x_baseline, y_baseline = np.array([0, x_perfect[-1]]), np.array([0, y_perfect[-1]])

    auc_score_baseline = auc(x_baseline, y_baseline)
    auc_score_perfect = auc(x_perfect, y_perfect) - auc_score_baseline
    auc_score_actual = auc(x_actual, y_actual) - auc_score_baseline

    return auc_score_actual / auc_score_perfect

def sep_qini_curve(y_true, uplift, treatment):
    """Compute Qini curve.

    For computing the area under the Qini Curve, see :func:`.qini_auc_score`.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.

    Returns:
        array (shape = [>2]), array (shape = [>2]): Points on a curve.

    See also:
        :func:`.uplift_curve`: Compute the area under the Qini curve.

        :func:`.perfect_qini_curve`: Compute the perfect Qini curve.

        :func:`.plot_qini_curves`: Plot Qini curves from predictions..

        :func:`.uplift_curve`: Compute Uplift curve.

    References:
        Nicholas J Radcliffe. (2007). Using control groups to target on predicted lift:
        Building and assessing uplift model. Direct Marketing Analytics Journal, (3):14–21, 2007.

        Devriendt, F., Guns, T., & Verbeke, W. (2020). Learning to rank for uplift modeling. ArXiv, abs/2002.05897.
    """

    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    desc_score_indices = np.argsort(uplift, kind="mergesort")[::-1]

    y_true = y_true[desc_score_indices]
    treatment = treatment[desc_score_indices]
    uplift = uplift[desc_score_indices]

    y_true_ctrl, y_true_trmnt = y_true.copy(), y_true.copy()

    y_true_ctrl[treatment == 1] = 0
    y_true_trmnt[treatment == 0] = 0

    distinct_value_indices = np.where(np.diff(uplift))[0]
    threshold_indices = np.r_[distinct_value_indices, uplift.size - 1]

    num_trmnt = stable_cumsum(treatment)[threshold_indices]
    y_trmnt = stable_cumsum(y_true_trmnt)[threshold_indices]

    num_all = threshold_indices + 1

    num_ctrl = num_all - num_trmnt
    y_ctrl = stable_cumsum(y_true_ctrl)[threshold_indices]

    curve_values = y_trmnt - y_ctrl * np.divide(num_trmnt[-1], num_ctrl[-1], out=np.zeros_like(num_trmnt), where=num_ctrl != 0)
    if num_all.size == 0 or curve_values[0] != 0 or num_all[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        num_all = np.r_[0, num_all]
        curve_values = np.r_[0, curve_values]

    return num_all, curve_values


def perfect_sep_qini_curve(y_true, treatment, negative_effect=True):
    """Compute the perfect (optimum) Qini curve.

    For computing the area under the Qini Curve, see :func:`.qini_auc_score`.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        treatment (1d array-like): Treatment labels.
        negative_effect (bool): If True, optimum Qini Curve contains the negative effects
            (negative uplift because of campaign). Otherwise, optimum Qini Curve will not
            contain the negative effects.
    Returns:
        array (shape = [>2]), array (shape = [>2]): Points on a curve.

    See also:
        :func:`.qini_curve`: Compute Qini curve.

        :func:`.qini_auc_score`: Compute the area under the Qini curve.

        :func:`.plot_qini_curves`: Plot Qini curves from predictions..
    """

    check_consistent_length(y_true, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)
    n_samples = len(y_true)

    y_true, treatment = np.array(y_true), np.array(treatment)

    if not isinstance(negative_effect, bool):
        raise TypeError(f'Negative_effects flag should be bool, got: {type(negative_effect)}')

    # express an ideal uplift curve through y_true and treatment
    if negative_effect:
        x_perfect, y_perfect = sep_qini_curve(
            y_true, y_true * treatment - y_true * (1 - treatment), treatment
        )
    else:
        ratio_random = (y_true[treatment == 1].sum() - len(y_true[treatment == 1]) *
                        y_true[treatment == 0].sum() / len(y_true[treatment == 0]))

        x_perfect, y_perfect = np.array([0, ratio_random, n_samples]), np.array([0, ratio_random, ratio_random])

    return x_perfect, y_perfect


def sep_qini_auc_score(y_true, uplift, treatment, negative_effect=True):
    """Compute normalized Area Under the Qini curve (aka Qini coefficient) from prediction scores.

    By computing the area under the Qini curve, the curve information is summarized in one number.
    For binary outcomes the ratio of the actual uplift gains curve above the diagonal to that of
    the optimum Qini curve.

    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        negative_effect (bool): If True, optimum Qini Curve contains the negative effects
            (negative uplift because of campaign). Otherwise, optimum Qini Curve will not contain the negative effects.

            .. versionadded:: 0.2.0

    Returns:
        float: Qini coefficient.

    See also:
        :func:`.qini_curve`: Compute Qini curve.

        :func:`.perfect_qini_curve`: Compute the perfect (optimum) Qini curve.

        :func:`.plot_qini_curves`: Plot Qini curves from predictions..

        :func:`.uplift_auc_score`: Compute normalized Area Under the Uplift curve from prediction scores.

    References:
        Nicholas J Radcliffe. (2007). Using control groups to target on predicted lift:
        Building and assessing uplift model. Direct Marketing Analytics Journal, (3):14–21, 2007.
    """

    # TODO: Add Continuous Outcomes
    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    if not isinstance(negative_effect, bool):
        raise TypeError(f'Negative_effects flag should be bool, got: {type(negative_effect)}')

    x_actual, y_actual = sep_qini_curve(y_true, uplift, treatment)
    x_perfect, y_perfect = perfect_sep_qini_curve(y_true, treatment, negative_effect)
    x_baseline, y_baseline = np.array([0, x_perfect[-1]]), np.array([0, y_perfect[-1]])

    auc_score_baseline = auc(x_baseline, y_baseline)
    auc_score_perfect = auc(x_perfect, y_perfect) - auc_score_baseline
    auc_score_actual = auc(x_actual, y_actual) - auc_score_baseline

    return auc_score_actual / auc_score_perfect

def get_perfect_gain_auc(test_data, percentage):
    test_data= np.array(test_data)
    desc_score_indices = np.argsort(test_data[:, 12], kind="mergesort", axis=0)[::-1]
    test_data = test_data[desc_score_indices]
    cate = test_data[:, 12]
    cate = cate[:int(len(cate) * percentage)]
    principal_0 = stable_cumsum(cate > 0)
    principal_2 = stable_cumsum(cate < 0)


    gain = (principal_0 - principal_2)
    x_actual = np.array(range(len(gain)))
    auc_score_perfect = auc(x_actual, gain)
    return auc_score_perfect

def get_true_gain_auc(test_data, uplift, percentage):
    auc_score_perfect = get_perfect_gain_auc(test_data, percentage)

    # 获取真正的cate大于0的样本，即使t=0,y=0。
    check_consistent_length(test_data, uplift)
    test_data, uplift = np.array(test_data), np.array(uplift)

    # [::-1] 是反转序列的切片操作。这里是将 uplift 按照从大到小的顺序排列
    desc_score_indices = np.argsort(uplift, kind="mergesort", axis=0)[::-1]

    test_data = test_data[desc_score_indices]
    uplift = uplift[desc_score_indices]

    cate = test_data[:, 12]

    cate = cate[:int(len(cate) * percentage)]

    # principal_0 = np.sum(principal == 0)
    # principal_2 = np.sum(principal == 2)
    # principal_1 = np.sum(principal == 1)

    principal_0 = stable_cumsum(cate > 0)
    principal_2 = stable_cumsum(cate < 0)


    gain = (principal_0 - principal_2)
    x_actual = np.array(range(len(gain)))
    auc_score_actual = auc(x_actual, gain)
    # return gain
    return auc_score_actual / auc_score_perfect

def get_true_gain(test_data, uplift, percentage):
    # 获取真正的cate大于0的样本，即使t=0,y=0。
    check_consistent_length(test_data, uplift)
    test_data, uplift = np.array(test_data), np.array(uplift)

    # [::-1] 是反转序列的切片操作。这里是将 uplift 按照从大到小的顺序排列
    desc_score_indices = np.argsort(uplift, kind="mergesort", axis=0)[::-1]

    test_data = test_data[desc_score_indices]
    uplift = uplift[desc_score_indices]

    cate = test_data[:, 12]

    cate = cate[:int(len(cate) * percentage)]

    principal_0 = np.sum(cate>0)
    principal_2 = np.sum(cate<0)
    # principal_1 = np.sum(principal == 1)

    # principal_0 = stable_cumsum(principal == 0)
    # principal_2 = stable_cumsum(principal == 2)


    gain = (principal_0 - principal_2)
    # x_actual = np.array(range(len(gain)))
    # auc_score_actual = auc(x_actual, gain)
    return gain
    # return auc_score_actual