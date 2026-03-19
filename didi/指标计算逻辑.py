# -*- coding: utf-8 -*-
import json
import os
import time


import datapane as dp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer
from pyspark.sql import functions as F
from pyspark.sql.types import *

pd.set_option('display.precision', 4)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.min_rows', 10000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_colwidth', 10000)
pd.set_option('display.max_info_rows', 10000)
pd.set_option('display.max_info_columns', 10000)
os.environ["DATAPANE_BY_DATAPANE"] = "true"

# If run on ether, html report will be put on hdfs, else it will be saved as an local file.
RUN_ON_ETHER = True
TRT_RATE_COL = 'trt_rate'
RANDOM_COL = 'random'
ACTUAL_COL = 'actual'
PREDICT_COL = 'predict'


class EtherEval:
    def __init__(self, model="ecr"):
        self.type = 'LR'
        self.data = []
        self.model = model

    def add_eval_res(self, name="", title="", explain="", score=0):
        if not self.model.strip():
            er = {"name": name, "title": title,
                  "explain": explain, "score": float(score)}
        else:
            new_name = "{0}_{1}".format(self.model, name)
            new_title = "{0}_{1}".format(self.model, title)
            er = {"name": new_name, "title": new_title,
                  "explain": explain, "score": float(score)}
        self.data.append(er)

    def get_json_str(self):
        for item in self.data:
            print(item)

        return json.dumps(self.__dict__)


class CausalModel:

    def __init__(self, spark, table, treatments, control, treatment_col, elas_pred_col, label_col, offset, treatment_weight, cate=None):
        """
        table: prediction table
        treatments: [100095, ...100100]
        control: 100100
        treatment_col: treatment column name
        elas_pred_col: ite prediction value column name
        label_col: label column name
        offset: index to select product type in multi_label_GRF output
        """
        self.spark = spark
        self.table = table
        self.control = control
        self.treatments = treatments
        self.treatment_col = treatment_col
        self.elas_pred_col = elas_pred_col
        self.label_col = label_col
        self.offset = offset
        self.bucket_cate_mape = cate
        self.treatment_weight = treatment_weight

    @staticmethod
    def calc_qini(df, outcome_col, treatment_col='trt', predict_col='ite'):
        # outcome_col 就是 th_abr
        # 根据模型得到的uplift进行排序, 并按排序后的顺序重制index
        df = df.sort_values(predict_col, ascending=False).reset_index(drop=True)
        df.index = df.index + 1 # 将索引从0到n-1改为从1到n, 为了语义符合 "前k个样本"

        df['cumsum_tr'] = df[treatment_col].cumsum()  # 进行累计, 第k个元素代表前k个样本中treatment的数量
        df['cumsum_ct'] = df.index.values - df['cumsum_tr']  # 第k个元素代表前k个样本中control的数量
        # 两个都是01值, 因此乘起来再累加等于前k个元素中treatment组中label=1的个数
        df['cumsum_tr_y'] = (df[outcome_col] * df[treatment_col]).cumsum() 
        # 相当于前k个元素中control组中label为1的个数
        df['cumsum_ct_y'] = (
            df[outcome_col] * (1 - df[treatment_col])).cumsum()
        
        # 这个比例是为了对df[PREDICT_COL]进行放缩, 由于df[ACTUAL_COL]计算的是treatment组的uplift, 
        # 而模型给出的df[PREDICT_COL]是所有样本的, 因此为了和df[ACTUAL_COL]可比（计算mape）, 需要乘一个treatment的比例进行拉齐
        df[TRT_RATE_COL] = df['cumsum_tr'] / df.index.to_series() # 前k个样本中treatment所占比例
        df[PREDICT_COL] = df[predict_col].cumsum() * df[TRT_RATE_COL] # 对估计的uplift曲线进行加权
        # 计算的是实际的uplift, 也就是我们的qini值
        # 对control组进行加权, 使两组的人数相等, 将control组的结果当成treatment组的反事实结果, 因此这个式子计算的是treatment这一组人实际的Y的增量
        df[ACTUAL_COL] = df['cumsum_tr_y'] - df['cumsum_ct_y'] * df['cumsum_tr'] / df['cumsum_ct'] 
        # 随机基线的计算简单将起点终点进行连接即可, 斜率描述的是这一整个数据中, treatment带来的提升
        df[RANDOM_COL] = df[ACTUAL_COL].values[-1] / df.index.values[-1] * df.index.to_series()

        qini = df[[TRT_RATE_COL, RANDOM_COL,
                   ACTUAL_COL, PREDICT_COL]].interpolate() # 插值填补空值
        total_qini = qini[ACTUAL_COL].values[-1]
        score_qini = (qini[ACTUAL_COL] - qini[RANDOM_COL]).mean() / total_qini # 标准化
        score_mape = (qini[ACTUAL_COL] - qini[PREDICT_COL]).abs().mean() / total_qini # 估计的uplift和实际的uplift之间的差值
        score_copc = qini[ACTUAL_COL].values[-1] / qini[PREDICT_COL].values[-1]
        error_rate = df[predict_col][df[predict_col] *
                                     total_qini < 0].count() / df[predict_col].count() # 预测的uplift和total_qini反向的样本比例

        auuc_lift1 = (qini[ACTUAL_COL] - qini[RANDOM_COL]).sum() # 即AUUC原始值
        auuc_lift2 = qini[RANDOM_COL].sum()
        auuc = (auuc_lift1) * 0.5 / auuc_lift2 # 进行放缩
        aucc_lift1 = (df[predict_col] *
                      (qini[ACTUAL_COL] - qini[RANDOM_COL])).sum()
        aucc_lift2 = (df[predict_col] * (qini[RANDOM_COL])).sum()
        aucc_ite = (aucc_lift1 + aucc_lift2) * 0.5 / aucc_lift2

        return qini, total_qini, auuc, aucc_ite, score_qini, score_mape, score_copc, error_rate

    def get_qini_profile(self, col_num=3, row_width=8, col_width=8):
        # data是一个pandas.dataframe分别有列 
        # | exp_discount_cross | th_delta_ecr  | th_abr    |
        # |    treatment_col   | elas_pred_col | label_col |
        data = self.spark.sql('select * from %s' % self.table) \
            .select(F.col(self.treatment_col), F.col(self.elas_pred_col), F.col(self.label_col)) \
            .toPandas()
        # 总共有多少个组合 - 对应画多少个图
        figure_num = len(self.treatments)
        # 也是画图参数 - 一行几个图/有几行
        row_num = np.ceil(figure_num / float(col_num))
        # 是每个treatment组合的比例？
        sample_num = sum(list(self.treatment_weight.values()))
        qini_curve = plt.figure(
            figsize=(col_width * col_num, row_width * row_num))
        qini_info = {'treatment': [], 'total_qini': [], 'auuc': [], 'aucc_ite': [], 'score_qini': [
        ], 'score_mape': [], 'score_copc': [], 'error_rate': [], 'treatment_weight': []}

        # 遍历所有的treatment组合进行记录
        for i in np.arange(figure_num):
            treatment = self.treatments[i] # 当前treatment的组合
            pair_data = data[data[self.treatment_col].isin(
                [self.control, treatment])].copy() # 取出符合control和当前treatment的所有样本的treatment组合
            pair_data['ite'] = pair_data[self.elas_pred_col].apply(
                lambda x: x[self.offset + i]) # 取出treatment对应的那一个delta_ecr - uplift
            # 用来标记是否是treatment组 - 将组合字符串变为布尔变量，
            # 例如100095就是0, 其余的是1, 
            # 注意我们这里是对pair_data进行处理, 在第一行时就进行了赋值, 这里面只包含了当前考察的treatment和control组样本
            pair_data['trt'] = pair_data[self.treatment_col].apply(
                lambda x: 0 if x == self.control else 1) 
            
            # 因此以上的操作是把当前考察的treatment取出来, 并将其对应的数据处理成一维的情况
            # pair_data 是一个dataframe :
            # |       treatment_col       |   elas_pred_col    | label_col |  ite          |         trt         ｜
            # | exp_discount_cross(字符串) | th_delta_ecr(数组)  | th_abr    | uplift(float) |是否treatment组(布尔值)｜
            
            (qini, total_qini, auuc, aucc_ite, score_qini, score_mape,
             score_copc, error_rate) = self.calc_qini(pair_data, self.label_col)
            treatment_sample_num = self.treatment_weight[treatment]
            treatment_ratio = treatment_sample_num * 1.0 / sample_num

            title = "[%s|qini:%.4f|-ite:%.4f|mape:%.4f]" % (
                str(treatment), score_qini, error_rate, score_mape)
            ax = plt.subplot(row_num, col_num, 1 + i)
            qini.plot(secondary_y=[TRT_RATE_COL], ax=ax,
                      rot=45, grid=True, title=title)
            ax.set_xlabel('ite_rank')
            ax.set_ylabel('uplift_y')
            ax.right_ax.set_ylabel(TRT_RATE_COL)

            qini_info['treatment'].append(str(treatment))
            qini_info['total_qini'].append(total_qini)
            qini_info['auuc'].append(auuc)
            qini_info['aucc_ite'].append(aucc_ite)
            qini_info['score_qini'].append(score_qini)
            qini_info['score_mape'].append(score_mape)
            qini_info['score_copc'].append(score_copc)
            qini_info['error_rate'].append(error_rate)
            qini_info['treatment_weight'].append(treatment_ratio)
        plt.subplots_adjust(wspace=0.23, hspace=0.3)
        qini_info = pd.DataFrame(qini_info)

        return qini_info, qini_curve

    def report_cm_subpart(self, metric_instance):
        qini_info, qini_curve = self.get_qini_profile()

        # pos_id = (qini_info['score_qini'].abs() < 1) & (qini_info['score_qini'] > 0) & (qini_info['auuc'] > 0)
        # neg_id = (qini_info['score_qini'].abs() < 1) & (qini_info['score_qini'] < 0) & (qini_info['auuc'] < 0)
        # err_id = (qini_info['score_qini'].abs() >= 1) | (qini_info['score_qini'] * qini_info['auuc'] <= 0)

        pos_id = (qini_info['total_qini'] > 0)
        neg_id = (qini_info['total_qini'] <= 0)
        err_id = (qini_info['total_qini'].abs() < 0)
        
        qini_info['F1_uplift_score'] = (2 * (qini_info['score_qini'] / 0.5) * (0.5 - qini_info['score_mape'])) / ((qini_info['score_qini'] / 0.5) + (0.5 - qini_info['score_mape']))
        qini_info['weighted_score'] = qini_info['score_qini'] * \
            qini_info['treatment_weight']
        qini_info['weighted_mape'] = qini_info['score_mape'] * \
            qini_info['treatment_weight']
        qini_info['weighted_copc'] = qini_info['score_copc'] * \
            qini_info['treatment_weight']
        qini_info['weighted_auuc'] = qini_info['auuc'] * \
            qini_info['treatment_weight']
        qini_info['weighted_aucc_ite'] = qini_info['aucc_ite'] * \
            qini_info['treatment_weight']

        pos_qini_num = qini_info['score_qini'][pos_id].count()
        neg_qini_num = qini_info['score_qini'][neg_id].count()
        abn_qini_num = qini_info['score_qini'][err_id].count()
        if pos_qini_num > 0:
            pos_f1_score = np.round(qini_info['F1_uplift_score'][pos_id].mean(), 6)
            pos_avg_qini = np.round(qini_info['score_qini'][pos_id].mean(), 6)
            pos_wavg_qini = np.round(
                qini_info['weighted_score'][pos_id].sum(), 6)
            pos_mape_qini = np.round(qini_info['score_mape'][pos_id].mean(), 6)
            pos_wmape_qini = np.round(
                qini_info['weighted_mape'][pos_id].sum(), 6)
            pos_copc = np.round(qini_info['score_copc'][pos_id].mean(), 6)
            pos_wcopc = np.round(qini_info['weighted_copc'][pos_id].sum(), 6)
            pos_auuc = np.round(qini_info['auuc'][pos_id].mean(), 6)
            pos_wauuc = np.round(qini_info['weighted_auuc'][pos_id].sum(), 6)
            pos_aucc_ite = np.round(qini_info['aucc_ite'][pos_id].mean(), 6)
            pos_waucc_ite = np.round(
                qini_info['weighted_aucc_ite'][pos_id].sum(), 6)
        else:
            pos_f1_score = 0.
            pos_avg_qini = 0.
            pos_wavg_qini = 0.
            pos_mape_qini = 0.
            pos_wmape_qini = 0.
            pos_copc = 0.
            pos_wcopc = 0.
            pos_auuc = 0.
            pos_wauuc = 0.
            pos_aucc_ite = 0.
            pos_waucc_ite = 0.

        if neg_qini_num > 0:
            neg_f1_score = np.round(qini_info['F1_uplift_score'][neg_id].mean(), 6)
            neg_avg_qini = np.round(qini_info['score_qini'][neg_id].mean(), 6)
            neg_wavg_qini = np.round(
                qini_info['weighted_score'][neg_id].sum(), 6)
            neg_mape_qini = np.round(qini_info['score_mape'][neg_id].mean(), 6)
            neg_wmape_qini = np.round(
                qini_info['weighted_mape'][neg_id].sum(), 6)
            neg_copc = np.round(qini_info['score_copc'][neg_id].mean(), 6)
            neg_wcopc = np.round(qini_info['weighted_copc'][neg_id].sum(), 6)
            neg_auuc = np.round(qini_info['auuc'][neg_id].mean(), 6)
            neg_wauuc = np.round(qini_info['weighted_auuc'][neg_id].sum(), 6)
            neg_aucc_ite = np.round(qini_info['aucc_ite'][neg_id].mean(), 6)
            neg_waucc_ite = np.round(
                qini_info['weighted_aucc_ite'][neg_id].sum(), 6)
        else:
            neg_f1_score = 0.
            neg_avg_qini = 0.
            neg_wavg_qini = 0.
            neg_mape_qini = 0.
            neg_wmape_qini = 0.
            neg_copc = 0.
            neg_wcopc = 0.
            neg_auuc = 0.
            neg_wauuc = 0.
            neg_aucc_ite = 0.
            neg_waucc_ite = 0.

        print(qini_info)
        # print("pos_avg_qini pos_wmape_qini pos_qini_num neg_avg_qini neg_wmape_qini neg_qini_num abn_qini_num:")
        # print(pos_avg_qini, pos_wmape_qini, pos_qini_num, neg_avg_qini, neg_wmape_qini, neg_qini_num, abn_qini_num)
        print("pos_f1_score pos_avg_qini pos_wavg_qini pos_mape_qini pos_wmape_qini pos_copc pos_wcopc pos_qini_num neg_f1_score neg_avg_qini neg_wavg_qini neg_mape_qini neg_wmape_qini neg_copc neg_wcopc neg_qini_num abn_qini_num:")
        print(pos_f1_score, pos_avg_qini, pos_wavg_qini, pos_mape_qini, pos_wmape_qini, pos_copc, pos_wcopc, pos_qini_num, neg_f1_score, neg_avg_qini, neg_wavg_qini, neg_mape_qini, neg_wmape_qini, neg_copc, neg_wcopc, neg_qini_num, abn_qini_num)
        print('pos_qini_treatments: ', qini_info['treatment'][pos_id].values)
        print('neg_qini_treatments: ', qini_info['treatment'][neg_id].values)
        print('err_qini_treatments: ', qini_info['treatment'][err_id].values)

        # metric_instance.add_eval_res(name="pos_qini_num", title="pos_qini_num", score=pos_qini_num)
        # metric_instance.add_eval_res(name="neg_qini_num", title="neg_qini_num", score=neg_qini_num)
        metric_instance.add_eval_res(
            name="abn_qini_num", title="abn_qini_num", score=abn_qini_num)

        metric_instance.add_eval_res(name="pos_f1_score", title="pos_f1_score", score=pos_f1_score)
        metric_instance.add_eval_res(
            name="pos_avg_qini", title="pos_avg_qini", score=pos_avg_qini)
        metric_instance.add_eval_res(
            name="pos_wavg_qini", title="pos_wavg_qini", score=pos_wavg_qini)
        metric_instance.add_eval_res(
            name="pos_mape_qini", title="pos_mape_qini", score=pos_mape_qini)
        metric_instance.add_eval_res(
            name="pos_wmape_qini", title="pos_wmape_qini", score=pos_wmape_qini)
        metric_instance.add_eval_res(
            name="pos_copc", title="pos_copc", score=pos_copc)
        metric_instance.add_eval_res(
            name="pos_wcopc", title="pos_wcopc", score=pos_wcopc)

        metric_instance.add_eval_res(name="neg_f1_score", title="neg_f1_score", score=neg_f1_score)
        metric_instance.add_eval_res(
            name="neg_avg_qini", title="neg_avg_qini", score=neg_avg_qini)
        metric_instance.add_eval_res(
            name="neg_wavg_qini", title="neg_wavg_qini", score=neg_wavg_qini)
        metric_instance.add_eval_res(
            name="neg_mape_qini", title="neg_mape_qini", score=neg_mape_qini)
        metric_instance.add_eval_res(
            name="neg_wmape_qini", title="neg_wmape_qini", score=neg_wmape_qini)
        metric_instance.add_eval_res(
            name="neg_copc", title="neg_copc", score=neg_copc)
        metric_instance.add_eval_res(
            name="neg_wcopc", title="neg_wcopc", score=neg_wcopc)

        cm_report_components = ['## Check Elastic Model Prediction',
                                'pos_f1_score: %.4f, pos_avg_qini: %.4f, pos_wavg_qini: %.4f, pos_mape: %.4f, pos_wmape: %.4f, pos_copc: %.4f, pos_wcopc: %.4f, pos_auuc: %.4f, pos_wauuc: %.4f, pos_aucc_ite: %.4f, pos_waucc_ite: %.4f, pos_qini_num: %d' % (
                                    pos_f1_score, pos_avg_qini, pos_wavg_qini, pos_mape_qini, pos_wmape_qini, pos_copc, pos_wcopc, pos_auuc, pos_wauuc, pos_aucc_ite, pos_waucc_ite, pos_qini_num),
                                'neg_f1_score: %.4f, neg_avg_qini: %.4f, neg_wavg_qini: %.4f, neg_mape: %.4f, neg_wmape: %.4f, neg_copc: %.4f, neg_wcopc: %.4f, neg_auuc: %.4f, neg_wauuc: %.4f, neg_aucc_ite: %.4f, neg_waucc_ite: %.4f, neg_qini_num: %d' % (
                                    neg_f1_score, neg_avg_qini, neg_wavg_qini, neg_mape_qini, neg_wmape_qini, neg_copc, neg_wcopc, neg_auuc, neg_wauuc, neg_aucc_ite, neg_waucc_ite, neg_qini_num),
                                'abn_qini_num: %d' % abn_qini_num,
                                'pos_qini_treatments: ' +
                                ','.join(
                                    qini_info['treatment'][pos_id].values),
                                'neg_qini_treatments: ' +
                                ','.join(
                                    qini_info['treatment'][neg_id].values),
                                'err_qini_treatments: ' +
                                ','.join(
                                    qini_info['treatment'][err_id].values),
                                dp.Table(qini_info, caption='Qini Score'),
                                dp.Plot(qini_curve, caption='Qini Curve')]

        if self.bucket_cate_mape:
            cm_report_components.append('## feature cate mape')
            cm_report_components.append(
                dp.Select(blocks=self.bucket_cate_mape))

        return cm_report_components


def make_report(spark,
                report_desc_string,
                model_param_map,
                table_param_map,
                report_component_keys):
    treatment_col = model_param_map['treatment_col']  # exp_discount_cross
    base_pred_col = model_param_map['base_pred_col']  # th_base_ecr
    elas_pred_col = model_param_map['elas_pred_col']  # th_delta_ecr
    label_col = model_param_map['label_col'] # th_abr
    control = model_param_map['control'] # 100095
    treatments = model_param_map['treatments'] # 所有的treatment组合
    offset = model_param_map['offset'] # 0
    model_name = model_param_map['model_name'] # 计算什么品类 - th_ecr
    treatment_weight = model_param_map['treatment_weight'] # 什么权重先验

    # table param
    elas_table = table_param_map['elas_table'] # 输入的表

    metric_instance = EtherEval(model_name)

    report_components = [report_desc_string]
    # causal model subpart
    if "causal_model_desc" in report_component_keys:
        cm = CausalModel(spark, elas_table, treatments, control, treatment_col,
                         elas_pred_col, label_col, offset, treatment_weight)
        report_components += cm.report_cm_subpart(metric_instance)

    return report_components, metric_instance


def save_metrics(spark, metric_instance, dfs_path):
    er_str = metric_instance.get_json_str()
    spark.sparkContext.parallelize(
        [er_str]).repartition(1).saveAsTextFile(dfs_path)


def save_reports(spark, report_components, dfs_path):
    if RUN_ON_ETHER:
        dirname = '%f' % time.time()
        os.system('mkdir -p %s' % dirname)
        filename = '%s/part-00000' % dirname
        dp.Report(*report_components).save(path=filename)
        spark.sparkContext.parallelize(
            [open(filename, 'r').read()], 1).saveAsTextFile(dfs_path)
        os.system('rm -r %s' % dirname)
    else:
        dp.Report(*report_components).save(path=dfs_path)


def start(spark, inputs, outputs):
    print('inputs: ', inputs)
    print('outputs: ', outputs)


    model_name = 'kc_ecr'
    if model_name == 'kc_ecr':
        model_param_map = {
        'treatment_col': 'kc_treatment',
        'base_pred_col': "kc_control_output",
        'elas_pred_col': 'kc_delta_output',
        'control': 104,
        'treatments': [103, 100, 97, 95],
        'treatment_weight':{103:0.25,100:0.25,97:0.25,95:0.25},
        }
        table_param_map = {
        'explore_table': '',
        'base_table': '',
        'elas_table': inputs[0],
        'simu_table': ''
        }

        model_param_map['label_col'] = 'kc_abr'
        model_param_map['offset'] = 0
        model_param_map['simu_pred_col'] = 'org_key_val_kuai'
        model_param_map['model_name'] = model_name

    elif model_name == 'kt_ecr':
        model_param_map = {
        'treatment_col': 'kc_treatment',
        'base_pred_col': "kt_control_output",
        'elas_pred_col': 'kt_delta_output',
        'control': 100,
        'treatments': [104, 103, 97, 95],
        'treatment_weight':{104:0.25,103:0.25,97:0.25,95:0.25},
        }

        table_param_map = {
        'explore_table': '',
        'base_table': '',
        'elas_table': inputs[0],
        'simu_table': ''
        }

        model_param_map['label_col'] = 'kt_abr'
        model_param_map['offset'] = 0
        model_param_map['simu_pred_col'] = 'org_key_val_kuaite'
        model_param_map['model_name'] = model_name
    else:
        print("Unsupported model: ", model_name)
        exit(-1)

    # report description - user defined string
    report_desc_str = """
    ## Report Basic Info
    - Control Treatment: {control} - Data Label: {label_col}
  """.format(**model_param_map)

    # report_component_keys: include 4 subpart: 'explore_desc','base_model_desc','causal_model_desc','simu_data_desc'
    report_component_keys = ['causal_model_desc']

    report_components, metric_instance = make_report(spark,
                                                     report_desc_str,
                                                     model_param_map,
                                                     table_param_map,
                                                     report_component_keys)
    save_metrics(spark, metric_instance, outputs[0])

    save_reports(spark, report_components, outputs[1])