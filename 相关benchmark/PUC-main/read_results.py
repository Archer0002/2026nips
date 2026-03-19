import numpy as np
# synthetic
# path = '/data/zhuminqin/PrincipleUplift/log/2025-3-27/synthetic/tarnet_pu/tarnet_pu.txt'
# path = '/data/zhuminqin/PrincipleUplift/log/2025-3-27/synthetic/euen_pu/euen_pu.txt'
path = '/data/zhuminqin/PrincipleUplift/log/2025-5-6/synthetic/ptonet/ptonet.txt'
# path = '/data/zhuminqin/PrincipleUplift/log/2025-3-27/synthetic/tlearner_pu/tlearner_pu.txt'

pehes = np.zeros(50)
pre_pus = np.zeros(50)
sep_uplifts = np.zeros(50)
sep_qinis = np.zeros(50)
join_uplifts = np.zeros(50)
join_qinis = np.zeros(50)
pred_pus = np.zeros(50)
gain_aucs = np.zeros(50)
with open(path, 'r') as f:
    results = f.readlines()
    n = len(results)
    for i in range(n-50, n):
        pehe = float(results[i].split()[1].split('(')[1].split(')')[0])
        sep_uplift = float(results[i].split()[2].split(':')[1])
        sep_qini = float(results[i].split()[3].split(':')[1])
        join_uplift = float(results[i].split()[4].split(':')[1])
        join_qini = float(results[i].split()[5].split(':')[1])
        pred_pu = float(results[i].split()[6].split(':')[1])
        gain_auc = float(results[i].split()[-1].split(':')[1])
        pehes[i-n+50] = pehe
        sep_uplifts[i-n+50] = sep_uplift
        sep_qinis[i-n+50] = sep_qini
        join_uplifts[i-n+50] = join_uplift
        join_qinis[i-n+50] = join_qini
        pred_pus[i-n+50] = pred_pu
        gain_aucs[i-n+50] = gain_auc

print('pehe: ' + str(round(pehes.mean(),4))+ ' ± '+ str(round(pehes.std(),3)))
print('SUC: ' + str(round(sep_uplifts.mean(),4))+ ' ± '+ str(round(sep_uplifts.std(),3)))
print('SQC: ' + str(round(sep_qinis.mean(),4))+ ' ± '+ str(round(sep_qinis.std(),3)))
print('JUC: ' + str(round(join_uplifts.mean(),4))+ ' ± '+ str(round(join_uplifts.std(),3)))
print('JQC: ' + str(round(join_qinis.mean(),4))+ ' ± '+ str(round(join_qinis.std(),3)))
print('PUC: ' + str(round(pred_pus.mean(),4))+ ' ± '+ str(round(pred_pus.std(),3)))
print('AUTGC: ' + str(round(gain_aucs.mean(),4))+ ' ± '+ str(round(gain_aucs.std(),3)))

# # criteo
# path = '/data/zhuminqin/PrincipleUplift/log/2024-8-1/criteo/slearner/slearner.txt'

# pre_pus = np.zeros(50)
# sep_uplifts = np.zeros(50)
# sep_qinis = np.zeros(50)
# join_uplifts = np.zeros(50)
# join_qinis = np.zeros(50)
# pred_pus = np.zeros(50)

# with open(path, 'r') as f:
#     results = f.readlines()
#     n = len(results)
#     for i in range(n-50, n):
#         # print(i,results[i].split()[1])
#         sep_uplift = float(results[i].split()[1].split(':')[1])
#         sep_qini = float(results[i].split()[2].split(':')[1])
#         join_uplift = float(results[i].split()[3].split(':')[1])
#         join_qini = float(results[i].split()[4].split(':')[1])
#         pred_pu = float(results[i].split()[5].split(':')[1])
#         sep_uplifts[i-n+50] = sep_uplift
#         sep_qinis[i-n+50] = sep_qini
#         join_uplifts[i-n+50] = join_uplift
#         join_qinis[i-n+50] = join_qini
#         pred_pus[i-n+50] = pred_pu

# print('sep_uplift: ' + str(round(sep_uplifts.mean(),4))+ ' ± '+ str(round(sep_uplifts.std(),3)))
# print('sep_qini: ' + str(round(sep_qinis.mean(),4))+ ' ± '+ str(round(sep_qinis.std(),3)))
# print('join_uplift: ' + str(round(join_uplifts.mean(),4))+ ' ± '+ str(round(join_uplifts.std(),3)))
# print('join_qini: ' + str(round(join_qinis.mean(),4))+ ' ± '+ str(round(join_qinis.std(),3)))
# print('pred_pu: ' + str(round(pred_pus.mean(),4))+ ' ± '+ str(round(pred_pus.std(),3)))
