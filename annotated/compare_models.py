import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import pandas as pd

import evaluate

if torch.cuda.is_available():
    device='cuda'
else:
    device= 'cpu'
torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#Specify models, checkpoints and fiducial parameter value
MODEL_PATH = 'results/checkpoint_samples/Run_2-17_14-28/0219_1919/validation/checkpoint_{}/samples.pkl' # sys.argv[1]
IDX_FIDUCIAL = 6 # sys.argv[2]
SAVE = True # sys.argv[3]
SAVEPATH = 'results/comparison/Run_2-17_14-28/0219_1919_valp10/' # sys.argv[4]
CHECKPOINTS = ['200000','220000','240000','260000']


def make_table(checkpoints, results_log, results_linear):
    rows = []
    for (ckp, log_stats, linear_stats) in zip(checkpoints, results_log, results_linear):
        res = {'checkpoint': ckp, 'gen50_true15_LOGfields_BIAS_pvalue': log_stats['gen50_true15']['bias_pvalue'],
               'PK_gen50_true15_LOGfields_meansem': [log_stats['gen50_true15']['pk_allparams'].mean(), log_stats['gen50_true15']['pk_allparams'].std(ddof=1)/np.sqrt(log_stats['gen50_true15']['pk_allparams'].size)],
               'PK_true_true_loo_LOGfields_meansem': [log_stats['true_loo']['pk_allparams'].mean(), log_stats['true_loo']['pk_allparams'].std(ddof=1)/np.sqrt(log_stats['true_loo']['pk_allparams'].size)],
               'PK_gen_true_loo_LOGfields_meansem': [log_stats['gen_loo']['pk_allparams'].mean(), log_stats['gen_loo']['pk_allparams'].std(ddof=1)/np.sqrt(log_stats['gen_loo']['pk_allparams'].size)],
               'PK_gen50_true15_LINfields_meansem': [linear_stats['gen50_true15']['pk_allparams'].mean(), linear_stats['gen50_true15']['pk_allparams'].std(ddof=1)/np.sqrt(linear_stats['gen50_true15']['pk_allparams'].size)],
               'PK_true_true_loo_LINfields_meansem': [linear_stats['true_loo']['pk_allparams'].mean(), linear_stats['true_loo']['pk_allparams'].std(ddof=1)/np.sqrt(linear_stats['true_loo']['pk_allparams'].size)],
               'PK_gen_true_loo_LINfields_meansem': [linear_stats['gen_loo']['pk_allparams'].mean(), linear_stats['gen_loo']['pk_allparams'].std(ddof=1)/np.sqrt(linear_stats['gen_loo']['pk_allparams'].size)]}
        rows.append(res)
    return pd.DataFrame(rows)
               

def analysis(checkpoints, path, save):
    print('STATS for ', MODEL_PATH)
    results_log = []
    results_linear = []
    for CKPNUM in checkpoints:
        # Pk and Bias for LOG fields
        logstats = evaluate.perform_powspec_analysis_short(MODEL_PATH, CKPNUM, idx_fiducial=IDX_FIDUCIAL, logfields=True, normalize=False, plot_chisqs=False)
        print('Fiducial Param: ', logstats['params'][IDX_FIDUCIAL])
        # Pk for LINEAR fields
        linearstats = evaluate.perform_powspec_analysis_short(MODEL_PATH, CKPNUM, idx_fiducial=IDX_FIDUCIAL, logfields=False, normalize=True, plot_chisqs=False)
        print('Fiducial Param: ', logstats['params'][IDX_FIDUCIAL])
        results_log.append(logstats)
        results_linear.append(linearstats)
        print("Number of parameters", logstats['params'].shape)
    
    combined = make_table(checkpoints, results_log, results_linear)
    if save:
        if not os.path.exists(path):
            os.makedirs(path)
        combined.to_csv(path+'/results.csv', index=False)
    return combined


if __name__=='__main__':
    stats = analysis(CHECKPOINTS, SAVEPATH, SAVE)
    print(stats)