import os
import torch
import pickle
import random
import torch.nn.functional as F
import numpy as np
import yaml
import matplotlib.pyplot as plt
from einops import rearrange
#from quantimpy import minkowski as mk
from torch.utils.data import DataLoader
from torch import nn
import sys
import time
from scipy import stats

import hf_diffusion
import utils
import main_helper

def get_mean_sem(rchisq):
    return rchisq.mean(), rchisq.std(ddof=1)/np.sqrt(len(rchisq))

def perform_powspec_analysis(model_path, ckpnum, idx_fiducial=None):
    samppath = model_path.format(ckpnum)
    resdict = pickle.load(open(samppath, 'rb'))
    allsamps = resdict['samples'][0][:, 0]

    print('First examine consistency of the pk of log fields')
    logfields=True
    truefields, val_params, rchisqmat = evaluate_paramwise_reducedchisq(samppath, 50, data_subtype='validation',
                                    return_truefields=True, logfields=logfields)

    print('CHECKPOINT: ', ckpnum, 'MODEL: ', model_path)
    print('Performing Analysis for:')
    print(val_params)
    print('#### Rchisq Matrix: 50 Gen vs 15 True ########')
    print(rchisqmat.mean(axis=1), np.std(rchisqmat, ddof=1, axis=1)/np.sqrt(rchisqmat.shape[1]))
    print("##############")
    print('Ckp: {} || Mean|Std across params= {:.2f} {:.2f}'.format(ckpnum, rchisqmat.mean(), rchisqmat.std(ddof=1)))
    print('Fiducial: ', rchisqmat.mean(axis=1)[idx_fiducial], rchisqmat.std(axis=1, ddof=1)[idx_fiducial])
    gen_true = {'log_pk_allparams': rchisqmat, 'log_pk_fiducial': [rchisqmat.mean(axis=1)[idx_fiducial], rchisqmat.std(axis=1, ddof=1)[idx_fiducial]]}
    print("##############")
    print('#### Leave one out: 15 Gen or True vs 14 other True #######')
    
    ttrchisq = []
    tgrchisq = []
    ggrchisq = []
    for ip in range(len(val_params)):
        print('+++++++++++++++')
        Btrue = 15
        ipsubset = np.arange(ip*Btrue, (ip+1)*Btrue)
        rchisq_true = leave_one_out_chisq_z(truefields[ipsubset], truefields[ipsubset], logfields=logfields)
        rchisq_mean, rchisq_sem = get_mean_sem(rchisq_true)
        ttrchisq.append((rchisq_mean, rchisq_sem))
        print(f'Params Om_m: {val_params[ip, 0]:.2f}, sig8: {val_params[ip, 1]:.2f}')
        print(f'A. True v True [Intrinsic Variability]')
        print(f'Mean RChisq: {rchisq_mean:.2f}, SEM: {rchisq_sem:.2f}')

        Bsamp = 50
        fields_samp = allsamps[np.arange(ip*Bsamp, (ip+1)*Bsamp)]
        samp15 = fields_samp[:15]
        rchisq_samp = leave_one_out_chisq_z(truefields[ipsubset], samp15, logfields=logfields)
        rchisq_mean, rchisq_sem = get_mean_sem(rchisq_samp)
        print(f'B. True v Gen [Similarity]')
        tgrchisq.append((rchisq_mean, rchisq_sem))
        print(f'Mean RChisq: {rchisq_mean:.2f}, SEM: {rchisq_sem:.2f}')

        rchisq_samp = leave_one_out_chisq_z(samp15, samp15, logfields=logfields)
        rchisq_mean, rchisq_sem = get_mean_sem(rchisq_samp)

        ggrchisq.append((rchisq_mean, rchisq_sem))
        print(f'C. Gen v Gen [Intrinsic Variability: Generated]')
        print(f'Mean RChisq: {rchisq_mean:.2f}, SEM: {rchisq_sem:.2f}')

        print('Another Subset of 15 generated fields')
        samp15 = fields_samp[15:30]
        rchisq_samp = leave_one_out_chisq_z(truefields[ipsubset], samp15, logfields=logfields)
        rchisq_mean, rchisq_sem = get_mean_sem(rchisq_samp)
        print(f'B. True v Gen [Similarity]')
        print(f'Mean RChisq: {rchisq_samp.mean():.2f}, SEM: {rchisq_samp.std()/np.sqrt(samp15.shape[0]):.2f}')

        rchisq_samp = leave_one_out_chisq_z(samp15, samp15, logfields=logfields)
        rchisq_mean, rchisq_sem = get_mean_sem(rchisq_samp)
        print(f'C. Gen v Gen [Intrinsic Variability: Generated]')
        print(f'Mean RChisq: {rchisq_samp.mean():.2f}, SEM: {rchisq_samp.std()/np.sqrt(samp15.shape[0]):.2f}')
        print('===========')
        print('Using 25 generated fields as a reference')
        Bsamp = 50
        fields_samp = allsamps[np.arange(ip*Bsamp, (ip+1)*Bsamp)]
        sampref = fields_samp[:25]

        rchisq_samp = leave_one_out_chisq_z(sampref, truefields[ipsubset], logfields=logfields)
        rchisq_mean, rchisq_sem = get_mean_sem(rchisq_samp)
        print(f'B. Gen v True [Similarity]')
        print(f'Mean RChisq: {rchisq_mean:.2f}, SEM: {rchisq_sem:.2f}')
        rchisq_samp = leave_one_out_chisq_z(sampref, fields_samp[-15:], logfields=logfields)
        rchisq_mean, rchisq_sem = get_mean_sem(rchisq_samp)
        print(f'C. Gen v Gen [Similarity]')
        print(f'Mean RChisq: {rchisq_mean:.2f}, SEM: {rchisq_sem:.2f}')

        print('Another Subset of 25 generated fields')
        sampref = fields_samp[25:]
        rchisq_samp = leave_one_out_chisq_z(sampref, truefields[ipsubset], logfields=logfields)
        rchisq_mean, rchisq_sem = get_mean_sem(rchisq_samp)
        print(f'B. Gen v True [Similarity]')
        print(f'Mean RChisq: {rchisq_mean:.2f}, SEM: {rchisq_sem:.2f}')
        rchisq_samp = leave_one_out_chisq_z(sampref, fields_samp[:15], logfields=logfields)
        rchisq_mean, rchisq_sem = get_mean_sem(rchisq_samp)
        print(f'C. Gen v Gen [Similarity]')
        print(f'Mean RChisq: {rchisq_mean:.2f}, SEM: {rchisq_sem:.2f}')

    print("##############")
    rchisq_truetrue = []
    for ip in range(len(val_params)):
        Btrue = 15
        ipsubset = np.arange(ip*Btrue, (ip+1)*Btrue)
        rchisq_true = leave_one_out_chisq_z(truefields[ipsubset], truefields[ipsubset], logfields=logfields)
        rchisq_truetrue.extend(rchisq_true)
    
    rchisq_truegen = []
    for ip in range(len(val_params)):
        Btrue = 15
        ipsubset = np.arange(ip*Btrue, (ip+1)*Btrue)
        Bsamp = 50
        fields_samp = allsamps[np.arange(ip*Bsamp, (ip+1)*Bsamp)]
        samp15 = fields_samp[:15]
        rchisq_samp = leave_one_out_chisq_z(truefields[ipsubset], samp15, logfields=logfields)
        rchisq_truegen.extend(rchisq_samp)

    plt.figure()
    plt.hist(np.array(rchisq_truetrue), bins=20, density=True, alpha=0.5, label='Ref: True[14] | True [15]', range=[0, 9])
    plt.legend()
    plt.show()
    plt.figure()
    plt.hist(np.array(rchisq_truegen), bins=20, density=True, alpha=0.5, label='Ref: True[14] | Gen [15]', range=[0, 9])
    plt.legend()
    plt.show()
    plt.figure()
    plt.hist(rchisqmat.flatten(), bins=20, density=True, alpha=0.5, label='Ref: True[15] | Gen [50]', range=[0, 9])
    plt.legend()
    plt.show()
    tgmeans = np.array([t[0] for t in tgrchisq])
    print("##############")
    print(f'Leave one out: {tgmeans.mean():.2f}')
    print('Fiducial: ', tgmeans[idx_fid])
    print("##############")
    plt.figure()
    im = plt.scatter(val_params[:, 0], val_params[:, 1], c=[t[0] for t in ttrchisq])
    plt.colorbar(im)
    plt.xlabel(r'$\Omega_m$')
    plt.ylabel(r'$\sigma_8$')
    plt.title('Train-Train LOO Rchisq')
    plt.show()

    plt.figure()
    im = plt.scatter(val_params[:, 0], val_params[:, 1], c=[t[0] for t in tgrchisq])
    plt.colorbar(im)
    plt.xlabel(r'$\Omega_m$')
    plt.ylabel(r'$\sigma_8$')
    plt.title('Train-Generated LOO Rchisq')
    plt.show()

    plt.figure()
    im = plt.scatter(val_params[:, 0], val_params[:, 1], c=[t[0] for t in ggrchisq])
    plt.colorbar(im)
    plt.xlabel(r'$\Omega_m$')
    plt.ylabel(r'$\sigma_8$')
    plt.title('Generated-Generated LOO Rchisq')
    plt.show()

    plt.figure()
    plt.errorbar(val_params[:, 0], [t[0] for t in ttrchisq], [t[1] for t in ttrchisq], label='True-True', fmt='o')
    plt.errorbar(val_params[:, 0], [t[0] for t in tgrchisq], [t[1] for t in tgrchisq], label='True-Gen', fmt='o')
    plt.errorbar(val_params[:, 0], [t[0] for t in ggrchisq], [t[1] for t in ggrchisq], label='Gen-Gen', fmt='o')
    plt.legend()
    plt.xlabel(r'$\Omega_m$')
    plt.show()

    plt.figure()
    plt.errorbar(val_params[:, 1], [t[0] for t in ttrchisq], [t[1] for t in ttrchisq], label='True-True', fmt='o')
    plt.errorbar(val_params[:, 1], [t[0] for t in tgrchisq], [t[1] for t in tgrchisq], label='True-Gen', fmt='o')
    plt.errorbar(val_params[:, 1], [t[0] for t in ggrchisq], [t[1] for t in ggrchisq], label='Gen-Gen', fmt='o')
    plt.legend()
    plt.xlabel(r'$\sigma_8$')
    plt.show()
    return val_params, truefields, rchisqmat, ttrchisq, tgrchisq, ggrchisq


def perform_powspec_analysis_short(model_path, ckpnum, idx_fiducial=None, logfields=False, normalize=True, plot_chisqs=True):
    samppath = model_path.format(ckpnum)
    resdict = pickle.load(open(samppath, 'rb'))
    allsamps = resdict['samples'][0][:, 0]

    logstr = 'LOG' if logfields else 'LINEAR'
    normstr = 'WITH' if normalize else 'WITHOUT'
    print(f'Examine consistency of the pk of {logstr} fields {normstr} normalization')
    truefields, val_params, rchisqmat = evaluate_paramwise_reducedchisq(samppath, 50, data_subtype='validation', return_truefields=True, logfields=logfields, normalize=normalize)

    print('CHECKPOINT: ', ckpnum, 'MODEL: ', model_path)
    print('Performing Analysis for:')
    print(val_params)
    print('#### Rchisq Matrix: 50 Gen vs 15 True ########')
    print(rchisqmat.mean(axis=1), np.std(rchisqmat, ddof=1, axis=1)/np.sqrt(rchisqmat.shape[1]))
    print("##############")
    print('Ckp: {} || Mean|Std across params= {:.2f} {:.2f}'.format(ckpnum, rchisqmat.mean(), rchisqmat.std(ddof=1)))
    assert (abs((val_params[idx_fiducial, 0] - 0.30) + (val_params[idx_fiducial, 1] - 0.82))<0.05), "Fiducial Parameter isn't actually fiducial"
    print('Fiducial: ', rchisqmat.mean(axis=1)[idx_fiducial], rchisqmat.std(axis=1, ddof=1)[idx_fiducial]/np.sqrt(rchisqmat.shape[1]))
    gen50_true15 = {'pk_allparams': rchisqmat, 'pk_fiducial': rchisqmat[idx_fiducial]}
    print("##############")
    print('#### Leave one out: 15 Gen or True vs 14 other True #######')
    
    ttrchisq = []
    tgrchisq = []
    ttbias_pvalues = []
    tgbias_pvalues = []
    for ip in range(len(val_params)):
        print('+++++++++++++++')
        # True vs True
        Btrue = 15
        ipsubset = np.arange(ip*Btrue, (ip+1)*Btrue)
        rchisq_true = leave_one_out_chisq_z(truefields[ipsubset], truefields[ipsubset], logfields=logfields, normalize=normalize)
        rchisq_mean, rchisq_sem = get_mean_sem(rchisq_true)
        ttrchisq.append(rchisq_true)
        print(f'Params Om_m: {val_params[ip, 0]:.2f}, sig8: {val_params[ip, 1]:.2f}')
        print(f'A. True v True [Intrinsic Variability]')
        print(f'Mean RChisq: {rchisq_mean:.2f}, SEM: {rchisq_sem:.2f}')

        Bsamp = 50
        fields_samp = allsamps[np.arange(ip*Bsamp, (ip+1)*Bsamp)]
        samp15 = fields_samp[:15]
        rchisq_samp = leave_one_out_chisq_z(truefields[ipsubset], samp15, logfields=logfields, normalize=normalize)
        rchisq_mean, rchisq_sem = get_mean_sem(rchisq_samp)
        tgrchisq.append(rchisq_samp)
        print(f'B1. True v Gen [Similarity]')
        print(f'Mean RChisq: {rchisq_mean:.2f}, SEM: {rchisq_sem:.2f}')

        print('Another Subset of 15 generated fields')
        samp15 = fields_samp[15:30]
        rchisq_samp = leave_one_out_chisq_z(truefields[ipsubset], samp15, logfields=logfields, normalize=normalize)
        rchisq_mean, rchisq_sem = get_mean_sem(rchisq_samp)
        print(f'B2. True v Gen [Similarity]')
        print(f'Mean RChisq: {rchisq_samp.mean():.2f}, SEM: {rchisq_samp.std()/np.sqrt(samp15.shape[0]):.2f}')

        if logfields:
            means_true = truefields[ipsubset].mean(axis=(1, 2))
            means_gen = fields_samp.mean(axis=(1, 2))
            ttbias_pvalues.append(stats.ttest_ind(means_true[:8], means_true[8:], equal_var=False).pvalue)
            tgbias_pvalues.append(stats.ttest_ind(means_gen, means_true, equal_var=False).pvalue)

    gen15_true14 = {'pk_allparams': np.vstack(tgrchisq), 'pk_fiducial': tgrchisq[idx_fiducial]}
    true15_true14 = {'pk_allparams': np.vstack(ttrchisq), 'pk_fiducial': ttrchisq[idx_fiducial]}
    if logfields:
        true15_true14.update({'half_split_bias_pvalue': ttbias_pvalues, 'fiducial_bias_pvalue': ttbias_pvalues[idx_fiducial]})
        gen50_true15.update({'bias_pvalue': tgbias_pvalues, 'fiducial_bias_pvalue': tgbias_pvalues[idx_fiducial]})
    
    if plot_chisqs:
        plt.figure()
        plt.hist(np.array(np.hstack(ttrchisq)), bins=20, density=True, alpha=0.5, label='Ref: True[14] | True [15]', range=[0, 9])
        plt.legend()
        plt.show()
        plt.figure()
        plt.hist(np.array(np.hstack(tgrchisq)), bins=20, density=True, alpha=0.5, label='Ref: True[14] | Gen [15]', range=[0, 9])
        plt.legend()
        plt.show()
        plt.figure()
        plt.hist(rchisqmat.flatten(), bins=20, density=True, alpha=0.5, label='Ref: True[15] | Gen [50]', range=[0, 9])
        plt.legend()
        plt.show()

    return {'params': val_params, 'gen50_true15': gen50_true15, 'true_loo': true15_true14, 'gen_loo': gen15_true14}


def leave_one_out_chisq_z(fields, otherfields, return_z=False, logfields=True, normalize=True, percentile_std=False):
    """
    Returns the chisq computed in a leave one out fashion.
    """
    if logfields:
        kvals, pklist = get_powspec_for_samples([fields, otherfields], normalize=normalize)
    else:
        kvals, pklist = get_powspec_for_samples([10.0**fields, 10.0**otherfields], normalize=normalize)
    rchisqs = []
    zscores = []
    for i in range(min(fields.shape[0], otherfields.shape[0])):
        pkref = np.delete(pklist[0], i, axis=0)
        assert pkref.shape[0]==(fields.shape[0]-1)
        
        if percentile_std:
            std_est = (np.percentile(pkref, 84, axis=0)- np.percentile(pkref, 16, axis=0))/2
        else:
            std_est = np.std(pkref, ddof=1, axis=0)
        meantrue, invvar = np.mean(pkref, axis=0),  std_est** (-2)
        if return_z:
            zscores.append((pklist[1][i, :] - meantrue)/std_est)
        rchisqs.append((np.sum((pklist[1][i, :] - meantrue)**2 * invvar) / (len(kvals)-1)))
    
    if return_z:
        return np.array(rchisqs), np.array(zscores)
    else:
        return np.array(rchisqs)


def plot_ps_samples(kvals, samplist, names, cols=['b', 'r'], logscale=True, k2pk=False, savefig_dict={}, put_wavenumber_on_x=True, percentiles=False, which_percs=[16, 84], select_k=False):
    '''
    :param kvals:
    :param samplist: List of power spectra for samples (eg: either from different models or the real fields)
    :param names:
    :param cols:
    :param percentiles: If True use the 16th and 84th percentiles to create the envelope in each k bin. Else, std.
        Remember, this will cause errors if the mean is outside the two limits since the envelope cant be negative.
    :return:
    '''
    if put_wavenumber_on_x:
        kvals = kvals*2*np.pi/25.0
    plt.figure(figsize=savefig_dict['figsize'] if 'figsize' in savefig_dict.keys() else [6, 6])
    for isd, samp in enumerate(samplist):
        assert len(samp.shape)==2
        if k2pk:
            samp = samp*(kvals**2) #check this line
        meanps = np.mean(samp, axis=0)
        if percentiles:
            stdps_upp = np.percentile(samp, which_percs[1], axis=0) - meanps
            stdps_low = meanps - np.percentile(samp, which_percs[0], axis=0)
        else:
            stdps_upp = stdps_low = np.std(samp, axis=0, ddof=1)
        style='solid' if isd==0 else 'dashed'
        if select_k:
            k_idx = np.unique(np.arange(len(kvals))[np.logspace(0,np.log10(len(kvals)),50).astype(int)])
        else:
            k_idx = np.arange(len(kvals))
        plt.plot(kvals[k_idx], meanps[k_idx], c=cols[isd], label=names[isd], linestyle=style)
        plt.fill_between(kvals[k_idx], (meanps-stdps_low)[k_idx], (meanps+stdps_upp)[k_idx], alpha=0.2, facecolor=cols[isd], edgecolor=cols[isd])
    if logscale:
        plt.xscale('log')
        plt.yscale('log')
    if put_wavenumber_on_x:
        plt.xlabel(r'Wavenumber $k (h$ Mpc$^{-1}$)')
    else:
        plt.xlabel(r'k')
    if k2pk:
        plt.ylabel(r'$k^2P(k)$')
    else:
        plt.ylabel(r'$P(k)$')
    if 'title' in savefig_dict:
        plt.title(savefig_dict['title'])
    plt.legend()
    if 'save_path' in savefig_dict.keys():
        plt.savefig(savefig_dict['save_path'], dpi=savefig_dict['dpi'] if 'dpi' in savefig_dict else 100, bbox_inches='tight')
    plt.show()
    return


def get_powspec_for_samples(samplist, normalize=True):
    '''
    :param samplist: list of np arrays with shape N_img, Nx, Nx
    :param hist_kwargs: bins, range, density
    :return:
    '''
    ps_list = []
    Nx = samplist[0].shape[-1]
    kvals = np.arange(0, Nx/2)
    for ist, samp in enumerate(samplist):
        assert len(samp.shape)==3
        assert samp.shape[-1]==Nx
        assert samp.shape[-2]==Nx
        outputs = [utils.calc_1dps_img2d(kvals, samp[ci, ...], to_plot=False, smoothed=0.25, normalize=normalize) for ci in range(samp.shape[0])]
        if normalize:
            pssamp = np.vstack([e[1] for e in outputs])
            kvals_st = np.vstack([e[0] for e in outputs])
            assert len(np.unique(kvals_st, axis=0))==1
            kvals = kvals_st[0, :]
            if ist>0:
                assert np.allclose(kvals, kvals_st[0, :]), "Seem to be comparing dissimilar field sizes."
            ps_list.append(pssamp)
        else:
            ps_list.append(np.vstack(outputs))
    return kvals, ps_list


def get_pixel_histogram_for_samples(samplist, hist_kwargs, names, cols, with_err=True, savefig_dict={}):
    '''
    :param samplist: list of torch tensors with shape N_img, Nx, Nx
    :param hist_kwargs: bins, range, density
    :return:
    '''
    sampwise_histmean  = []
    sampwise_histstd = []
    for samp in samplist:
        hist_all = np.zeros((samp.shape[0], len(hist_kwargs['bins'])-1))
        for img in range(samp.shape[0]):
            vals = np.histogram(samp[img][:], **hist_kwargs)
            hist_all[img, :] = vals[0]
        sampwise_histmean.append(hist_all.mean(0))
        sampwise_histstd.append(np.std(hist_all, axis=0, ddof=1))
    #bins
    bins = hist_kwargs['bins']
    bins_low = bins[:-1]
    bins_upp = bins[1:]
    bins_mid = (bins_upp+bins_low)/2
    bins_width = bins_upp - bins_low
    plt.figure()
    for isa in range(len(samplist)):
        if with_err:
            plt.bar(bins_mid, sampwise_histmean[isa], yerr=sampwise_histstd[isa]/np.sqrt(len(samplist[0])),width=bins_width,
                 label=names[isa], color=cols[isa], alpha=0.2, ecolor=cols[isa])
        else:
            plt.bar(bins_mid, sampwise_histmean[isa], width=bins_width,
                 label=names[isa], color=cols[isa], alpha=0.2)
    plt.legend()
    plt.xlabel('Pixel intensity')
    if 'save_path' in savefig_dict.keys():
        plt.savefig(savefig_dict['save_path'], dpi=savefig_dict['dpi'] if 'dpi' in savefig_dict else 100, bbox_inches='tight')
    plt.show()
    return sampwise_histmean, sampwise_histstd

      
def plot_panel_images(images,titles, nrow, ncol, figsize, savefig_dict):
    fig, ax = plt.subplots(figsize=figsize, nrows = nrow, ncols=ncol)
    ax = ax.ravel()
    vmin, vmax = savefig_dict['vmin'] if 'vmin' in savefig_dict.keys() else None, savefig_dict['vmax'] if 'vmax' in savefig_dict.keys() else None
    for ii, img in enumerate(images):
        c = ax[ii].imshow(img, origin='lower', vmin=vmin, vmax=vmax)
        if 'no_colorbar' not in savefig_dict.keys():
            plt.colorbar(c, ax=ax[ii], fraction=0.05)
        if titles is not None:
            ax[ii].set_title(titles[ii])
        ax[ii].axis('off')
    if 'wspace' in savefig_dict.keys():
        fig.subplots_adjust(wspace=savefig_dict['wspace'])
    if 'save_path' in savefig_dict.keys():
        plt.savefig(savefig_dict['save_path'], dpi=savefig_dict['dpi'] if 'dpi' in savefig_dict else 100, bbox_inches='tight')
    plt.show()
    return

def plot_scatter_prediction_truth(params_true, mean_NN, std_NN=None, params= ['Om_m', 'Sig_8', 'A_SN1', 'A_AGN1', 'A_SN2', 'A_AGN2']):
    for ip, pnam in enumerate(params[:2]):
        plt.figure()
        if std_NN is None:
            plt.scatter(params_true[:, ip], mean_NN[:, ip], color='b', s=3)
        else:
            plt.errorbar(params_true[:, ip], mean_NN[:, ip], yerr=std_NN[:, ip], fmt="o", color='b', markersize=3, elinewidth=1)
        plt.plot(params_true[:, ip], params_true[:, ip], c='k')
        plt.xlabel(pnam+': True')
        plt.ylabel(pnam+': Predicted')
        plt.show()
    return
        
        
### CONDITIONAL
class Preprocessed_data_for_param_evaluation():
    def __init__(self, data, verbose, device='cpu', meanstd_norm=None, time=None, diffusion=None, labels_subset=np.arange(6)):
        '''
        Class that takes input fields and processes them using the transform specified.
        Currently written for a single channel. Only handles a single tuple as input data. Doesn't do any rotating or flipping.
        Args:
            :param data: Tuple of Arrays [Fields, Params] ASSUMES you're working with the Log Fields
            Eg: [Samples from DM, Input params]
            :param meanstd_norm: [mean, std] to norm the fields by.
            :param verbose: Print statements.
            :param time: Timestep (index) corresponding to the Noise level to be added
            If time<=0 then the image is returned as if with no noise added
            Else: for all t\in [1, T], a noise level corresponding to beta[t-1] is added
            :param diffusion: Mapping from time to the variance of the noise level to be added
        '''
        fields, params = data
        self.noised=False
        if time is not None:
            self.noised=True
        
        self.size, C, H, W = fields.shape
        assert C==1
        assert params.shape[0]==self.size
        assert params.shape[1]==6
        params = params[:, labels_subset]

        # normalize params
        minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])[labels_subset]
        maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])[labels_subset]
        params = (params - minimum) / (maximum - minimum)

        print('%.3f < F(all|resc)  < %.3f' % (np.min(fields), np.max(fields)))
        #NOT removing mean
        # retrieve mean, std and standard scale fields
        mean, std = meanstd_norm
        print('Mean-norm={%.3f}, Std-norm={%.3f}' % (mean, std))

        #Standard Scaling
        fields = (fields - mean) / std 
        if verbose:
            print('%.3f < F(all|norm) < %.3f' % (np.min(fields), np.max(fields)))
        
        xt = torch.tensor(fields, dtype=torch.float32, device='cpu')
        
        if self.noised: # if you want to add noise
            self.time = torch.full((len(fields),), int(time), dtype=int)
            if time>0:
                xt = diffusion.q_sample_incl_t0(x_start=xt, t=self.time, noise=None)
        
        self.x = xt.to(device)
        self.y = torch.tensor(params, dtype=torch.float32, device=device)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.noised:
            return self.x[idx], self.time[idx], self.y[idx]
        else:
            return self.x[idx], self.y[idx]

    
def get_results_model(model, device, test_loader):
    '''
    Get the parameter predictions given a classifier model. 
    Args:
        model: classifier. This accommodates both time dependent and time independent models.
        test_loader: (Images, t: optional, Labels)
    Returns:
        dataset: N_images x 18. dataset[:, :6]: True params | dataset[:, 6:12]: Mean Prediction NN | dataset[:, 12:] Sigma Prediction NN
        Norm_error: RMSE of the predictions relative to the target in the transformed space.
    '''
    ####### Preloaded model ########
    # get the number of maps in the test set
    num_maps = len(test_loader.dataset)
    print('\nNumber of maps in the test set: %d' % num_maps)

    # define the arrays containing the value of the parameters
    params_true = np.zeros((num_maps, 6), dtype=np.float32)
    params_NN = np.zeros((num_maps, 6), dtype=np.float32)
    errors_NN = np.zeros((num_maps, 6), dtype=np.float32)

    # get test loss: TODO: What purpose does this snippet serve in their code?
    g = [0, 1, 2, 3, 4, 5]
    test_loss1, test_loss2 = torch.zeros(len(g)).to(device), torch.zeros(len(g)).to(device)
    test_loss, points = 0.0, 0
    model.eval()
    
    one_bs = next(iter(test_loader))
    if len(one_bs)==3:
        TIME_DEP = True
    else:
        assert len(one_bs)==2
        TIME_DEP = False
    print(TIME_DEP)

    for input_tup in test_loader:     
        with torch.no_grad():
            if TIME_DEP:
                x, tx, y = input_tup
                mod_input = x, tx 
            else:
                x, y = input_tup
                mod_input = [x]
            
            bs = x.shape[0]  # batch size
            p = model(*mod_input)  # prediction for mean and variance
            y_NN = p[:, :6]  # prediction for mean
            e_NN = p[:, 6:]  # prediction for error
            loss1 = torch.mean((y_NN[:, g] - y[:, g]) ** 2, axis=0)
            loss2 = torch.mean(((y_NN[:, g] - y[:, g]) ** 2 - e_NN[:, g] ** 2) ** 2, axis=0)
            test_loss1 += loss1 * bs
            test_loss2 += loss2 * bs

            # save results to their corresponding arrays
            params_true[points:points + x.shape[0]] = y.cpu().numpy()
            params_NN[points:points + x.shape[0]] = y_NN.cpu().numpy()
            errors_NN[points:points + x.shape[0]] = e_NN.cpu().numpy()
            points += x.shape[0]
    test_loss = torch.log(test_loss1 / points) + torch.log(test_loss2 / points)
    test_loss = torch.mean(test_loss).item()
    print('Test loss = %.3e\n' % test_loss)

    Norm_error = np.sqrt(
        np.mean((params_true - params_NN) ** 2, axis=0))  # RMSE before rescaling to minmax of param range
    print('Normalized Error Omega_m = %.3f' % Norm_error[0])
    print('Normalized Error sigma_8 = %.3f' % Norm_error[1])
    print('Normalized Error A_SN1   = %.3f' % Norm_error[2])
    print('Normalized Error A_AGN1  = %.3f' % Norm_error[3])
    print('Normalized Error A_SN2   = %.3f' % Norm_error[4])
    print('Normalized Error A_AGN2  = %.3f\n' % Norm_error[5])

    # de-normalize
    minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
    maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
    params_true = params_true * (maximum - minimum) + minimum
    params_NN = params_NN * (maximum - minimum) + minimum
    errors_NN = errors_NN * (maximum - minimum)

    error = np.sqrt(np.mean((params_true - params_NN) ** 2, axis=0))  # RMSE
    print('Error Omega_m = %.3f' % error[0])
    print('Error sigma_8 = %.3f' % error[1])
    print('Error A_SN1   = %.3f' % error[2])
    print('Error A_AGN1  = %.3f' % error[3])
    print('Error A_SN2   = %.3f' % error[4])
    print('Error A_AGN2  = %.3f\n' % error[5])

    mean_error = np.absolute(
        np.mean(errors_NN, axis=0))  # |<std_NN>| Q: Why should an absolute even be required? Independent of accuracy
    print('Bayesian error Omega_m = %.3f' % mean_error[0])
    print('Bayesian error sigma_8 = %.3f' % mean_error[1])
    print('Bayesian error A_SN1   = %.3f' % mean_error[2])
    print('Bayesian error A_AGN1  = %.3f' % mean_error[3])
    print('Bayesian error A_SN2   = %.3f' % mean_error[4])
    print('Bayesian error A_AGN2  = %.3f\n' % mean_error[5])

    rel_error = np.sqrt(np.mean((params_true - params_NN) ** 2 / params_true ** 2, axis=0))  # sqrt(<z^2>): Z score
    print('Relative error Omega_m = %.3f' % rel_error[0])
    print('Relative error sigma_8 = %.3f' % rel_error[1])
    print('Relative error A_SN1   = %.3f' % rel_error[2])
    print('Relative error A_AGN1  = %.3f' % rel_error[3])
    print('Relative error A_SN2   = %.3f' % rel_error[4])
    print('Relative error A_AGN2  = %.3f\n' % rel_error[5])

    rel_error = 100*np.mean(errors_NN/params_NN, axis=0)  # 
    print('PNAS "Accuracy", (Precision) Omega_m = %.3f' % rel_error[0])
    print('PNAS "Accuracy", (Precision) sigma_8 = %.3f' % rel_error[1])
    
    dataset = np.zeros((num_maps, 18), dtype=np.float32)
    dataset[:, :6] = params_true
    dataset[:, 6:12] = params_NN
    dataset[:, 12:] = errors_NN # convert variance to std
    return dataset, Norm_error


def get_train_cosmo_params(Bsize=20, path='/n/holylfs06/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/data_processed/params_IllustrisTNG.txt', seed=0, replace=False):
    #To make this compatible with existing code, use replace = True
    params = np.loadtxt(path)
    train_upplim = int(len(params)*0.7)
    rng = np.random.default_rng(seed)
    idx= rng.choice(train_upplim, Bsize, replace=replace)
    params_actual = params[:train_upplim, :][idx]
    return params_actual.astype(np.float32)



def get_train_cosmo_params_in_range(param_range, labels_subset, path='../data_processed/params_IllustrisTNG.txt', get_indices=False):
    params = np.loadtxt(path)
    train_upplim = int(len(params)*0.7)
    params_actual = params[:train_upplim, :]
    mask = np.all(params_actual[:, labels_subset]>=param_range['min'], axis=1) * np.all(params_actual[:, labels_subset]<=param_range['max'], axis=1)
    print('Incl %age', np.sum(mask)/len(mask))
    if get_indices:
        return params_actual.astype(np.float32)[mask, :], np.arange(len(params))[:train_upplim][mask]
    return params_actual.astype(np.float32)[mask, :]



def get_validation_cosmo_params(Bsize=20, valpath='/n/holylfs06/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/data_processed/LogMaps_Mcdm_IllustrisTNG_LH_z=0.00_Nx256_val.npz', seed=0, replace=False):
    valmmap = np.load(valpath, mmap_mode='r')
    valbatch = valmmap['params']
    rng = np.random.default_rng(seed)
    idx= rng.choice(len(valbatch), Bsize, replace=replace)
    params_actual = valbatch[idx]
    return params_actual

def get_simba_cosmo_params(Bsize=20, path='/n/holylfs06/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/data_processed/LogMaps_Mcdm_SIMBA_LH_z=0.00_Nx256.npz', seed=0, replace=False):
    mmap = np.load(path, mmap_mode='r')
    params = mmap['params']
    rng = np.random.default_rng(seed)
    idx= rng.choice(len(params), Bsize, replace=replace)
    params_actual = params[idx]
    return params_actual

def save_samples_from_checkpoints(params_actual, sdpaths, seed=2, labels_subset=np.array([0, 1]),
                        device='cpu', noise_input=None, Nx=256, savepath=None, cond_kwargs=None, return_reverse_indices=None, get_transform=False, use_ema=False, ddp=False):
    '''
    :param params_actual: Nfieldsx6. Np array like the output of the above funcs.
    :param sdpaths: List of checkpoint files
    :param seed:
    :param labels_subset: Param Mask to be used for conditioning
    :param device:
    :param noise_input: Noise Input: 2000, 2, 1, Nx, Nx or None
    :param Nx: 256
    :param savepath: Pickle to save dictionary in or None
    :param return_reverse_indices: Subset of samples to return 
    :return: resdict: Dictionary with params, models, labels_subset, samples
    '''
    if get_transform:
        dirfiles = np.array([sdpath[:sdpath.rindex('/')+1] for sdpath in sdpaths])
        assert len(np.unique(dirfiles))==1
        dirname = str(dirfiles[0])
        print(dirname, type(dirname))
        tr, invtr = retrieve_data_transforms(dirname)

    else:
        #Get image transforms
        print('Using default minmax transform pegged to Nx=256 train set')
        fields256 = np.load('../data_processed/LogMaps_Mcdm_IllustrisTNG_LH_z=0.00_Nx256_train.npy')
        RANGEMIN, RANGEMAX = np.min(fields256), np.max(fields256)
        tr, invtr = hf_diffusion.get_minmax_transform(torch.tensor(RANGEMIN), torch.tensor(RANGEMAX))
        del fields256

    torch.manual_seed(seed)
    np.random.seed(seed)
    #normalized, tensor params
    cond_params = torch.tensor(hf_diffusion.normalize(params_actual[:,labels_subset], labels_subset), device=device)
    ckpsamps = []
    resdict = {'params': params_actual, 'models': sdpaths, 'labels_subset': labels_subset}
    for sdpath in sdpaths:
        if return_reverse_indices is not None:
            samps = utils.get_samples_given_saved_dict(sdpath, cond_params.shape[0], cond_params, device=device,
                                                       sample_image_size=Nx, return_multiple_timesteps=True, noise_input=noise_input, cond_kwargs=cond_kwargs, return_reverse_indices=return_reverse_indices, use_ema=use_ema, ddp=ddp)
        else: 
            samps = utils.get_samples_given_saved_dict(sdpath, cond_params.shape[0], cond_params, device=device,
                                                       sample_image_size=Nx, return_multiple_timesteps=False, noise_input=noise_input, cond_kwargs=cond_kwargs, use_ema=use_ema, ddp=ddp)
            samps = invtr(torch.from_numpy(samps))
        ckpsamps.append(samps)
    resdict.update({'samples': ckpsamps})
    if savepath is not None:
        pickle.dump(resdict, open(savepath, 'wb'))
    return resdict

def plot_param_inference_network_predictions(resdict, model, device='cpu', save_config=None, normkey='Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy'):
    '''
    Plots the parameters predicted by the parameter inference networks on the y axis against the true parameters.
    :param resdict: Output of save_samples_from_checkpoints. Keys:
        params: True params
        samples: List of Fields to pass to the model. Log fields [or Log 1+Mstar ] but NOT already normalized.
         samples.shape[0] == params.shape[0]
         len(shape)==4
        models: Generative Models. len(samples) == len(models).
    :param model: Param inference network.
    :param device:
    :return: List of (p_true, mean_NN_prediction, sigma_NN_prediction) of len(models)
    '''
    params_actual, field_samples = resdict['params'], resdict['samples']
    
    resultlist = []
    assert len(resdict['samples']) == len(resdict['models'])
    for imod, samples in enumerate(field_samples):
        print('Results for Samples from ', resdict['models'][imod])
        testdata = [samples.data.cpu().numpy(), params_actual]
        normdict = pickle.load(
            open('../diffusion-models-for-cosmological-fields/annotated/results/misc/IllustrisTNGLH_norms.pkl', 'rb'))
        mapnorm = normdict[normkey]
        sampsdata = Preprocessed_data_for_param_evaluation(testdata, verbose=True,
                        device=device, meanstd_norm=[mapnorm['mean'], mapnorm['std']], labels_subset=np.arange(6))
        shuffle = False # does this need to be true?
        batch_size = len(sampsdata)
        testloader = DataLoader(dataset=sampsdata, batch_size=batch_size, shuffle=shuffle)
        results = get_results_model(model, device, testloader)
        params_data = results[0]
        params_true, mean_NN, std_NN = params_data[:, :6], params_data[:, 6:12], params_data[:, 12:]
        params = [r'$\Omega_m$', r'$\sigma_8$', 'A_SN1', 'A_AGN1', 'A_SN2', 'A_AGN2']

        for ip, pnam in enumerate(params[:2]):
            plt.figure()
            plt.errorbar(params_true[:, ip], mean_NN[:, ip], yerr=std_NN[:, ip], fmt="o", color='b', markersize=3, elinewidth=1)
            plt.plot(params_true[:, ip], params_true[:, ip], c='k')
            plt.xlabel(pnam+': True')
            plt.ylabel(pnam+': Predicted')
            if save_config:
                plt.savefig(save_config['save_path'][imod]+f'p_{ip}.pdf', dpi=save_config['dpi'] if 'dpi' in save_config else 100, bbox_inches='tight')
            plt.show()
        print('#####################################################')
        resultlist.append((params_true, mean_NN, std_NN))
    return resultlist



def get_truefields_for_sampled_fields(resdict, type='train', custom_file=None):
    '''
    :param resdict: ASSUMES ANY repeated parameters are contiguous!!!
    :param type: 'train', 'validation', 'custom', 'simba'
    :param custom_file:
    Returns:
        params_sampled: The parameters corresponding to the sampled fields in resdict.
        truefields: min(15, count_per_param): True fields corresponding to the parameters in resdict.
        Shape[0] of params_sampled = Shape[0] of truefields ONLY when the fields per parameter are all <=15 in resdict.
    '''
    # Retrieve train fields for the parameters that were saved
    #TODO: CLean this up and move into a dataset
    #TODO: Repackage all files into train,val, test npz files!!!!
    DATADIR = '/n/holylfs06/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/data_processed/'
    if type=='train':
        assert custom_file is None
        paramfile = os.path.join(DATADIR, 'params_IllustrisTNG.txt')
        fieldfile = os.path.join(DATADIR, 'LogMaps_Mcdm_IllustrisTNG_LH_z=0.00_Nx256_train.npy')
        params_all = np.loadtxt(paramfile)
        train_fields = np.load(fieldfile)
    elif type=='validation':
        paramfile = fieldfile = os.path.join(DATADIR, 'LogMaps_Mcdm_IllustrisTNG_LH_z=0.00_Nx256_val.npz')
        valfiledict = np.load(fieldfile)
        params_all = valfiledict['params'] #so you dont have to handle the trainfields offset in idx_field_in_ffile
        train_fields = valfiledict['fields']
    elif type=='custom':
        fieldfile = custom_file
        paramfile = os.path.join(DATADIR, 'params_IllustrisTNG.txt')
        if custom_file.endswith('.npy'):
            params_all = np.loadtxt(paramfile)
            train_fields = np.load(fieldfile)
        else:
            assert custom_file.endswith('.npz')
            fdict = np.load(custom_file)
            params_all = fdict['params']
            train_fields = fdict['fields']
    elif type=='simba':
        paramfile = fieldfile = os.path.join(DATADIR, 'LogMaps_Mcdm_SIMBA_LH_z=0.00_Nx256.npz')
        filedict = np.load(fieldfile)
        params_all = filedict['params'] #you dont have to handle the trainfields offset in idx_field_in_ffile
        train_fields = filedict['fields']
    else:
        raise NotImplementedError

    print(f'Loading true fields from {fieldfile}. Loading params from {paramfile}.')
    print('Length of field file = {}. Length of param file = {}'.format(len(train_fields), len(params_all)))
    if len(train_fields)!=15*len(params_all):
        print('!!WARNING!!: Length of train_fields does not match length of params_all')

    # get resdict params
    uniparams, idx_unip, count_unip = np.unique(resdict['params'], axis=0, return_index=True, return_counts=True)
    rearrange_to_occorder = np.argsort(idx_unip)
    uniparams, idx_unip, count_unip = uniparams[rearrange_to_occorder], idx_unip[rearrange_to_occorder], count_unip[rearrange_to_occorder]

    # ip: param index in resdict['params']
    paramlist = []
    truefieldlist = []
    for param_no, ip in enumerate(idx_unip):
        count_for_param = count_unip[param_no]
        genparam = resdict['params'][ip]
        idx_p_in_pfile = np.argmin(np.abs(params_all[:, 0] - genparam[0]))
        assert np.allclose(params_all[idx_p_in_pfile], genparam)
        print(f'Parameter{ip}:', genparam, params_all[idx_p_in_pfile], uniparams[param_no], count_for_param)
        idx_field_in_ffile = np.arange(idx_p_in_pfile*15, (idx_p_in_pfile + 1) * 15).astype(int)
        print(idx_field_in_ffile, train_fields.dtype, count_for_param)
        truefields = train_fields[idx_field_in_ffile][:count_for_param]
        paramlist.append(np.vstack([genparam]*count_for_param))
        truefieldlist.append(truefields)
    params_sampled = np.vstack(paramlist)
    truefields_all = np.vstack(truefieldlist)
    assert np.all(np.equal(params_sampled, resdict['params']))
    #Warning: this func assumes all repeated params are contiguous. If not, it'll return the wrong params_sampled and break at the above assertion.
    del train_fields
    return params_sampled, truefields_all



def compare_true_sampled_powerspectra(resdict, names_custom=None, type='train'):
    '''
    :param resdict: Output of save_samples_from_checkpoints
    :return:
    '''
    '''#Retrieve train fields for the parameters that were saved
    fieldfile, paramfile = '../data_processed/LogMaps_Mcdm_IllustrisTNG_LH_z=0.00_Nx256_train.npy', '../data_processed/params_IllustrisTNG.txt'
    train_fields = np.load(fieldfile)
    params_all = np.loadtxt(paramfile)
    print(f'Loading train fields from {fieldfile}. Loading params from {paramfile}.')

    #get resdict params
    uniparams, idx_unip, idx_where, count_unip = np.unique(resdict['params'], axis=0, return_index=True, return_inverse=True, return_counts=True)'''
    params_true, truefields = get_truefields_for_sampled_fields(resdict, type)
    uniparams, idx_unip, count_unip = np.unique(resdict['params'], axis=0, return_index=True, return_counts=True)
    rearrange_to_occorder = np.argsort(idx_unip)
    uniparams, idx_unip, count_unip = uniparams[rearrange_to_occorder], idx_unip[rearrange_to_occorder], count_unip[
        rearrange_to_occorder]

    names = names_custom if names_custom is not None else ['True']+resdict['models']
    genfieldlist = [samp[:, 0] for samp in resdict['samples']]

    assert np.all(np.array([truefields.shape == samps.shape for samps in genfieldlist]))

    for param_no, posn in enumerate(idx_unip): #ip: param index in resdict['params']
        count = count_unip[param_no]
        genparam = resdict['params'][posn]
        print('Parameter', genparam, params_true[posn], count)
        samprange = np.arange(posn, posn+count)
        print(samprange)
        allfields = [truefields[samprange]] + [gen[samprange] for gen in genfieldlist]
        print(allfields[0].shape)
        kvals, pklist = get_powspec_for_samples(allfields)
        plot_ps_samples(kvals, pklist, names, cols=['k', 'b', 'r', 'g', 'y'], k2pk=True)
        print('#########')
    return


def compare_true_sampled_pixelhist(resdict, tuple_params, type='train'):
    '''
    :param resdict: Output of save_samples_from_checkpoints
    :param tuple_params: Which params in the order of uniparams should you compare
    :return:
    '''
    params_true, truefields = get_truefields_for_sampled_fields(resdict, type)
    genfieldlist = [samp[:, 0] for samp in resdict['samples']]
    allfields = [truefields] + genfieldlist
    names = ['True'] + [modname.split('/')[-1].split('.')[0] for modname in resdict['models']]

    uniparams, idx_unip, count_unip = np.unique(resdict['params'], axis=0, return_index=True, return_counts=True)
    rearrange_to_occorder = np.argsort(idx_unip)
    uniparams, idx_unip, count_unip = uniparams[rearrange_to_occorder], idx_unip[rearrange_to_occorder], count_unip[
        rearrange_to_occorder]
    ip1, ip2 = tuple_params
    param1, param2 = uniparams[ip1], uniparams[ip2]
    posn1, posn2 = idx_unip[ip1], idx_unip[ip2]
    count_unip1, count_unip2 = count_unip[ip1], count_unip[ip2]
    samprange1, samprange2 = np.arange(posn1, posn1+count_unip1), np.arange(posn2, posn2+count_unip2)
    print(names)
    for imod, model in enumerate(names):
        print('For model', imod, model)
        plt.figure()
        paramrange1, paramrange2 = params_true[samprange1], params_true[samprange2]

        assert len(np.unique(paramrange1, axis=0))==1
        assert len(np.unique(paramrange2, axis=0)) == 1
        assert np.all(np.equal(paramrange1[0], param1))
        assert np.all(np.equal(paramrange2[0], param2))

        fields1, fields2 = allfields[imod][samprange1], allfields[imod][samprange2]
        plt.figure()
        for ind, i in enumerate(range(fields1.shape[0])):
            plt.hist(fields1[i].flatten(), bins=100, alpha=0.2, color='b',
                     label=None if ind != 0 else r'$\Omega_m={:.2f}, \sigma_8={:.2f}$'.format(param1[0], param1[1]))
            plt.hist(fields2[i].flatten(), bins=100, alpha=0.2, color='r',
                     label=None if ind != 0 else r'$\Omega_m={:.2f}, \sigma_8={:.2f}$'.format(param2[0], param2[1]))
        plt.legend()
        plt.title(names[imod])
        plt.show()
        print('#########')
    return

def retrieve_data_transforms(dirname):
    '''
    :param dirname: Looks into a results directory and retrieves the data transforms used.
    :return: Currently not backwards compatible with Nx=64 runs that are not using the pickled minmax transforms.
    '''
    allfiles = os.listdir(dirname)
    yamlfiles = []
    for f in allfiles:
        if f.endswith('.yaml'):
            yamlfiles.append(f)
    assert len(yamlfiles)==1
    yamlfile = os.path.join(dirname, yamlfiles[0])
    print('Getting transforms from ', yamlfile)

    with open(yamlfile, 'r') as stream:
        config_dict = yaml.safe_load(stream)
    print(config_dict['data']['transforms'])
    '''
    Currently no longer backwards compatible with the standardscale runs. Uncomment if you want that.
    #correct path: Make sure this works with the older minmax transform AND with the more recent one AND doesnt botch up the abspath stuff.
    if ('normfile' in config_dict['data']) and not (config_dict['data']['normfile'].startswith('/n/')): #For the newer minmax transforms pegged to the full dataset
        config_dict['data']['normfile'] = '../diffusion-models-for-cosmological-fields/annotated/'+ config_dict['data']['normfile']
    
    '''
    
    if ('normfile' not in config_dict['data']) and ((config_dict['data']['transforms'] == 'minmax') or (config_dict['data']['transforms'] == 'minmax+randfliprot')):
        # When you don't have normfile, peg to minmax of the Nx=256 train set
        print('Using default minmax transform pegged to Nx=256 train set')
        fields256 = np.load('../data_processed/LogMaps_Mcdm_IllustrisTNG_LH_z=0.00_Nx256_train.npy')
        RANGEMIN, RANGEMAX = np.min(fields256), np.max(fields256)
        tr, invtr = hf_diffusion.get_minmax_transform(torch.tensor(RANGEMIN), torch.tensor(RANGEMAX))
        del fields256
    else:
        tr, invtr = main_helper.get_data_transforms(None, config_dict)
    return tr, invtr

# Grid-Based likelihood estimate functions

def compute_conditional_likelihood_for_fields(fields, params_eval, model, vlbt_seed, diff=None, vlb_timesteps=None, reseed_over_timesteps=True, no_grad=False):
    '''
    This batches over the fields for a SINGLE parameter.
    :param fields: Tensor. NfieldsxCxHxW, ALREADY normalized
    :param params_eval: (N_eval=1)x2. SINGLE value of params_eval, ALREADY normalized.
    :param model:
    :param diff: Only if you're using the VLB.
    :param vlb_timesteps: Get single timestep VLB term.
    :return: Evaluates the Conditional Negative Log Likelihood/Nx^2 of a
     set of input fields relative to a given parameter params-eval
     #Tensor of either Nfields or NfieldsxT
    '''

    assert fields.mean()<5, "Recheck input field normalization"
    if reseed_over_timesteps:
        print('Reseeding over timesteps')
        rng = np.random.default_rng(seed=vlbt_seed)
        seed_per_timestep = rng.choice(2000, len(vlb_timesteps), replace=False)
        print('Seeds for t', seed_per_timestep)
    class CondModel(nn.Module):
        def __init__(self, model, params_fixed):
            print('Conditional Model for param =', params_fixed)
            super().__init__()
            assert len(params_fixed.shape) == 1
            self.model = model
            self.params = params_fixed

        def forward(self, x, t):
            labels = torch.vstack([self.params] * x.shape[0])
            return model(x, t, labels)

    condlikmodel = CondModel(model, params_eval)
    if vlb_timesteps is None:
        likvals = diff.compute_vlb(condlikmodel, fields)
    else:
        likvals = torch.zeros((len(vlb_timesteps), fields.shape[0]))
        for idxt, t in enumerate(vlb_timesteps):
            idxt_seed = seed_per_timestep[idxt] if reseed_over_timesteps else vlbt_seed
            #TODO: Batch over timesteps instead of looping to speed up. Add a seed.
            # the assignment is probably what broke it.
            likvals[idxt, :] = diff.get_single_vlb_term(condlikmodel, fields, t_index=t, seed=idxt_seed, no_grad=no_grad) # what is the difference between timesteps and t_index 
        likvals = likvals.transpose(0, 1)
    return likvals

def compute_conditional_likelihood_for_fields_fast(fields, params_eval, model, diff, vlbt_seed, use_sde=False, likfunc=None, vlb_timesteps=None, reseed_over_timesteps=False):
    '''
    This batches over the parameters at which you're evaluating the conditional likelihoods. Here, the condmodel is an ensemble of |N_eval| models and you get the conditional likelihoods for N_eval parameter points in a single pass.
    :param fields: Tensor. (Nfields=1)xCxHxW, ALREADY normalized. Single field. Nfields=1
    :param params_eval: N_evalx2 of params_eval, ALREADY normalized.
    :param model:
    :param diff: Only if you're using the VLB.
    :param vlb_timesteps: Get single timestep VLB term.
    :param reseed_over_timesteps: Each timestep gets a different seed even though the different p_eval gridpoints get the same seed.
    :return: Evaluates the Conditional Negative Log Likelihood/Nx^2 of a
     set of input fields relative to a set of given parameters params-eval
     #Tensor of either Neval or NevalxT
    '''
    assert fields.shape[0]==1, "Must be a single sample"
    assert fields.mean()<5, "Recheck input field normalization"
    class CondModel(nn.Module):
        def __init__(self, model, params_fixed):
            #print('Conditional Model for param =', params_fixed)
            super().__init__()
            assert len(params_fixed.shape) == 2
            self.model = model
            self.params = params_fixed.to(torch.float32)

        def forward(self, x, t):
            assert x.shape[0]==1, "Must be a single sample"
            xbatch = x.expand(self.params.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(torch.float32)
            tbatch = t.expand(self.params.shape[0])
            return model(xbatch, tbatch, self.params)
    
    if reseed_over_timesteps:
        print('Reseeding over timesteps')
        rng = np.random.default_rng(seed=vlbt_seed)
        seed_per_timestep = rng.choice(2000, len(vlb_timesteps), replace=False)
        print('Seeds for t', seed_per_timestep)
        
    condlikmodel = CondModel(model, params_eval)
    if use_sde:
        t0 = time.time()
        likvals = likfunc(condlikmodel, fields)
        t1 = time.time()
    else:
        t0 = time.time()
        if vlb_timesteps is None:
            likvals = diff.compute_vlb(condlikmodel, fields)
        else:
            likvals = torch.zeros((len(vlb_timesteps), params_eval.shape[0]))
            for idxt, t in enumerate(vlb_timesteps):
                idxt_seed = seed_per_timestep[idxt] if reseed_over_timesteps else vlbt_seed
                likvals[idxt, :] = diff.get_single_vlb_term(condlikmodel, fields, t_index=t, seed=idxt_seed)
            likvals = likvals.transpose(0, 1)
        t1 = time.time()
    return likvals


def get_fields_for_params_and_compute_likelihoods(params_fields, params_eval, sdpath, NSAMP, data_subtype='train', labels_subset=np.array([0, 1]), device='cpu', use_ema=False, ddp=False, vlb_timesteps=None, use_nx64=False, batch_over_params=True, seed=26, N_EvalBatches=2, vlbt_seed=2, reseed_over_t=False, perturbation=None, perturbation_kwargs=None):
    '''
    1. Retrieves field and param transforms.
    2. Gets model & diff
    3. For each θ_true (param_fields)
        3.1. Retrieve NSAMP fields + Tr
        3.2. -> compute_likelihood_for_fields_batchpoints_debug()
    Only setup to work with vlb_steps is NOT none. Usually gets all samples of a particular θ_true as input.
    :param fields_input: NORMALIZED fields you want the lik evaluation of. (Fields of θ_true)
    :param params_eval_normed: NORMALIZED params conditional on which you want to evaluate likelihoods p(x|θ_eval)
    :param model: diff model
    :return: Returns the likelihood of TRAIN distribution fields evaluated by a trained diffusion model for different parameters.
    Shape: Neval x Ntrue x Nsamps x T
    '''
    print("Get fields reseed param", reseed_over_t)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('Transform seed', seed)
    assert NSAMP<= 15
    assert vlb_timesteps is not None
    dirname = sdpath[:sdpath.rindex('/') + 1]
    print('Dirname: ', dirname)
    tr, _ = retrieve_data_transforms(dirname)

    # setup labels transforms
    params_minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5], dtype=np.float32)[labels_subset]
    params_maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0], dtype=np.float32)[labels_subset]

    params_eval_normed = (params_eval[:, labels_subset] - params_minimum)/(params_maximum - params_minimum)
    params_eval_normed = torch.tensor(params_eval_normed).to(device)

    # load model and losses
    sdict = torch.load(sdpath, map_location=device)

    if sdict['model_type'] == 'latentdiffusionmodule':
        raise NotImplementedError('This would NOT account for the encoder / decoder likelihood contribution.')
    else:
        model, diff = utils.load_model(sdpath, device, use_ema, ddp)
    print('ckp')

    if vlb_timesteps is None:
        cond_likelihoods = np.zeros((len(params_eval), len(params_fields), NSAMP))
    else:
        cond_likelihoods = np.zeros((len(params_eval), len(params_fields), NSAMP, len(vlb_timesteps)))

    for idxptrue in range(len(params_fields)): # loop over p_true
        params_sub = np.vstack([params_fields[idxptrue]] * NSAMP)
        if use_nx64:
            _, truefields = get_truefields_for_sampled_fields({'params': params_sub}, type=data_subtype)
        else:
            _, truefields = get_truefields_for_sampled_fields({'params': params_sub}, type=data_subtype)
        
        # Add some perturbation
        if perturbation:
            truefields = perturbation(truefields, **perturbation_kwargs)
        fields_input = tr(torch.tensor(truefields).unsqueeze(1).to(device))
        print('Input field top level', truefields.shape, fields_input.shape, truefields[0, 0, 0], fields_input[0, 0, 0, 0])
        #call the main compute likelihood func
        cond_likelihoods[:, idxptrue, ...] = compute_likelihood_for_fields_batchgrid_debug(fields_input, params_eval_normed, model, diff, vlb_timesteps, N_EvalBatches, vlbt_seed, reseed_over_t = reseed_over_t) #N_eval x Nfields x T
    return cond_likelihoods #Neval x Ntrue x Nsamps x T


def compute_likelihood_for_fields_batchgrid_debug(fields_input, params_eval_normed, model, diff, vlb_timesteps=None, N_EvalBatches=3, vlbt_seed=20, reseed_over_t = False):
    '''
    Arguments:
        :param fields_input: NORMALIZED fields you want the lik evaluation of. (Fields of θ_true) Shape=4, Nfields x 1 x Nx x Nx
        :param params_eval_normed: NORMALIZED params conditional on which you want to evaluate likelihoods p(x|θ_eval)
        :param model: diff model
    Returns the likelihood of fields_input evaluated by a trained diffusion model for different parameters.
    Shape: len(params_eval_normed), len(fields_input), len(vlb_timesteps)

    1. For each inputfield (loop over fields) 
        1.1. For each pevalsub (loop over subset of pevalgrid)
            1.1.1. Set seed
            1.1.2. -> compute_conditional_likelihood_for_fields_fast(inputfield, pevalsub...)
    Only setup to work with vlb_steps is NOT none. Usually gets all samples of a particular θ_true as input.
    
    '''
    print('Batchgrid debug reseed', reseed_over_t)
    assert vlb_timesteps is not None
    
    pevalbatched = np.array_split(params_eval_normed, N_EvalBatches, axis=0)
    cond_likelihoods = np.zeros((len(params_eval_normed), len(fields_input), len(vlb_timesteps)))

    for idf in range(fields_input.shape[0]):
        inputfield = fields_input[idf].unsqueeze(0)
        likres_batched = []
        for ip, pevalsub in enumerate(pevalbatched):
            likres_batched.append(compute_conditional_likelihood_for_fields_fast(inputfield, pevalsub, model, diff, vlbt_seed, use_sde=False, likfunc=None, vlb_timesteps=vlb_timesteps, reseed_over_timesteps=reseed_over_t)) #N_evalsubxT
        cond_likelihoods[:, idf, ...] = np.vstack(likres_batched) #N_evalxT
    return cond_likelihoods #Neval x NFields x T


def get_likelihood_for_params(params_fields, params_eval, sdpath, NSAMP, data_subtype='train', labels_subset=np.array([0, 1]), device='cpu', use_ema=False, lik_kw={}, uniform=False, ddp=False, vlb_timesteps=None, use_nx64=False, batch_over_params=True):
    '''
    :param params_fields: UNNORMALIZED params whose fields you want the lik evaluation of.
    :param model: diff model
    :param params_eval: UNNORMALIZED params you want to evaluate the likelihood conditional on.
    :return: Returns the likelihood of TRAIN distribution fields evaluated by a trained diffusion model for different parameters.
    Shape: len(params_eval), len(params_fields), NSAMP, len(vlb_timesteps))
    '''
    # get truefield tuples in a dataset with the appropriate normalization
    # neither are normalized!
    # get field transforms
    assert NSAMP<= 15
    dirname = sdpath[:sdpath.rindex('/') + 1]
    print('Dirname: ', dirname)
    tr, invtr = retrieve_data_transforms(dirname)

    # setup labels transforms
    params_minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5], dtype=np.float32)[labels_subset]
    params_maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0], dtype=np.float32)[labels_subset]

    params_eval_normed = (params_eval[:, labels_subset] - params_minimum)/(params_maximum - params_minimum)
    params_eval_normed = torch.tensor(params_eval_normed).to(device)

    # load model and losses
    sdict = torch.load(sdpath, map_location=device)

    if sdict['model_type'] == 'latentdiffusionmodule':
        raise NotImplementedError('This would NOT account for the encoder / decoder likelihood contribution.')
    else:
        model, diff = utils.load_model(sdpath, device, use_ema, ddp)
    print('ckp')

    if vlb_timesteps is None:
        cond_likelihoods = np.zeros((len(params_eval), len(params_fields), NSAMP))
    else:
        cond_likelihoods = np.zeros((len(params_eval), len(params_fields), NSAMP, len(vlb_timesteps)))

    
    #for idxpeval in range(len(params_eval)):
    for idxptrue in range(len(params_fields)):
        params_sub = np.vstack([params_fields[idxptrue]] * NSAMP)
        if use_nx64:
            paramstrue, truefields = get_truefields_for_sampled_fields({'params': params_sub}, type='custom', custom_file='../data_processed/LogMaps_Mcdm_IllustrisTNG_LH_z=0.00_Nx64_train.npy')
        else:
            paramstrue, truefields = get_truefields_for_sampled_fields({'params': params_sub}, type=data_subtype)
        fields_input = tr(torch.tensor(truefields).unsqueeze(1).to(device)) #NSAMPxCHW
        for s in range(fields_input.shape[0]):
            cond_likelihoods[:, idxptrue, s,...] = compute_conditional_likelihood_for_fields_fast(fields_input[s].unsqueeze(0), params_eval_normed, model, diff, vlb_timesteps=vlb_timesteps).detach().cpu().numpy()
    return cond_likelihoods


### Chisq functions
def evaluate_paramwise_reducedchisq(savepath, num_gen_samples_for_chisq, data_subtype='train', return_truefields=False, custom_file=None, num_true=15, logfields=True, normalize=True):
    """
    Args:
        savepath: Path to dictionary with samples, params.
        num_gen_samples_for_chisq: Number of sampled fields to use to compute the reduced chisq. [NCols in rchisqmat]
        data_subtype: Arg of get_truefields_for_sampled...
        return_truefields: Boolean. True if you want to return truefields.
        custom_file: Arg of get_truefields...
        num_true: How many true fields to use to compute the rchisq. Almost always 15.
        logfields: If True (default), power spectrum of the log fields. Else, 10^fields. This is the transformation applied to the FIELD and not to the pk.
    Returns:
        if return_truefields:
            Truefields corresponding to |Uniquesampledparams|x15, Uniquesampledparams, Rchisqmat: Nparams_Unique x Num_samples
            These are the logfields regardless of whether logfields=True or False.
        else:
            Uniquesampledparams, Rchisqmat: Nparams_Unique x Num_samples
    """
    resdict = pickle.load(open(savepath, 'rb'))
    assert len(resdict['samples']) == 1
    uni_sampledparams, idx_unip, count_unip = np.unique(resdict['params'], axis=0, return_index=True, return_counts=True)
    assert len(np.unique(count_unip)) == 1, "The number of samples for each unique parameter must be the same"
    nsamps_available = count_unip[0]
    assert num_gen_samples_for_chisq<=nsamps_available, "The number of available generated samples must be more than the number you want to use to compute the rchisq"
    rearrange_to_occorder = np.argsort(idx_unip)  # Order of occurrence
    uni_sampledparams, idx_unip, count_unip = uni_sampledparams[rearrange_to_occorder], idx_unip[rearrange_to_occorder], count_unip[
        rearrange_to_occorder]
    print('Number of unique sampled params: ', len(uni_sampledparams))
    
    rchisqmatrix = np.zeros((len(uni_sampledparams), num_gen_samples_for_chisq))

    # Retrieve true fields for all parameters
    ptrue_all = []
    for ip in range(uni_sampledparams.shape[0]):
        # for a single parameter: ip
        ptrue_all.append(np.repeat(uni_sampledparams[ip].reshape((1, -1)), 15, axis=0)) #get ALL 15 fields for each of the params in the dict
    
    ptrue = np.vstack(ptrue_all)
    ptrue, fieldstrue = get_truefields_for_sampled_fields({'params': ptrue}, type=data_subtype, custom_file=custom_file)
    #<p0>_15, <p1>_15.... Fieldstrue has length Nfieldsx15.

    assert ptrue.shape[0]==fieldstrue.shape[0]
    assert resdict['params'].shape[0]==resdict['samples'][0].shape[0]

    if not logfields:
        print('Raising fields to the power of 10 before computing P(k)')
    for ip in range(uni_sampledparams.shape[0]):
        # index only the fields belonging to uni_sampledparams[ip]
        isubsamp = np.arange(ip*nsamps_available, nsamps_available*(ip+1)).astype(int) #indices for resdict['samples'] and resdict['params']
        isubtrue = np.arange(ip * 15, 15 * (ip + 1)).astype(int) #indices for fieldstrue, ptrue
        assert np.allclose(np.repeat(uni_sampledparams[ip].reshape((1, -1)), 15, axis=0), ptrue[isubtrue]), "Error in pulling out the correct true fields for the parameter"
        assert np.allclose(np.repeat(uni_sampledparams[ip].reshape((1, -1)), nsamps_available, axis=0),
                           resdict['params'][isubsamp]), "Mismatch in the unique sampled parameters and the parameters corresponding to the generated samples"
        
        allfields = [fieldstrue[isubtrue]] + [samp[isubsamp, 0] for samp in resdict['samples']]
        if logfields:
            kvals, pklist = get_powspec_for_samples(allfields, normalize=normalize)
        else:
            kvals, pklist = get_powspec_for_samples([10.0**f for f in allfields], normalize=normalize)
        # get the mean of the samples
        meantrue, invvar = np.mean(pklist[0][:num_true], axis=0), np.std(pklist[0][:num_true], ddof=1, axis=0) ** (-2)
        samppk = pklist[1][:num_gen_samples_for_chisq] #only use num_samples to compute_rchisq
        rchisqmatrix[ip, :] = np.sum((samppk - meantrue) ** 2 * invvar, axis=1) / (len(kvals) - 1) # Corrected to len(kvals) in notebooks.
    assert fieldstrue.max() < 20, 'Fields seem to have been exponentiated.'
    if return_truefields:
        return fieldstrue, uni_sampledparams, rchisqmatrix
    else:
        return uni_sampledparams,rchisqmatrix
