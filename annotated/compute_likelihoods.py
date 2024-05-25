"""Grid-based parameter inference."""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import scipy
import os
import argparse

import utils
import hf_diffusion as hfd
import evaluate

from lampe.inference import NPE

if torch.cuda.is_available():
    device='cuda'
else:
    device= 'cpu'

def dchisq_interp(x, a, b, c):
    return np.abs(a) * x**2 + b * x + c

def estimate_local_gaussian_parameters(xpts, dchisqpts):
    """
    Fit a mean and sigma to a parabola to estimate the local minimum of the chi-squared surface.
    Args:
        xpts: 1D array of x values
        dchisqpts: 1D array of chi-squared values. i.e. -2*lnL
    Returns:
        Tuple of (mu, sigma)
    """
    dmin = np.min(dchisqpts)
    dchisqpts = dchisqpts - dmin # technically dont even need the -dmin
    popt, _ = scipy.optimize.curve_fit(dchisq_interp, xpts, dchisqpts)
    sig_estimate = 1/np.sqrt(np.abs(popt[0]))
    mu_estimate = -popt[1]/(2*popt[0])
    return mu_estimate, sig_estimate


def get_68_perc_interval(pdf):
    cdf = np.cumsum(pdf)
    return [np.where(cdf>0.16)[0][0], np.argmax(pdf), np.where(cdf>0.84)[0][0]] #upper index is included

def get_marginal_predictions(peval, condlikvals, NDISC=50, return_marginals=False, interpolate=False):
    delnll = 2*(condlikvals - np.min(condlikvals, axis=0, keepdims=True)) # convert to chisquared dbn
    pdf = np.exp(-0.5*delnll)
    pdf_t0 = pdf[:, 0, 0, 0] #TODO: swap to isamp
    om_mrange, sig8_range = peval[:, 0].reshape((NDISC, NDISC)).mean(axis=0), peval[:, 1].reshape((NDISC, NDISC)).mean(axis=1)
    marg_omm, marg_sig8 = np.sum(pdf_t0.reshape((NDISC, NDISC)), axis=0), np.sum(pdf_t0.reshape((NDISC, NDISC)), axis=1) # marginalize by summing over the other axis
    # axis=0 => columns intact (Omega_m intact from peval[:, 0].reshape.mean(axis=0) ) => summing over rows => marginalizing over sigma_8

    marg_pdf_omm = marg_omm/marg_omm.sum()
    marg_pdf_sig8 = marg_sig8/marg_sig8.sum()
    if interpolate:
        assert not return_marginals
        dchi_omm = -2*np.log(marg_pdf_omm)
        dchi_sig8 = -2*np.log(marg_pdf_sig8)
        #omm
        idx_sort = np.argsort(dchi_omm)
        # we want the points adjacent to the minimum on either side to avoid weird behavior
        omm_minpts = om_mrange[idx_sort][:2]
        pol = omm_minpts[1] - omm_minpts[0]
        idx=2
        while (om_mrange[idx_sort][idx] - omm_minpts[0])*pol > 0: # break when you find the opp polarity
            idx+=1
        omm_minpts = np.hstack([omm_minpts, om_mrange[idx_sort][idx]])
        dchi_min = np.hstack([dchi_omm[idx_sort][:2], dchi_omm[idx_sort][idx]])
        mu_est, sig_est = estimate_local_gaussian_parameters(omm_minpts, dchi_min)
        omm_preds = [mu_est - sig_est, mu_est, mu_est+sig_est]

        #sig8
        idx_sort = np.argsort(dchi_sig8)
        sig8_minpts = sig8_range[idx_sort][:2]
        pol = sig8_minpts[1] - sig8_minpts[0]
        idx=2
        while (sig8_range[idx_sort][idx] - sig8_minpts[0])*pol > 0:
            idx+=1
        sig8_minpts = np.hstack([sig8_minpts, sig8_range[idx_sort][idx]])
        dchi_min = np.hstack([dchi_sig8[idx_sort][:2], dchi_sig8[idx_sort][idx]])
        mu_est, sig_est = estimate_local_gaussian_parameters(sig8_minpts, dchi_min)
        sig8_preds = [mu_est - sig_est, mu_est, mu_est+sig_est]
        return omm_preds, sig8_preds

    else:
        #omm
        is1_low, ipred, is1_upp = get_68_perc_interval(marg_pdf_omm)
        print('Omm indices:', is1_low, ipred, is1_upp)
        omm_preds = om_mrange[is1_low], om_mrange[ipred], om_mrange[is1_upp]
        
        #sig8
        is1_low, ipred, is1_upp = get_68_perc_interval(marg_pdf_sig8)
        print(is1_low, ipred, is1_upp)
        sig8_preds = sig8_range[is1_low], sig8_range[ipred], sig8_range[is1_upp]
        if return_marginals:
            return omm_preds, sig8_preds, [pdf_t0, marg_pdf_omm, marg_pdf_sig8]
        else:
            return omm_preds, sig8_preds

def get_marginal_predictions_given_pdf(peval, pdf, NDISC=50):
    """
    Get marginals using NPE.
    Args:
        peval: 2D param_grid flattened
        pdf: 2D grid of p(theta_eval | x_stat) flattened
        NDISC: discretization along each axis for unflattening
    Returns:
        Tuple of
        Om_m_68p_low, Om_m_minimum, Om_m_68p_high
        sig_8_68p_low, sig_8_minimum, sig_8_68p_high
    """
    om_mrange, sig8_range = peval[:, 0].reshape((NDISC, NDISC)).mean(axis=0), peval[:, 1].reshape((NDISC, NDISC)).mean(axis=1)
    # mean not really needed, it's the same as peval[:,0].reshape((NDISC, NDISC))[0, :]
    marg_omm, marg_sig8 = np.sum(pdf.reshape((NDISC, NDISC)), axis=0), np.sum(pdf.reshape((NDISC, NDISC)), axis=1)
    
    marg_pdf_omm = marg_omm/marg_omm.sum()
    marg_pdf_sig8 = marg_sig8/marg_sig8.sum()
    #omm
    is1_low, ipred, is1_upp = get_68_perc_interval(marg_pdf_omm)
    print('Omm indices:', is1_low, ipred, is1_upp)
    omm_preds = om_mrange[is1_low], om_mrange[ipred], om_mrange[is1_upp]
    
    #sig8
    is1_low, ipred, is1_upp = get_68_perc_interval(marg_pdf_sig8)
    print(is1_low, ipred, is1_upp)
    sig8_preds =sig8_range[is1_low], sig8_range[ipred], sig8_range[is1_upp]
    return omm_preds, sig8_preds

def get_npe_baseline_predictions(params, fields, npe_path, NDISC=50, D_THETA=2, D_STAT=128, npe_kwargs={'transforms': 3, 'hidden_features': [64] * 3}, grid_width=0.1):
    estimator = NPE(D_THETA, D_STAT, **npe_kwargs)
    estimator.load_state_dict(torch.load(npe_path))
    estimator.eval()

    assert params.shape[0]==fields.shape[0]
    rng = np.random.default_rng()
    _, pk = evaluate.get_powspec_for_samples([fields])
    xvals = torch.tensor(pk[0])
    om_predictions, sig8_predictions, pdf2d = [], [], []
    for ip in range(len(params)):
        ptruth = params[ip].reshape((1, -1))
        pevalgrid = np.meshgrid(np.linspace(max(ptruth[0, 0]-grid_width, 0.1), min(ptruth[0, 0]+grid_width, 0.5), NDISC),
                            np.linspace(max(ptruth[0, 1]-grid_width, 0.6), min(ptruth[0, 1]+grid_width, 1.0), NDISC))
        peval = np.vstack([pevalgrid[0].flatten(), pevalgrid[1].flatten(),
                                rng.uniform(0.5, 2.0, NDISC ** 2 * 4).reshape((4, NDISC ** 2))]).T
        peval_sub = peval[:, :2]
        
        # prepare data
        stat = xvals[ip:ip+1].expand((peval_sub.shape[0], xvals.shape[1]))
        peval_sub = torch.tensor(peval_sub).to(torch.float32)
        stat = stat.to(torch.float32)
        print(stat.shape, peval_sub.shape)

        pdfvals = estimator(torch.tensor(peval_sub), stat)

        om_preds, sig8_preds = get_marginal_predictions_given_pdf(peval_sub.numpy(), pdfvals.detach().numpy(), NDISC)
        om_predictions.append(om_preds)
        sig8_predictions.append(sig8_preds)
        pdf2d.append((peval_sub, pdfvals))
    return om_predictions, sig8_predictions, pdf2d

def list_of_ints(arg: str):
    return list(map(int, arg.split(',')))
#geeksforgeeks 

#https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-in-a-range-using-arg
def my_float(x: str):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))
    return x

def save_condlikelihoods_for_fields(args):
    print('Device', device, flush=True)
    if args.data_subtype== 'validation':
        ptrueall = evaluate.get_validation_cosmo_params(Bsize=args.num_params, seed=args.pseed)
    elif args.data_subtype == 'simba':
        ptrueall = evaluate.get_simba_cosmo_params(Bsize=args.num_params, seed=args.pseed)
    else:
        raise NotImplementedError('data_subtype must be validation or simba')

    for ipf in range(len(ptrueall)):
        print('#####')
        ptruth = ptrueall[ipf:ipf+1]
        print(f'True Param {ipf}', ptruth)
        NDISC = args.ndiscretization

        rng = np.random.default_rng(22) #If the other parameters don't matter at all then the grid is constant. 
        pevalgrid = np.meshgrid(np.linspace(max(ptruth[0, 0]-args.grid_extent, 0.1), min(ptruth[0, 0]+args.grid_extent, 0.5), NDISC),
                            np.linspace(max(ptruth[0, 1]-args.grid_extent, 0.6), min(ptruth[0, 1]+args.grid_extent, 1.0), NDISC))
        peval = np.vstack([pevalgrid[0].flatten(), pevalgrid[1].flatten(),
                           rng.uniform(0.5, 2.0, NDISC ** 2 * 4).reshape((4, NDISC ** 2))]).T
        for t_seed in args.transform_seeds: # loop over seeds
            for vlb_seed in args.vlb_seeds:
                savename = 'p_{}_tseed_{}_vlbseed_{}.pkl'.format(ipf, t_seed, vlb_seed)
                print('Transform Seed=', t_seed, ' VLB Seed =', vlb_seed, '-> Saved to ', savename)
                if os.path.exists(os.path.join(args.savedir, savename)):
                    print('Skipping. Output Already Exists.')
                else:
                    condlikvals = evaluate.get_fields_for_params_and_compute_likelihoods(ptruth, peval, args.sdpath, args.nsamples, args.data_subtype, device=device, vlb_timesteps=args.vlb_timesteps, N_EvalBatches=args.Nbatches, seed=t_seed, vlbt_seed=vlb_seed, reseed_over_t=args.reseed_over_time) #Neval x Ntrue=1 x Nsamps x T
                    sav_obj = {'ptruth': ptruth, 'peval': peval, 'condlikvals': condlikvals, 'vlb_timesteps': args.vlb_timesteps, 'transform_seed': t_seed, 'vlb_seed': vlb_seed}
                    pickle.dump(sav_obj, open(os.path.join(args.savedir, savename), 'wb'))
    return

def process_plot_contours(args):
    pfiles = os.listdir(args.savedir)
    NDISC = args.ndiscretization
    for ipf, pfile in enumerate(pfiles):
        if pfile in ['args.pkl', 'plots']:
            continue
        print(f'Plot for {pfile}')
        likdict = pickle.load(open(os.path.join(args.savedir, pfile), 'rb'))
        ptruth = likdict['ptruth']
        peval = likdict['peval']
        condlikvals = likdict['condlikvals']
        delnll = 2*(condlikvals - np.min(condlikvals, axis=0, keepdims=True))
        for s in range(delnll.shape[2]):
            for idxt in range(len(args.vlb_timesteps)):
                plt.figure()
                c = plt.contourf(peval[:, 0].reshape((NDISC, NDISC)), peval[:, 1].reshape((NDISC, NDISC)), delnll[:, 0, s, idxt].reshape((NDISC, NDISC)),
                                 levels=60, cmap='viridis')
                plt.contour(peval[:, 0].reshape((NDISC, NDISC)), peval[:, 1].reshape((NDISC, NDISC)), delnll[:, 0, s, idxt].reshape((NDISC, NDISC)),
                            [2.30, 6.18, 11.83], colors=['r', 'r', 'r'])
                plt.scatter(ptruth[0, 0], ptruth[0, 1], s=100, marker='x', c='k')
                plt.colorbar(c)
                plt.xlabel('$\Omega_m$')
                plt.ylabel('$\sigma_8$')
                plt.title(r'-2$\Delta ln \mathcal{L}$ t=' + str(args.vlb_timesteps[idxt]))
                plt.savefig(args.plotsdir + pfile[:pfile.rindex('.pkl')]+'_sample_{}_t_{}.png'.format(s, args.vlb_timesteps[idxt]))
                plt.show()
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdpath", type=str, required=True) #Checkpoint
    parser.add_argument("--savedir", type=str, required=True) #Saving directory
    parser.add_argument("--data_subtype", type=str, choices=['validation', 'simba'], default='validation')
    parser.add_argument("--ndiscretization", type=int, default=50) #Number of grid points along each dimension
    parser.add_argument("--nsamples", type=int, default=1) #Field samples per parameter
    parser.add_argument("--pseed", type=int, default=2) #Parameter seed
    parser.add_argument("--transform_seeds", type=list_of_ints, default="3, 10") #Transform seed
    parser.add_argument("--vlb_seeds", type=list_of_ints, default="4, 30") #VLB t seed
    parser.add_argument("--reseed_over_time", action='store_true') #Whether each timestep has a different seed so the noise pattern isnt the same for all timesteps. Note, the seed for different parameters on the grid will still be the same. False if omitted.
    parser.add_argument("--vlb_timesteps", type=list_of_ints, default="0, 2, 5, 8, 10, 15, 20") #VLB seed
    parser.add_argument("--num_params", type=int, default=5) #Number of params
    parser.add_argument("--grid_extent", type=my_float, default=0.1) #Number of params
    parser.add_argument("--Nbatches", type=int, default=50) #Number of batches to divide params_eval into
    parser.add_argument("--plotsdir", type=str, required=True) #Plots directory
    args = parser.parse_args()
    pickle.dump(args.__dict__, open(os.path.join(args.savedir, 'args.pkl'), 'wb'))
    save_condlikelihoods_for_fields(args)
    # process_plot_contours(args)
