import numpy as np
import os
import torch
from torch import nn
import datetime
import pickle
import hamiltorch
from functools import partial
hamiltorch.set_random_seed(123) # making sure it's reproducible across repeats?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import evaluate
import utils
import hf_diffusion as hfd

class TimeBatchedCondModel(nn.Module):
        def __init__(self, model, timesteps, x, diffusion, num_batches=1, compute_gradients=False):
            super().__init__()
            self.model = model
            self.timesteps = timesteps
            self.diff = diffusion
            self.x = x
            self.num_batches = num_batches
            size = int(len(timesteps)/self.num_batches)
            self.timestep_batches = torch.split(timesteps, size)
            print(f'Dividing timesteps into {self.num_batches} batches of size', len(self.timestep_batches[0]))
            labels_subset = np.array([0, 1])
            self.PMIN = torch.tensor(np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])[labels_subset]).to(torch.float32).to(device)
            self.PMAX = torch.tensor(np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])[labels_subset]).to(torch.float32).to(device)
            self.compute_gradients = compute_gradients

        def normalize(self, theta):
            """
            theta: dim=1. len=2.
            """
            params_out = (theta - self.PMIN) / (self.PMAX - self.PMIN)
            return params_out
            
        def get_log_probability(self, param, eval_gradients=True):
            """
            param: NOT transformed.
            """
            print('Proposal: ', param.detach().cpu().numpy())
            if ((param[0]-self.PMIN[0])>=0) and ((param[1]-self.PMIN[1])>=0) and ((self.PMAX[0] - param[0])>=0) and ((self.PMAX[1] - param[1])>=0):
                with torch.set_grad_enabled(True): # This is the default, but let's be explicit.
                    # reimplements a t-batched version of get_single_vlb_timestep
                    for it, tbatch in enumerate(self.timestep_batches):
                        xbatch = self.x.expand(len(tbatch), self.x.shape[1], self.x.shape[2], self.x.shape[3]) #.to(torch.float32)
                        pbatch = param.view(1, len(param)).expand(len(tbatch), len(param))
                        x_t = self.diff.q_sample(xbatch, tbatch)  # [1, T-1] in index -> x_[2, T] in math #1*1*Nx^2
                        if not eval_gradients:
                            print(xbatch.shape)
                        cond_denoise_model_func = lambda x_input, time_input: self.model(x_input, time_input, self.normalize(pbatch)) # because p_post wants something that takes x and t
                        mean2, logvar2 = self.diff.p_posterior_mean_logvariance(cond_denoise_model_func, x_t, tbatch)  # p [1, T-1] | [2, T] in math

                        # for [1, T-1]
                        tnzmask = (tbatch!=0)
                        mean1, logvar1 = self.diff.q_posterior_mean_logvariance(xbatch[tnzmask], x_t[tnzmask], tbatch[tnzmask])  # q(x_t-1 | x_0, x_t): x_[1, T-1] |x_0, x_[2, T] in math
                        kl_divs = hfd.normal_kl(mean1, logvar1, mean2[tnzmask], logvar2[tnzmask]).sum(dim=[1, 2, 3])  # BCHW -> B = len(timesteps) 
                        #BUG: this throws the nan. Maybe because there was already a nan here?
                        assert kl_divs.shape[0]==tnzmask.sum()
                        # for 0: mean2 = mu0_mean in get_single_vlb_term
                        if it==0:
                            # assuming that timesteps is sorted in ascending order.
                            nll = torch.sum(hfd.neglog_gaussian(xbatch[~tnzmask], mean2[~tnzmask], self.diff.betas[0]), dim=[1, 2, 3])
                            vlb_loss = -(nll.sum()+kl_divs.sum())  # sum over timesteps
                            if self.compute_gradients and eval_gradients: # we're batching gradients AND the current eval wants the gradient.
                                # print('Computing gradients: ', vlb_loss, param)
                                accumulated_grad = torch.autograd.grad(vlb_loss, param, retain_graph=False)[0]
                
                        else:
                            value = kl_divs.sum()
                            if self.compute_gradients and eval_gradients:
                                accumulated_grad += torch.autograd.grad(value, param, retain_graph=False)[0] # NOT vlb_loss since you've already accounted for its grad
                            vlb_loss += value # sum over timesteps
            else: # Outside prior
                vlb_loss = torch.tensor(-1e8, requires_grad=True) # Let's use 1e8 as a standin for inf
                if self.compute_gradients and eval_gradients:
                    accumulated_grad = torch.zeros([0.0, 0.0]).to(torch.float32).to(param.device)
            if self.compute_gradients and eval_gradients:
                return vlb_loss, accumulated_grad  # Return vlb_loss
            else:
                return vlb_loss

### Config
MODEL = 'Run_10-30_2-32'
CKPNUM = '260000'
sdpath = f'results/samples_exps/{MODEL}/checkpoint_{CKPNUM}.pt'
N_TIME = 20
num_time_batches = 2
ACCUMULATE_GRADIENTS = True
PSEED = 1995

# get field transform for x
dirname = sdpath[:sdpath.rindex('/')+1]
print(dirname, type(dirname))
tr, invtr = evaluate.retrieve_data_transforms(dirname)
model, diff = utils.load_model(sdpath, device, False, False)
Tsub = list(np.arange(N_TIME).astype(int))



#get field
trueparam = evaluate.get_validation_cosmo_params(Bsize=1, replace=False, seed=PSEED)
trueparams, truefields = evaluate.get_truefields_for_sampled_fields({'params': trueparam}, type='validation')
ttfield = tr(torch.tensor(truefields).unsqueeze(0).to(device))
print('Running parameter inference for true parameter: ', trueparam, ' on device ', device)
condmodel = TimeBatchedCondModel(model, torch.tensor(Tsub).to(device), ttfield, diff, num_batches=num_time_batches, compute_gradients=ACCUMULATE_GRADIENTS)

#Init Param
theta = torch.tensor(np.array([0.3, 0.8])).to(torch.float32).to(device)
params_init = theta

# HMC configs
sampler = hamiltorch.Sampler.HMC
hmc_kwargs = {'step_size': 5e-4, 'num_steps_per_sample': 5, 'num_samples': 10, 
              'inv_mass': torch.tensor([1, 5]).to(torch.float32).to(device), 'accumulated_gradient': ACCUMULATE_GRADIENTS}

params_hmc = hamiltorch.sample(log_prob_func=condmodel.get_log_probability,
                params_init=params_init, sampler=sampler, **hmc_kwargs)#, desired_accept_rate=hmc_target_rate, burn=burn)

psamps = np.array([p.detach().cpu().numpy() for p in params_hmc])

# Generate the current date and time
now = datetime.datetime.now()
date_format = now.strftime("%m%d_%H%M")
outdir = f'results/hmc_samps/exp1_Ntime/{date_format}/'

os.makedirs(outdir) 
np.save(outdir+'samps.npy', psamps)

config = {
    'MODEL': MODEL,
    'CKPNUM': CKPNUM,
    'sdpath': sdpath,
    'Tsub': Tsub,
    'trueparam': trueparam,
    'pseed': PSEED,
    'truefields': truefields,
    'params_init': theta.detach().cpu().numpy(),
    'params_hmc': psamps,
    'sampler': str(sampler),
    'num_time_batches': num_time_batches,
    #'inv_mass_matrix': inv_mass_matrix.detach().cpu().numpy(),
}
for k, v in hmc_kwargs.items():
    if k=='inv_mass':
        v = v.detach().cpu().numpy()
    config[k] = v

# Save the config dictionary to a file
config_file = outdir + 'config.pkl'
with open(config_file, 'wb') as f:
    pickle.dump(config, f)