import torch
import time
import random
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

#https://github.com/openai/improved-diffusion/blob/783b6740edb79fdb7d063250db2c51cc9545dcd1/improved_diffusion/losses.py#L12
## KL Div between 2 Gaussians
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians in ONE dimension.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    Since the MV distributions are all diagonal you can factorize over the sum.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]
    # -1 + ln|Σ_q| - ln|Σ_p| + tr(Σ_p/Σ_q) + (μ_2 - μ_1)^T Σ_q^(-1) (μ_2 - μ_1)
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def normal_covkl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians in ONE dimension. Does not assume a diagonal covariance matrix.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]
    # -1 + ln|Σ_q| - ln|Σ_p| + tr(Σ_p/Σ_q) + (μ_2 - μ_1)^T Σ_q^(-1) (μ_2 - μ_1)
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def neglog_gaussian(x, mean, variance):
    term1  = ((x - mean)**2/(2*variance)) 
    term2 = 0.5*torch.log(2*np.pi*variance)
    return term1 + term2


class Diffusion():
    def __init__(self, betas):
        #NOTE: you're choosing to make loss-type an argument of p_losses and not the diffusion model class
        #you should then save it separately when saving a run
        # calculations for diffusion q(x_t | x_0) and others
        self.betas = betas
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)  # alpha_bar
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        # x_t = sqrt_alphas_cumprod* x_0 + sqrt_one_minus_alphas_cumprod * eps_t
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod) #Eq 7
        self.timesteps = len(self.betas)

        #extra things to save for when you compute the elbo
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)) #Eq 7 Coeff 1
        self.posterior_mean_coef2 = torch.sqrt(self.alphas)*(1.0 - self.alphas_cumprod_prev)/(1.0 - self.alphas_cumprod) #Eq 7 Coeff 2
        self.posterior_logvariance = torch.log(self.posterior_variance)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt((1. / self.alphas_cumprod) - 1)

        # loss_factor
        self.sqrt_loss_factor = F.pad(self.betas[1:] / (torch.sqrt(
            2.0 * self.posterior_variance[1:] * self.alphas[1:]) * self.sqrt_one_minus_alphas_cumprod[1:]), (1, 0), value=1.0)

    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None):
        '''
        Here, t=0 corresponds to x1
        t=t correponds to x_t+1
        t=[0, T-1] (T possible values).
        t is the index of betas
        '''
        if noise is None:
            noise = torch.randn_like(x_start)
        #print('Debugging q_sample noise', noise[-1, -1, -1, -1])
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_sample_incl_t0(self, x_start, t, noise=None):
        '''
        Here, t=0 corresponds to x0
        t=t correponds to xt
        t=[0, T] (T+1 possible values)
        t-1 is the index of betas
        '''
        if noise is None:
            noise = torch.randn_like(x_start)
        
        output = x_start.detach().clone()
        tnzmask = t!=0
        t_beta_index = t[tnzmask]-1
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t_beta_index, x_start[tnzmask].shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t_beta_index, x_start[tnzmask].shape)
        output[tnzmask] = sqrt_alphas_cumprod_t * x_start[tnzmask] + sqrt_one_minus_alphas_cumprod_t * noise[tnzmask]
        
        return output

    def q_mean_logvariance(self, x_start, t):
        '''
        Returns the mean and variance of q(x_t|x_0)
        :param x_start: x_0
        :param t: t is the INDEX and goes from [0, T-1] to give x_t[1, T]|x_0 because beta_t in the math is beta_{t-1} in the code.
        :return:
        '''
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, log_variance

    def q_posterior_mean_logvariance(self, x_start, x_t, t):
        '''
        Implements Eq 6.
        Returns the mean and variance of q(x_t-1 | x_t, x_0). This is INDEPENDENT of the noise_model
        :param x_start:
        :param x_t:
        :param t: INDEX of betas. goes from [0, T-1].
        :return:
        '''
        mean = (extract(self.posterior_mean_coef1, t, x_start.shape) * x_start) + (extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        logvariance = extract(self.posterior_logvariance, t, x_start.shape)
        return mean, logvariance

    def predict_x0_from_noise(self, x_t, t, noise):
        """Implements Eq 4, but in reverse."""
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t) - (extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def p_posterior_mean_logvariance(self, denoise_model, x_t, t):
        #Returns the mean and variance of p_θ(x_t-1 | x_t)
        #this is DEPENDENT on the CONDITIONAL noise_model | y
        noise_pred = denoise_model(x_t, t) #[2, T] in math. [1, T-1] in indices. THis is actually x_t+1
        x0_recon = self.predict_x0_from_noise(x_t, t, noise_pred) #Eq 11 # Exp: N_eval*1*Nx^2
        p_mean, _ = self.q_posterior_mean_logvariance(x0_recon, x_t, t) #Eq11: mu_θ = mu_tilda(x_t, x0_estimated)
        p_logvariance = extract(self.posterior_logvariance, t, x_t.shape) #TODO: Check that this shouldn't be t+/-1
        return p_mean, p_logvariance


    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1", labels=None):
        # L_CE <= L_VLB ~ Sum[eps_t - MODEL(x_t(x_0, eps_t), t) ]
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_t, t, labels)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == 'weightedl2':
            weightednoise = extract(self.sqrt_loss_factor, t, noise.shape)*noise
            weightedprednoise = extract(self.sqrt_loss_factor, t, noise.shape)*predicted_noise
            loss = F.mse_loss(weightednoise, weightedprednoise)
        elif loss_type == 'weightedhuber':
            weightednoise = extract(self.sqrt_loss_factor, t, noise.shape)*noise
            weightedprednoise = extract(self.sqrt_loss_factor, t, noise.shape)*predicted_noise
            loss = F.smooth_l1_loss(weightednoise, weightedprednoise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss

    @torch.no_grad()
    def timewise_loss(self, denoise_model, x_start, t, noise=None, loss_type="l1", labels=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_t, t, labels)
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise, reduction='none')
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise, reduction='none')
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise, reduction='none')
        else:
            raise NotImplementedError()
        loss = torch.mean(loss, dim=[-3, -2, -1]) #mean over all spatial dims
        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, label=None, cond_kwargs=None, noise=None):
        '''
        Eq 11: mu_theta(xt, t) = sqrt(1/alpha_t) * [xt - beta_t * MODEL(xt, t)/ sqrt(1-alpha_cumprod_t)]
        For t=0: mu_theta(xt, t)
        for t>0: mu_theta(xt, t) + posterior_std_t * N(0, 1)
        ^ MATH
        :param model:
        :param x: x_t+1. Due to indexing. x goes from x_1 to x_T (N(0, 1).
        :param t: [0, T-1]. Due to indexing. t goes from 0 to T-1. Corresponds to the time of the output in math.
        :param label:
        :param cond_kwargs: cond_fn, mask_params, classifier, classifier_scale. Passed as an external argument during p_sample
        :returns: p(x_t | x_t+1). Thus x_0 is the actual image and not the first noised timestep. x_T-1 is the first step in the chain given x_T which is the std normal.
        '''
        assert len(torch.unique(t))==1
        t_index = t[0]
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_output = model(x, t) if label is None else model(x, t, label) #epsilon
        if cond_kwargs is not None:
            cond_fn = cond_kwargs['cond_fn']
            #rescale= None
            if 'default_rescale' in cond_kwargs.keys():
                cond_kwargs['y'] = label
                rem_keys = ['cond_fn', 'default_rescale']
                model_kw = {k: cond_kwargs[k] for k in set(list(cond_kwargs.keys())) - set(rem_keys)}
                alpha_bar = extract(self.alphas_cumprod, t, x.shape)
                cond_term = - (1 - alpha_bar).sqrt() * cond_fn(x, t + 1, **model_kw)
                # eps = eps - rt(1-alpha_bar)*cond_fn
                # t+1 was because when you trained the noise-dependent classifier, t=0 corresponds to no noise
                # in the diffusion notation, t=0 corresponds to the first timestep

            elif 'rescale' in cond_kwargs.keys():
                cond_kwargs['y'] = label
                rescale = cond_kwargs['rescale']
                rem_keys = ['cond_fn', 'rescale']
                model_kw = {k: cond_kwargs[k] for k in set(list(cond_kwargs.keys())) - set(rem_keys)}
                alpha_bar = extract(self.alphas_cumprod, t, x.shape)
                cond_term = - (1 - alpha_bar).sqrt() * cond_fn(x, t + 1, **model_kw)
                rescale_factor = rescale * torch.abs(model_output).mean() / torch.abs(cond_term).mean()
                print(f'Timestep{t_index}: Rescale Factor', rescale_factor)
                cond_term *= rescale_factor

            else:
                assert 'gammas' in cond_kwargs.keys()
                gammas = cond_kwargs['gammas']
                assert len(gammas) == len(self.alphas_cumprod) #Gamma 0: step nearest to the image, Gamma 1k: nearest to N(0, 1)
                cond_kwargs['y'] = label
                rem_keys = ['cond_fn', 'gammas']
                model_kw = {k: cond_kwargs[k] for k in set(list(cond_kwargs.keys())) - set(rem_keys)}
                cond_term = - gammas[t_index] * cond_fn(x, t + 1, **model_kw)
                if t_index%200==0:
                    print('Model_output', torch.sqrt((model_output**2).mean()), ' Cond Fn Magnitude', torch.sqrt((cond_term**2).mean())) 

            #Add eps_cond = eps - cond_fn
            model_output = model_output + cond_term

        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        ) #Eq 11
        posterior_variance_t = extract(self.posterior_variance, t, x.shape)

        if t_index == 0:
            #print('Last step')
            return model_mean
        else:
            if noise is None:
                noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise




    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, model, shape, labels=None, return_all_timesteps=True, cond_kwargs=None, noise=None, return_specific_samples=None):
        device = next(model.parameters()).device
        print('sample device', device)
        b = shape[0]

        if labels is not None:
            assert labels.shape[0] == shape[0]

        # start from pure noise (for each example in the batch)
        if noise is None:
            img = torch.randn(shape, device=device)
        elif len(noise.shape)==4:
            img = noise.detach().clone()
        else:#len(noise.shape)==5
            img = noise[0]
        
        imgs = [img.cpu().numpy()]
        t0 = time.time()
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            #sample from reverse diff
            if i%200==0:
                print('Timestep', i, time.time()-t0)
            if noise is not None and len(noise.shape)==5:
                noise_i = noise[i]
            else:
                noise_i = None #since somehow fixing the same noise everytime hurts, the noise.shape==4 case corresponds to only keeping the initial guess constant
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), labels, cond_kwargs, noise=noise_i)
            if return_all_timesteps:
                imgs.append(img.cpu().numpy())
        if return_all_timesteps and return_specific_samples is None:
            return imgs
        elif return_all_timesteps and return_specific_samples is not None:
            return imgs[return_specific_samples]
        else:
            return img.cpu().numpy()
    
    @torch.no_grad()
    def p_sample_loop_mem_efficient(self, model, shape, labels=None, return_multiple_timesteps=True, cond_kwargs=None, noise=None, return_specific_samples=None):
        '''
        :param model:
        :param shape:
        :param labels:
        :param return_multiple_timesteps: Return multiple snapshots from the reverse trajectory
        :param cond_kwargs:
        :param noise: If length=4. Then JUST fixing the starting draw from the standard normal and leaving the trajectory noise to be variable.
                      If length==5. Then fixing the initial draw AND the trajectory noise samples.
        :param return_specific_samples: Indices of the snapshots to return from the REVERSE trajectory.
        0: Noise
        t increases, more image like
        T(usually 2000 or 1000): Image
        :return:
        '''
        print('mem efficient')
        
        device = next(model.parameters()).device
        print('sample device', device)
        b = shape[0]

        if labels is not None:
            assert labels.shape[0] == shape[0]

        # start from pure noise (for each example in the batch)
        if noise is None:
            img = torch.randn(shape, device=device)
        elif len(noise.shape)==4:
            img = noise.detach().clone()
        else:
            assert len(noise.shape)==5
            img = noise[self.timesteps] # should probably be self.timesteps, was 0 earlier.

        if return_specific_samples is not None and 0 in return_specific_samples:
            assert np.all(np.array(return_specific_samples)>=0)
            assert np.all(np.array(return_specific_samples)<=self.timesteps) #reverse_index can take T+1 values from 0 (noise) to T (image)
            imgs = [img.cpu().numpy()]
        else:
            imgs = []
        
        t0 = time.time()
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            #sample from reverse diff
            #reverse_index=0 is the noise (input to this loop)
            #the end of the i=T-1^th iteration gives you reverse_index=1
            #the last iteration, i=0, gives you the output (reverse_index=T)
            # thus reverse_index + i = T
            if i%200==0:
                print('Timestep', i, time.time()-t0)
            if noise is not None and len(noise.shape)==5:
                noise_i = noise[i]
            else:
                noise_i = None #since fixing the same noise everytime is a delta fn (not a Gaussian), the noise.shape==4 case corresponds to only keeping the initial guess constant
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), labels, cond_kwargs, noise=noise_i)
            if return_multiple_timesteps and self.timesteps-i in return_specific_samples:
                imgs.append(img.cpu().numpy())
        if return_multiple_timesteps:
            return imgs
        else:
            return img.cpu().numpy()

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3, labels=None, return_all_timesteps=True, cond_kwargs=None):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), labels=labels, return_all_timesteps=return_all_timesteps, cond_kwargs=cond_kwargs)


    @torch.no_grad()
    def compute_vlb(self, cond_model, x_inputs):
        '''
        :param model: Trained noise prediction model | y (Conditional on y)
        :param x_inputs: Tensor BCHW. ALREADY Normalized inputs. Must lie between [-1, 1].
        :return:
        Computes an upper bound on the Neg Log Likelihood of the data.
        NOTE: t in the math / paper corresponds to t-1 in terms of indexing into any of the chain variables in this structure and q_sample
        '''
        D = torch.prod(torch.tensor(x_inputs.shape[1:]))

        #Loss_T term (T-1th index for the beta chain in code)
        lossT_term = self.get_single_vlb_term(cond_model, x_inputs, self.timesteps)

        #Loss_t terms for the states going from x_1| x_2 to x_T-1 | x_T in math
        #Check upper limit here
        losst_terms = []
        for t_index in range(1, self.timesteps): #[0, T-2] |[1, T-1] in indices, which is [1, T-1] |[2, T] in math / paper (because q_sample and all the other variables in the chain take t=0 to actually mean beta_1 or the first timestep in math) or use q_sample_incl_t0
            losst_term = self.get_single_vlb_term(cond_model, x_inputs, t_index)
            losst_terms.append(losst_term)
        losst_combined = torch.stack(losst_terms) #TB
        losst_sum = torch.sum(losst_combined, dim=0) #B

        #Loss_0 term: Continuous NLL for a Gaussian
        loss0_term = self.get_single_vlb_term(cond_model, x_inputs, 0)
        return (losst_sum + loss0_term + lossT_term)/D #TODO ??? Figuring out summing and stuff

    # @torch.no_grad() # remove torch.no_grad()
    def get_single_vlb_term(self, cond_model, x_inputs, t_index, seed=20, no_grad=True):
        '''
        :param model: Trained noise prediction model | y (Conditional on y)
        :param x_inputs: Tensor BCHW. ALREADY Normalized inputs. Must lie between [-1, 1].
        :param t_index: int, NOT a tensor. Index of the timestep to compute the VLB term for. Must be in [0, T]
        :return:
        Computes an upper bound on the Neg Log Likelihood of the data.
        NOTE: t in the math / paper corresponds to t-1 in terms of indexing into any of the chain variables in this structure and q_sample
        '''
        with torch.set_grad_enabled(not no_grad):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            # print("Grad status", x_inputs.requires_grad)
            #print('VLB Seed= ', seed)
            if t_index==self.timesteps:
                # Loss_T term (T-1th index for the beta chain in code)
                T = torch.full((x_inputs.shape[0],), self.timesteps - 1, device=x_inputs.device, dtype=torch.long)
                mean1, logvar1 = self.q_mean_logvariance(x_inputs, t=T)
                mean2, logvar2 = 0.0, 0.0
                return normal_kl(mean1, logvar1, mean2, logvar2).sum(dim=[1, 2, 3])  # BCHW->B

            elif t_index==0:
                # Loss_0 term: Continuous NLL for a Gaussian
                t0 = torch.full((x_inputs.shape[0],), 0, device=x_inputs.device, dtype=torch.long)
                x_1 = self.q_sample(x_inputs, t0)  # WAIT: But this assumes t=0 is the first timestep in the chain, so yeah this should be right.
                #print('Seed', seed)
                #print('x_0', x_inputs[0, 0, 0, 0]) # Issue: THIS is also changing
                #print('x_1', x_1[0, 0, 0, 0])
                mu0_mean = self.p_sample(cond_model, x_1, t0)  # p(x_0 | x_1)
                return torch.sum(neglog_gaussian(x_inputs, mu0_mean, self.betas[0]), dim=[1, 2, 3]) #loss0_term: #B
            else:
                # Loss_t terms for the states going from x_1| x_2 to x_T-1 | x_T in math
                # [0, T-2] |[1, T-1] in indices, which is [1, T-1] |[2, T] in math / paper (because q_sample and all the other variables in the chain take t=0 to actually mean beta_1 or the first timestep in math) or use q_sample_incl_t0
                assert t_index in range(1, self.timesteps), "t_index must be in [0, T]"
                t = torch.full((x_inputs.shape[0],), t_index, device=x_inputs.device, dtype=torch.long)
                x_t = self.q_sample(x_inputs, t)  # [1, T-1] in index -> x_[2, T] in math #1*1*Nx^2. So this is actually x_t+1.
                mean1, logvar1 = self.q_posterior_mean_logvariance(x_inputs, x_t, t)  # q(x_t-1 | x_0, x_t): x_[1, T-1] |x_0, x_[2, T] in math
                mean2, logvar2 = self.p_posterior_mean_logvariance(cond_model, x_t, t)  # p [1, T-1] | [2, T] in math
                return normal_kl(mean1, logvar1, mean2, logvar2).sum(dim=[1, 2, 3])  # BCHW -> B


def eval_loss_state(loss_args, get_individual=False):
    loss_version = loss_args['loss_version']
    loss_type =  loss_args['loss_type']
    recon_weight = float(loss_args['recon_weight'])

    def loss_func(diffusion, model, batch, t, labels):
        if loss_version=='standard':
            loss = diffusion.p_losses(model, batch, t, loss_type=loss_type, labels=labels)
            if get_individual:
                return loss, {'loss': loss.item()}
            else:
                return loss
            return 
        elif loss_version=='reconstruction':
            #print(batch.shape, t.shape)
            loss1 = diffusion.p_losses(model, batch, t, loss_type=loss_type, labels=labels)
            #recon_loss
            z_0 = diffusion.q_sample(x_start=batch, t=torch.zeros(batch.shape[0], device=batch.device, dtype=int))
            norm = diffusion.sqrt_one_minus_alphas_cumprod[0] if loss_args['recon_norm']=='Sigma' else 1.0
            diff = (z_0 - batch)/norm
            loss_recon = recon_weight*torch.mean((diff)**2)
            if get_individual:
                return loss1+loss_recon, {'diffusion_loss': loss1.item(), 'recon_loss': loss_recon.item(), 'loss': loss1.item()+loss_recon.item()}
            else:
                return loss1 + loss_recon
        else:
            raise NotImplementedError
    return loss_func


def compute_encdecdiffloss_state(loss_args, get_individual=False):
    loss_version = loss_args['loss_version']
    loss_type = loss_args['loss_type']
    recon_weight = float(loss_args['recon_weight'])

    def loss_func(diffusion, model, batch, t, labels):
        if loss_version=='standard':
            loss = diffusion.p_losses(model, batch, t, loss_type=loss_type, labels=labels)
            if get_individual:
                return loss, {'loss': loss.item()}
            else:
                return loss
            return
        elif loss_version=='reconstruction':
            #print(batch.shape, t.shape)
            loss1 = diffusion.p_losses(model, batch, t, loss_type=loss_type, labels=labels)
            #recon_loss
            z_0 = diffusion.q_sample(x_start=batch, t=torch.zeros(batch.shape[0], device=batch.device, dtype=int))
            norm = diffusion.sqrt_one_minus_alphas_cumprod[0] if loss_args['recon_norm']=='Sigma' else 1.0
            diff = (z_0 - batch)/norm
            loss_recon = recon_weight*torch.mean((diff)**2)
            if get_individual:
                return loss1+loss_recon, {'diffusion_loss': loss1.item(), 'recon_loss': loss_recon.item(), 'loss': loss1.item()+loss_recon.item()}
            else:
                return loss1 + loss_recon
        else:
            raise NotImplementedError
    return loss_func



if __name__=='__main__':
    import schedules
    import models
    betas = schedules.log_custom_schedule(1000)
    diff = Diffusion(betas)

    sdict = torch.load('../../../checkpoints_test/Run_4-26/checkpoint_200000.pt', map_location='cpu')
    model = models.UnetExplicitConditional(**sdict['model_kwargs'])
    B, Nx = 3, 256
    test_img = torch.randn(size=(B, 1, Nx, Nx))
    ytest = torch.rand((2))
    vlbloss = diff.compute_vlb(model, test_img, ytest, device='cpu')
    print(3)
