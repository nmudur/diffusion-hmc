import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

def linear_beta_schedule(timesteps, beta_start, beta_end):
    #beta_start = 0.0001
    #beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def custom_linear_beta_schedule(timesteps):
    #beta_start = 0.0001
    #beta_end = 0.02
    assert timesteps == 1000
    usual_linear = torch.linspace(1e-4, 0.01, timesteps)
    first_500 = usual_linear[:500]
    next_500 = usual_linear[500:]*torch.linspace(1, 3, 500)
    return torch.cat([first_500, next_500])

def quadratic_beta_schedule(timesteps, beta_start, beta_end):
    #beta_start = 0.0001
    #beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start, beta_end):
    #beta_start = 0.0001
    #beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def cosine_beta_schedule(timesteps, s=0.015):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0, 0.999) #removed 1e-4 clip for small sqrt 1-alphascp, changed to 0.5 clip: bit weird rn cause the last two beta steps are identical, changed back to 0.999 on 9/16

def log_custom_schedule(timesteps=1000):
    #matches a linear schedule with 2000 timesteps in terms of the marginals for the first 500 timesteps and then the cumulative variance is A + C* log(D+time)
    #Latter half: 500-999
    d = -475
    c = (1 - 0.8528) / np.log((d + 999) / (d + 500))
    a = 1 - c * np.log((d + 999))
    latter = a + c * torch.log(d + torch.arange(500, 1000)) # 1 + (1-0.8528)/ log |(d + t)/(d+999)|

    # First half
    betas = torch.linspace(1e-4, 0.02, 2000) 
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)  # alpha_bar
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)# x_t = sqrt(alpha_bar) * x_0 + sqrt_one_minus_alphas_cumprod_t* std_normal_noise

    #New sqrt one minus alphas_cumprod
    custom_alphas_sqonemin = torch.cat([sqrt_one_minus_alphas_cumprod[:500], latter])
    custom_alphas_cumprod = 1 - custom_alphas_sqonemin ** 2
    custom_alphas = torch.cat(
        [torch.tensor([custom_alphas_cumprod[0]]), custom_alphas_cumprod[1:] / custom_alphas_cumprod[:-1]])
    custom_betas = 1 - custom_alphas
    return torch.clip(custom_betas, 0, 0.6)

def sigmoid_cumulativeposteriorsigma_schedule(timesteps=1000):
    '''
    SqrtOneMinusAlphasCP is Sigmoid
    :param timesteps:
    :return:
    '''
    #New sqrt one minus alphas_cumprod
    custom_alphas_sqonemin = torch.sigmoid(torch.linspace(-6, 6, timesteps))
    custom_alphas_cumprod = 1 - custom_alphas_sqonemin ** 2
    custom_alphas = torch.cat(
        [torch.tensor([custom_alphas_cumprod[0]]), custom_alphas_cumprod[1:] / custom_alphas_cumprod[:-1]])
    custom_betas = 1 - custom_alphas
    return custom_betas


def plot_variance_schedule_over_time(schedule, timesteps, name):
    betas = schedule(timesteps)
    plt.figure()
    plt.plot(np.linspace(0, timesteps, timesteps), betas.numpy())
    plt.title(f'{name}: Betas')
    plt.show()
    print('Betas start, end', betas[0], betas[-1])

    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    plt.figure()
    plt.plot(np.linspace(0, timesteps, timesteps), alphas_cumprod.numpy())
    plt.title(f'{name}: AlphaBars')
    plt.show()
    print('MinMax Sqrt alphabars', torch.sqrt(alphas_cumprod[0]), torch.sqrt(alphas_cumprod[-1]))
    return

#ported from https://github.com/openai/improved-diffusion/blob/783b6740edb79fdb7d063250db2c51cc9545dcd1/improved_diffusion/resample.py
class TimestepSampler():
    def __init__(self, sampler_type='uniform', history=None, nstart=None, timesteps=None, uniweight=None, device='cuda', custom_weights=None):
        self.type= sampler_type
        print('Sampler type', self.type)
        if self.type not in ['uniform', 'loss_aware', 'custom_weights']:
            raise NotImplementedError()
        if self.type=='loss_aware':
            self.sqloss_history = torch.ones((history, timesteps), device=device)*np.nan #L^2[b, t]
            self.nstart = nstart
            self.uniweight = 1/timesteps if uniweight is None else uniweight
            print('Nstart', nstart, type(nstart))
            print('History', history, type(history))
            print('Uniweight', self.uniweight, type(self.uniweight))
        if self.type=='custom_weights':
            self.custom_weights=custom_weights

        self.timesteps = timesteps
        self.uniform = 1/timesteps
        self.device=device
        self.history_per_term = torch.zeros(timesteps, device=device, dtype=int)
        self.not_enough_history = True

    def get_weights(self, batch_size, iteration):
        #relevant to custom_weights and loss_aware
        if self.type=='custom_weights':
            return self.custom_weights

        elif (iteration<self.nstart) or self.not_enough_history:
            return np.ones(self.timesteps)*self.uniform
        else:
            laweights = torch.sqrt(torch.mean(self.sqloss_history**2, dim=0))
            laweights /= laweights.sum()
            laweights *= (1-self.uniweight)
            laweights += self.uniweight/self.timesteps
            return laweights
        #fast way to evaluate / store the loss for different timesteps??
        #do you need a different sampler for storing history?

    def update_history(self, tl, loss_timewise):
        if self.not_enough_history:  # not-full loss history array
            for (t, tloss) in zip(tl, loss_timewise):
                if self.history_per_term[t] == self.sqloss_history.shape[0]:  # enough history
                    self.sqloss_history[:-1, t] = self.sqloss_history[1:, t]
                    self.sqloss_history[-1, t] = tloss
                else:
                    self.sqloss_history[self.history_per_term[t], t] = tloss
                    self.history_per_term[t] += 1
                    if self.history_per_term.min()==self.sqloss_history.shape[0]:
                        self.not_enough_history = False
                        print('Enough history for all')
        else:#enough history for all terms
            #test if this works fine
            self.sqloss_history[:-1, tl] = self.sqloss_history[1:, tl]
            self.sqloss_history[-1, tl] = loss_timewise
        return

    def get_timesteps(self, batch_size, iteration):
        if self.type=='uniform':
            return torch.randint(0, self.timesteps, (batch_size,), device=self.device).long() #[0, T-1]: indices of betas. t corresponds to noise level to be added to get to xt+1. i.e. 0->beta1
        elif self.type=='loss_aware':
            weights = self.get_weights(batch_size, iteration)
            return torch.tensor(list(torch.utils.data.WeightedRandomSampler(weights, batch_size, replacement=True)), device=self.device).long()
        elif self.type=='custom_weights':
            weights = self.get_weights(batch_size, iteration)
            return torch.tensor(list(torch.utils.data.WeightedRandomSampler(weights, batch_size, replacement=True)), device=self.device).long()
        else:
            raise NotImplementedError()


if __name__=='__main__':
    '''
    sch_baseline = partial(linear_beta_schedule, beta_start=1e-4, beta_end=2e-2)
    sch1 = partial(linear_beta_schedule, beta_start=1e-4, beta_end=1e-2)
    sch2 = partial(linear_beta_schedule, beta_start=1e-6, beta_end=2e-2)
    plot_variance_schedule_over_time(sch_baseline, 2000, name='Linear BL')
    plot_variance_schedule_over_time(sch1, 2000, name='Linear Smaller Beta_T')
    plot_variance_schedule_over_time(sch2, 2000, name='Linear Smaller Beta_0')
    #plot_variance_schedule_over_time(cosine_beta_schedule, 1000)
    '''

