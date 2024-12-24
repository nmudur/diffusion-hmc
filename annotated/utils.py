import time
import os
import torch
import torchvision
import numpy as np
from numpy.fft import *
from torchvision.utils import save_image

import torch.nn.functional as F
import torchvision.transforms as T
from tqdm.auto import tqdm
from torch import nn

import hf_diffusion

from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict

def get_fieldidx_for_index(idx):
    return np.arange(idx*15, (idx+1)*15, dtype=int)

def preprocess_resize_field(path, Nx):
    field = np.load(path)
    tfields = torch.tensor(np.log10(field)) #take log field
    resizedfields = T.Resize([Nx, Nx], interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)(tfields)
    return resizedfields


def save_fields_as_png(ftensor, directory, abbrev):
    if os.path.isdir(directory):
        print("Error: directory already exists")
    else:
        os.mkdir(directory)
    for i in range(ftensor.shape[0]):
        save_image(ftensor[i], os.path.join(directory, abbrev+str(i).zfill(5)+'.jpeg'))
    return


def augment_fields(fields, transformation_list):
    #fields: numpy array
    assert len(fields.shape)==4 #BCHW
    tfields = torch.tensor(fields)
    tflist = [tfields]
    for transform in transformation_list:
        tflist.append(transform(tfields))
    return torch.cat(tflist).numpy()

def power(x,x2=None):
    """(Adapted from Carol's repo)
    Parameters
    ---------------------
    x: the input field, in torch tensor

    x2: the second field for cross correlations, if set None, then just compute the auto-correlation of x

    ---------------------
    Compute power spectra of input fields
    Each field should have batch and channel dimensions followed by spatial
    dimensions. Powers are summed over channels, and averaged over batches.

    Power is not normalized. Wavevectors are in unit of the fundamental
    frequency of the input.

    source code adapted from
    https://nbodykit.readthedocs.io/en/latest/_modules/nbodykit/algorithms/fftpower.html#FFTBase
    """
    signal_ndim = x.dim() - 2
    signal_size = x.shape[-signal_ndim:]

    kmax = min(s for s in signal_size) // 2
    even = x.shape[-1] % 2 == 0

    x = torch.fft.rfftn(x, s=signal_size)  # new version broke BC
    if x2 is None:
        x2 = x
    else:
        x2 = torch.fft.rfftn(x2, s=signal_size)
    P = x * x2.conj()

    P = P.mean(dim=0)
    P = P.sum(dim=0)

    del x, x2

    k = [torch.arange(d, dtype=torch.float32, device=P.device)
         for d in P.shape]
    k = [j - len(j) * (j > len(j) // 2) for j in k[:-1]] + [k[-1]]
    k = torch.meshgrid(*k,indexing="ij")
    k = torch.stack(k, dim=0)
    k = k.norm(p=2, dim=0)

    N = torch.full_like(P, 2, dtype=torch.int32)
    N[..., 0] = 1
    if even:
        N[..., -1] = 1

    k = k.flatten().real
    P = P.flatten().real
    N = N.flatten().real

    kbin = k.ceil().to(torch.int32)
    k = torch.bincount(kbin, weights=k * N)
    P = torch.bincount(kbin, weights=P * N)
    N = torch.bincount(kbin, weights=N).round().to(torch.int32)
    del kbin

    # drop k=0 mode and cut at kmax (smallest Nyquist)
    k = k[1:1+kmax]
    P = P[1:1+kmax]
    N = N[1:1+kmax]

    k /= N
    P /= N

    return k, P, N


def calc_1dps_img2d(kvals, img, to_plot=True, smoothed=0.5, normalize=True, box_size=25.0):
    Nx = img.shape[0]
    # k_conv = 2*np.pi/25 Do this outside
    if normalize:
        img = img/img.sum() # notebook: normalizes to compute overdensity pk
        timg = torch.tensor(img).unsqueeze(0).unsqueeze(0)
        assert len(timg.shape)==4
        k, pk, _ = power(timg)
        pk *= (box_size**2)
        # k *= k_conv
        return k.numpy(), pk.numpy()
    else:
        fft_zerocenter = fftshift(fft2(img)) #Earlier had /Nx**2
        impf = abs(fft_zerocenter) ** 2.0
        x, y = np.meshgrid(np.arange(Nx), np.arange(Nx))
        R  = np.sqrt((x-(Nx/2))**2+(y-(Nx/2))**2) #Aug
        filt = lambda r: impf[(R >= r - smoothed) & (R < r + smoothed)].mean()
        mean = np.vectorize(filt)(kvals)
        return mean


def get_normalization(fnorm):
    data_norm = np.load(fnorm, mmap_mode='r') 
    if 'Mstar' in fnorm:
        data_norm = np.log10(data_norm+1)
    else:
        data_norm = np.log10(data_norm)

    mean, std = np.mean(data_norm), np.std(data_norm, ddof=1)
    return mean, std

def extract_state_dict(sdict, ddp, use_ema):
    #This ONLY works without LDM rn.
    if use_ema:
        print('Loading EMA weights')
        state_dict = sdict['model_ema_state_dict']
    else:
        state_dict = sdict['model_state_dict']
    if ddp:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict


def load_model(sdpath, device, use_ema=False, ddp=False):
    '''
    returns: model, diff if not LDM model.
    '''
    sdict = torch.load(sdpath, map_location='cpu')  # earlier cpu
    if 'model_type' in sdict.keys():
        print(sdict['model_type'])
        if sdict['model_type'] == 'latentdiffusionmodule':
            # change path
            if 'ckpt' in sdict['decoder_config'] and (sdict['decoder_config']['ckpt'] is not None):
                sdict['decoder_config']['ckpt'] = '../diffusion-models-for-cosmological-fields/annotated/' + \
                                                  sdict['decoder_config']['ckpt']
            sampler = hf_diffusion.TimestepSampler(timesteps=len(sdict['diffusion_config']['diffusion_kwargs']['betas']), device=device, **sdict['sampler_args'])
            ldm = hf_diffusion.LatentDiffusionModule(sdict['encoder_config'], sdict['decoder_config'],
                                                     sdict['model_config'], sdict['diffusion_config'], sampler,
                                                     sdict['loss_kwargs'], device=device)
                                                    
            if 'ldm_state_dict' not in sdict.keys():
                ldm.load_state_dict(sdict['model_state_dict'])
            else:
                ldm.load_state_dict(sdict['ldm_state_dict'])
            ldm.eval()
            return ldm
        else:
            mkw = sdict['model_kwargs']
            if 'dim' not in sdict['model_kwargs'].keys():
                mkw['dim'] = mkw.pop('image_size')

            if sdict['model_type'] == 'baseline':
                model = hf_diffusion.Unet(**mkw)
            elif sdict['model_type'] == 'explicitconditional':
                model = hf_diffusion.UnetExplicitConditional(**mkw)
            else:
                raise NotImplementedError()
            state_dict = extract_state_dict(sdict, ddp, use_ema)
            model.to(device)
            model.load_state_dict(state_dict)

            betas = sdict['betas']
            diff = hf_diffusion.Diffusion(betas)
            print('Beta shape', betas.shape)
            model.eval()
            return model, diff


def get_samples_given_saved_dict(sdpath, numsamples, samplabels=None, device='cpu', sample_image_size=None,
        return_multiple_timesteps=True, cond_kwargs=None, noise_input=None, return_reverse_indices=None, use_ema=False, ddp=False, return_time=False):
    '''
    Args:
        sdpath: Checkpoint path
        numsamples: Nsamples
        samplabels: parameter for each sample. SAME shape as numsamples. Already normalized.
    Returns:
        samples: Also in x0 space i.e invtr(x0) has NOT been applied.
    '''
    assert numsamples == samplabels.shape[0], 'Conditioning params must be the same as the number of samples.'
    sdict = torch.load(sdpath, map_location='cpu') # earlier cpu
    print(sdict['model_type'])
    if sdict['model_type']=='latentdiffusionmodule':
        #change path
        ldm = load_model(sdpath, device)
        p_sample_args = {'shape': (numsamples, sdict['model_config']['model_kwargs']['channels'], sample_image_size, sample_image_size), 'return_multiple_timesteps': return_multiple_timesteps,
                'cond_kwargs': cond_kwargs, 'noise': noise_input, 'return_specific_samples': return_reverse_indices}
        samples = ldm.sample(samplabels, p_sample_args)
        samples = samples.detach().cpu().numpy()
    else: #default
        assert sdict['model_type']== 'baseline' or sdict['model_type']== 'explicitconditional'
        model, diff = load_model(sdpath, device, use_ema, ddp)
        #output size
        if sample_image_size is not None:
            image_size=sample_image_size
        elif 'image_size' in sdict['model_kwargs'].keys():
            image_size = sdict['data']['image_size']
        else:
            image_size = sdict['model_kwargs']['dim'] #for all the September runs where dim=image_size
        t0 = time.time()
        samples = diff.p_sample_loop_mem_efficient(model, shape=(numsamples, sdict['model_kwargs']['channels'], image_size, image_size),
                       labels=samplabels, return_multiple_timesteps=return_multiple_timesteps,
                       cond_kwargs=cond_kwargs, noise=noise_input, return_specific_samples=return_reverse_indices)
        t1 = time.time()
    if return_time:
        return samples, (t1-t0)
    else:
        return samples


def denoise_images(noisy_images, trsigma, sdict, transformations=[nn.Identity(), nn.Identity()], num_samples=1, device='cpu'):
    '''
    :param noisy_image: Image+Noise B*C*Nx*Nx
    :param sigma: Noises added to Image in the transformed space
        trsigma = sigma_data * 2 / (RANGE_MAX - RANGE_MIN)

    :param sdict: Saved checkpoint to use model
    :param transformations: whether to transform before and after
    :param num_samples: 1
    :return: Two lists of the denoised images, the denoised imgaes with the invtr applied. Shape: num_samples x C x H x W
    '''
    B = noisy_images.shape[0]
    tr, invtr = transformations
    trnoisyimages = tr(torch.tensor(noisy_images, dtype=torch.float32))

    #load model
    sdict = torch.load(sdict, map_location=device)
    diff = hf_diffusion.Diffusion(torch.tensor(sdict['betas'].cpu()))
    model = hf_diffusion.Unet(**sdict['model_kwargs'])
    model.load_state_dict(sdict['model_state_dict'])
    model.eval()
    model.to(device)

    #find what timestep the noise levels map to
    tn_index = np.zeros(B, dtype=int)
    for b in range(B):
        mindiff = np.abs(diff.sqrt_one_minus_alphas_cumprod.numpy() - trsigma[b])
        tn_index[b] = np.where(mindiff == np.min(mindiff))[0]
    print('Timesteps corr to noise', tn_index)

    #need a loop since different numbers of steps :(
    denoised_images = []
    invtr_denoised_images = []
    for b in range(B):
        if num_samples==1:
            trnoisyimg = torch.unsqueeze(trnoisyimages[b], 0)
        else:
            trnoisyimg = trnoisyimages[b].repeat(num_samples, 1, 1, 1)
        t_input = torch.tensor(tn_index[b])
        t_reversed = np.flip(np.arange(0, int(t_input.numpy())))
        imgs_denoising = []
        img_tminus1 = trnoisyimg.to(device)
        print(f'Image {b}')
        for t in t_reversed:
            img_tminus1 = diff.p_sample(model, img_tminus1, t=torch.tensor(np.array([t]*num_samples), device=device))
            imgs_denoising.append(img_tminus1.cpu().numpy()) #images T-1 to 0 given the noisy image
        denoised_images.append(imgs_denoising[-1]) #num_samples x C x H x W
        invtr_denoised_images.append(invtr(torch.tensor(imgs_denoising[-1])).numpy())

    return denoised_images, invtr_denoised_images #num_samples x C x H x W
