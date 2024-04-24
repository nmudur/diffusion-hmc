import os
import sys
import wandb
import yaml
import copy
import torch
import datetime
import numpy as np
import astropy
import shutil

import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, CenterCrop, Resize
from astropy.io import fits
import matplotlib.pyplot as plt
from functools import partial
import pickle
import random


import hf_diffusion
from hf_diffusion import *
from main_helper import *
import utils
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

'''
Unit test to check that it is indeed getting a different set of timesteps within each iteration.
'''


with open(sys.argv[1], 'r') as stream:
    config_dict = yaml.safe_load(stream)

SEED = int(config_dict['seed'])
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEBUG= True

dt = datetime.datetime.now()
name = f'Run_{dt.month}-{dt.day}_{dt.hour}-{dt.minute}'
print(name)

timesteps = int(config_dict['diffusion']['timesteps'])
max_iterations = 10 #Fix Max Iterations: Irrelevant because it still does one complete train loop.
beta_schedule_key = config_dict['diffusion']['beta_schedule']
DATAPATH = config_dict['data']['path']
BATCH_SIZE = int(config_dict['train']['batch_size'])
LR = float(config_dict['train']['learning_rate'])

if torch.cuda.is_available(): 
    device = 'cuda'
else: 
    device='cpu'
print(device)

#moving to diffusion
beta_func = getattr(hf_diffusion, beta_schedule_key)
beta_args = config_dict['diffusion']['schedule_args']
beta_schedule = partial(beta_func, **beta_args)
betas = beta_schedule(timesteps=timesteps)
diffusion = Diffusion(betas)


def train(model, dataloader, optimizer, max_iter, loss_args={'loss_type': 'huber'}, sampler=None, conditional=False, resdir=None,
          misc_save_params=None, inverse_transforms=None, start_itn=0, start_epoch=0, scheduler=None, valdataloader=None, ema_handler=None):
    itn = start_itn
    epoch = start_epoch
    loss_spike_flg = 0
    print('Start Epoch', start_epoch)
    lversi = loss_args['loss_version']
    print(f'Using {lversi} loss')
    loss_type = loss_args['loss_type']
    loss_func_verbose = hf_diffusion.eval_loss_state(loss_args, get_individual=True)
    while itn < max_iter:  # Epochs: number of full passes over the dataset
        print('Epoch: ', epoch)

        #train loop
        for step, batch in enumerate(dataloader):  # Step: each pass over a batch
            model.train()
            optimizer.zero_grad() # prevents gradient accumulation
            if conditional:
                batch, labels = batch
                labels = labels.to(device)

            batch = batch.to(device)
            batch_size = batch.shape[0]

            # Sample t
            t = sampler.get_timesteps(batch_size, itn) #[0, T-1]
            print('Batch elems', batch[0, 0, 0, 0])
            print(f'Time at iter {itn} = ', t)

            itn += 1
        
        epoch+=1
    return itn, epoch

if __name__ == '__main__':
    config_dict = get_defaults_config_dict(config_dict)
    CONDITIONAL = bool(config_dict['data']['conditional'])

    ### get training data and image_size
    if 'fits' in DATAPATH:
        with fits.open(DATAPATH, memmap=True) as hdul:
            imgmemmap = hdul[0].data
    else:
        imgmemmap = np.load(DATAPATH, mmap_mode='r')

    image_size = imgmemmap[0].shape[-1]
    if len(imgmemmap.shape)==3:
        channels = 1
    else:
        assert len(imgmemmap.shape)==4
        channels = imgmemmap.shape[1]
    
    config_dict['data'].update({'image_size': image_size, 'channels': channels})
    NTRAIN = imgmemmap.shape[0]

    ### get validation data
    VALFLG = False
    if 'valdata_path' in config_dict['data'].keys():
        VALFLG = True
        VALDATAPATH = config_dict['data']['valdata_path']
        valmemmap = np.load(VALDATAPATH, mmap_mode='r')
        assert image_size==valmemmap['fields'].shape[-1]


    #wandb initialization
    if not DEBUG:
        if 'resume_id' in config_dict.keys(): #Keep resume id ONLY when you want to continue the same run in wandb
            wandb.init(project='diffmod_cosmo0', job_type='conditional' if CONDITIONAL else 'unconditional',
                       config=config_dict, name=name, id = config_dict['resume_id'], resume='must')
        else:
            wandb.init(project='diffmod_cosmo0', job_type='conditional' if CONDITIONAL else 'unconditional',
                       config=config_dict, name=name)


    #retrieve data transformations
    transforms, inverse_transforms = get_data_transforms(None, config_dict) #removed imgmemmap from input since you're no longer pegging to train range.
    
    if not CONDITIONAL or config_dict['data']['labels_subset']=='all':
        labels_subset = None
    else:
        labels_subset = np.array([int(elem) for elem in config_dict['data']['labels_subset']])

    traindata = CustomTensorDataset(imgmemmap, transforms=transforms, labels_path=config_dict['data']['labels'] if CONDITIONAL else None,
                    labels_subset=labels_subset, labels_normalize=bool(config_dict['data']['labels_normalize']))
    dataloader = DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True)
    if VALFLG:
        valdata = CustomTensorDataset(valmemmap, transforms=transforms, labels_subset=labels_subset, labels_normalize=bool(config_dict['data']['labels_normalize']),
                    subset_type='validation')
        valdataloader = DataLoader(valdata, batch_size=BATCH_SIZE, shuffle=True)


    ### get model
    model_kwargs = get_default_model_kwargs(image_size, channels, config_dict)
    if config_dict['architecture']['model'] == 'baseline':
        model = Unet(**model_kwargs)
    elif config_dict['architecture']['model'] == 'explicitconditional':
        model = UnetExplicitConditional(**model_kwargs)
    else:
        raise NotImplementedError()
    model.to(device)

    if 'resume_name' in config_dict:
        sdpath = 'results/samples_exps/'+config_dict['resume_name']+'/'+config_dict['resume_ckp']
        sdict = torch.load(sdpath, map_location='cpu')
        print(f'Loading model weights from {sdpath}')
        model.load_state_dict(sdict['model_state_dict'])


    if config_dict['train']['optimizer']=='Adam':
        if 'resume_id' in config_dict and 'optimizer_reset' not in config_dict['train']['optimizer']:
            print('Starting from saved optimizer state dict')
            optimizer = Adam(model.parameters())
            optimizer.load_state_dict(sdict['optimizer_state_dict'])
        else:
            print('Reinitializing optimizer')
            optimizer = Adam(model.parameters(), lr=LR)
    elif config_dict['train']['optimizer']=='SGD':
        if 'resume_id' in config_dict:
            print('Starting from saved optimizer state dict')
            optimizer = SGD(model.parameters())
            optimizer.load_state_dict(sdict['optimizer_state_dict'])
        else:
            optimizer = SGD(model.parameters(), lr=LR)
    else:
        raise NotImplementedError()

    if 'ema' in config_dict['train']:
        model_ema = copy.deepcopy(model)
        model_ema.eval()
        model_ema.requires_grad_(False)
        ema_handler = EMAUpdater(model_ema, config_dict['train']['ema']['start_epoch'], config_dict['train']['ema']['update_interval'], config_dict['train']['ema']['decay'])
    else:
        ema_handler = None

    # get scheduler
    scheduler = get_scheduler(config_dict, optimizer)

    #get sampler type
    sampler = TimestepSampler(timesteps=timesteps, device='cuda', **config_dict['diffusion']['sampler_args'])

    #Main Train Loop
    resdir = f'results/samples_exps/{name}/'
    os.mkdir(resdir)
    shutil.copy(sys.argv[1], resdir+sys.argv[1][sys.argv[1].rindex('/')+1:])
    misc_save_params = {'model_type': config_dict['architecture']['model'],
                "model_kwargs": model_kwargs,
                "schedule": beta_schedule_key, "schedule_args": beta_args, "betas": betas}
    start_itn = 0 if 'resume_id' not in config_dict.keys() else sdict['itn']
    start_epoch = 0 if 'resume_id' not in config_dict.keys() else sdict['epoch']
    end_itn, end_epoch = train(model, dataloader, optimizer, max_iter=max_iterations, loss_args=config_dict['train']['loss_args'], sampler=sampler, conditional=CONDITIONAL,
          resdir=resdir, misc_save_params=misc_save_params, inverse_transforms=inverse_transforms, start_itn=start_itn, start_epoch=start_epoch, scheduler=scheduler,
          valdataloader=valdataloader if VALFLG else None, ema_handler=ema_handler)
    
    
