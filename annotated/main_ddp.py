import os
import sys
import wandb
import yaml
import copy
import torch
import datetime
import numpy as np
import shutil
import argparse

import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, CenterCrop, Resize
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
from functools import partial
import pickle

from hf_diffusion import *
from main_helper import *
import hf_diffusion
import utils
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    with open(args.config, 'r') as stream:
        config_dict = yaml.safe_load(stream) #Check that multiple processes reading this is ok

    # Setup DDP:
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    if 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])

    dist.init_process_group(backend="nccl", rank=args.rank, world_size=args.world_size, store=None)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."  # world size is Nnodes*Ngpus/node
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()  # if we have 2 gpus, rank 0 and 1 will be on gpu 0, rank 2 and 3 will be on gpu 1
    seed = config_dict['global_seed'] * dist.get_world_size() + rank #Seed for this process
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup Diffusion variables ############
    timesteps = int(config_dict['diffusion']['timesteps'])
    max_iterations = int(config_dict['train']['max_iterations'])
    beta_schedule_key = config_dict['diffusion']['beta_schedule']
    DATAPATH = config_dict['data']['path']

    LR = float(config_dict['train']['learning_rate'])
    beta_func = getattr(hf_diffusion, beta_schedule_key)
    beta_args = config_dict['diffusion']['schedule_args']
    beta_schedule = partial(beta_func, **beta_args)
    betas = beta_schedule(timesteps=timesteps)
    diffusion = Diffusion(betas)  # TODO: Check that this ok. this will exist on each process. no synchronization is needed since it's initialized the same way everywhere and never updated. The model is anyway passed as an argument.
    config_dict = get_defaults_config_dict(config_dict)
    CONDITIONAL = bool(config_dict['data']['conditional'])

    #Sampler + Dataset ############
    #Example codes do seem to read in the dataset on each process so...
    imgmemmap = np.load(DATAPATH, mmap_mode='r')
    image_size = imgmemmap[0].shape[-1]
    if len(imgmemmap.shape) == 3:
        channels = 1
    else:
        assert len(imgmemmap.shape) == 4
        channels = imgmemmap.shape[1]
    config_dict['data'].update({'image_size': image_size, 'channels': channels})
    ### get validation data
    VALFLG = False
    if 'valdata_path' in config_dict['data'].keys():
        VALFLG = True
        VALDATAPATH = config_dict['data']['valdata_path']
        valmemmap = np.load(VALDATAPATH, mmap_mode='r')
        assert image_size == valmemmap['fields'].shape[-1]

    # retrieve data transformations
    transforms, inverse_transforms = get_data_transforms(None, config_dict) #Removed imgmemmap from here since you're no longer pegging any transforms to the train dataset.
    if not CONDITIONAL or config_dict['data']['labels_subset']=='all':
        labels_subset = None
    else:
        labels_subset = np.array([int(elem) for elem in config_dict['data']['labels_subset']])
    traindata = CustomTensorDataset(imgmemmap, transforms=transforms,
                                    labels_path=config_dict['data']['labels'] if CONDITIONAL else None,
                                    labels_subset=labels_subset,
                                    labels_normalize=bool(config_dict['data']['labels_normalize']))
    datasampler = DistributedSampler(traindata, num_replicas=dist.get_world_size(), rank=rank, shuffle=True,
                                     seed=config_dict['global_seed'])
    traindataloader = DataLoader(traindata, batch_size=int(args.global_batch_size // dist.get_world_size()),
                                 shuffle=False, sampler=datasampler,
                                 pin_memory=True, num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))  # Is num_workers=0 ok?
    if VALFLG:
        valdata = CustomTensorDataset(valmemmap, transforms=transforms, labels_subset=labels_subset,
                                      labels_normalize=bool(config_dict['data']['labels_normalize']),
                                      subset_type='validation')
        #no data sampler here?
        valdataloader = DataLoader(valdata, batch_size=int(args.global_batch_size//dist.get_world_size()), shuffle=True)


    # Setup Wandb ############
    if rank == 0:
        dt = datetime.datetime.now()
        name = f'Run_{dt.month}-{dt.day}_{dt.hour}-{dt.minute}'
        print(name)
        if not args.debug:
            if 'resume_id' in config_dict.keys():  # Keep resume id ONLY when you want to continue the same run in wandb
                wandb.init(project='diffmod_cosmo0', job_type='conditional' if CONDITIONAL else 'unconditional',
                           config=config_dict, name=name, id=config_dict['resume_id'], resume='must')
            else:
                wandb.init(project='diffmod_cosmo0', job_type='conditional' if CONDITIONAL else 'unconditional',
                           config=config_dict, name=name)

        #Setup experiment folder ############ 
        resdir = f'results/samples_exps/{name}/'
        os.mkdir(resdir)
        shutil.copy(args.config, resdir + args.config[args.config.rindex('/') + 1:])

    #Setup Model ############
    model_kwargs = get_default_model_kwargs(image_size, channels, config_dict)
    if config_dict['architecture']['model'] == 'baseline':
        model = Unet(**model_kwargs)
    elif config_dict['architecture']['model'] == 'explicitconditional':
        model = UnetExplicitConditional(**model_kwargs)
    else:
        raise NotImplementedError()

    if 'resume_name' in config_dict:
        sdpath = 'results/samples_exps/'+config_dict['resume_name']+'/'+config_dict['resume_ckp']
        sdict = torch.load(sdpath, map_location='cpu')
        print(f'Loading model weights from {sdpath}')
        state_dict = utils.extract_state_dict(sdict, ddp=True, use_ema=False) #removes the module. prefix
        model.load_state_dict(state_dict)
    model = DDP(model.to(device), device_ids=[rank])


    #Setup optimizer, scheduler, ema, sampler ############
    if config_dict['train']['optimizer']=='Adam':
        if 'resume_id' in config_dict and 'optimizer_reset' not in config_dict['train']['optimizer']:
            optimizer = Adam(model.parameters())
            optimizer.load_state_dict(sdict['optimizer_state_dict'])
        else:
            print('Reinitializing optimizer')
            optimizer = Adam(model.parameters(), lr=LR)
    elif  config_dict['train']['optimizer']=='SGD':
        if 'resume_id' in config_dict:
            optimizer = SGD(model.parameters())
            optimizer.load_state_dict(sdict['optimizer_state_dict'])
        else:
            optimizer = SGD(model.parameters(), lr=LR)
    else:
        raise NotImplementedError()

    if 'ema' in config_dict['train']:
        model_ema = copy.deepcopy(model).to(device)
        model_ema.eval()
        model_ema.requires_grad_(False)
        ema_handler = EMAUpdater(model_ema, config_dict['train']['ema']['start_epoch'], config_dict['train']['ema']['update_interval'], config_dict['train']['ema']['decay'])
    else:
        ema_handler = None

    # get scheduler
    scheduler = get_scheduler(config_dict, optimizer)
    # get sampler type
    sampler = TimestepSampler(timesteps=timesteps, device=device, **config_dict['diffusion']['sampler_args'])

    # Pretraining Prep ############
    model.train()
    misc_save_params = {'model_type': config_dict['architecture']['model'], "model_kwargs": model_kwargs,
                        "schedule": beta_schedule_key, "schedule_args": beta_args, "betas": betas}
    start_itn = 0 if 'resume_id' not in config_dict.keys() else sdict['itn']
    start_epoch = 0 if 'resume_id' not in config_dict.keys() else sdict['epoch']
    loss_args = config_dict['train']['loss_args']

    ############# Actual Train Loop ############
    itn = start_itn
    epoch = start_epoch
    loss_spike_flg = 0
    print('Start Epoch', start_epoch)
    lversi = loss_args['loss_version']
    print(f'Using {lversi} loss')
    loss_type = loss_args['loss_type']
    loss_func_verbose = hf_diffusion.eval_loss_state(loss_args, get_individual=True)
    while itn < max_iterations:  # Epochs: number of full passes over the dataset. Earlier an epoch loop. Now replaced with iterations.
        if rank==0:
            print('Epoch: ', epoch)
        datasampler.set_epoch(epoch)  # Why is this needed?  to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be always used.
        # train loop
        for step, batch in enumerate(traindataloader):  # Step: each pass over a batch
            model.train()
            optimizer.zero_grad()  # prevents gradient accumulation
            if CONDITIONAL:
                batch, labels = batch
                labels = labels.to(device)
            batch = batch.to(device)
            batch_size = batch.shape[0]

            # Sample t
            t = sampler.get_timesteps(batch_size, itn)  # [0, T-1]

            loss, losses = loss_func_verbose(diffusion, model, batch, t,
                                             labels if CONDITIONAL else None)  # diffusion.p_losses(model, batch, t, loss_type=loss_type, labels=labels if conditional else None)
            loss.backward()
            optimizer.step()

            if ema_handler is not None:
                ema_handler.update(model, epoch)

            if sampler.type == 'loss_aware':
                with torch.no_grad():
                    #Check if this behavior is correct with multiple GPUs.
                    loss_timewise = diffusion.timewise_loss(model, batch, t, loss_type=loss_type,
                                                            labels=labels if CONDITIONAL else None)
                    sampler.update_history(t, loss_timewise)
            if (step % 100 == 0) and (rank== 0):
                print("Loss=", losses['loss'], flush=True)
            if not args.debug and (rank== 0):
                logvars = {"iteration": itn, "epoch": epoch}
                logvars.update(losses)
                wandb.log(logvars)
            if loss > 0.1 and itn > 300 and (loss_spike_flg < 2) and (rank== 0):
                badbdict = {'batch': batch.detach().cpu().numpy(), 'itn': itn, 't': t.detach().cpu().numpy(),
                            'loss': loss}
                pickle.dump(badbdict, open(resdir + f'largeloss_{itn}.pkl', 'wb'))
                loss_spike_flg += 1

            if (rank== 0):
                if (itn % 4000 == 0):
                    misc_save_params.update({'epoch': epoch, 'itn': itn, 'model_state_dict': model.state_dict(),
                                             'optimizer_state_dict': optimizer.state_dict()})
                    if ema_handler is not None and ema_handler.state:
                        misc_save_params.update({'model_ema_state_dict': ema_handler.model_ema.state_dict()})
                    torch.save(misc_save_params, resdir + f'checkpoint_{itn}.pt')
                    # save samples
                    if (itn > 10000):  # added >10k for the Nx=256 case
                        NSAMP = 5
                        samplabels = labels[:NSAMP, :]
                        samples = diffusion.p_sample_loop_mem_efficient(model, shape=(
                            samplabels.shape[0], misc_save_params["model_kwargs"]["channels"], image_size, image_size),
                                                                        labels=samplabels, return_multiple_timesteps=False,
                                                                        cond_kwargs=None, noise=None)
                        invtsamples = inverse_transforms(torch.tensor(samples)).detach().cpu().numpy()
                        np.savez(resdir + f'samples_{itn}.npz', params_normed=samplabels.detach().cpu().numpy(),
                                 samples=invtsamples)
            itn += 1

        # "val" part (epoch level mods)
        if VALFLG:
            with torch.no_grad():
                model.eval()
                lossvaltotal = 0
                nvalpts = 0
                for step, batch in enumerate(valdataloader):
                    if CONDITIONAL:
                        batch, labels = batch
                        labels = labels.to(device)
                    batch_size = batch.shape[0]
                    batch = batch.to(device)

                    t = torch.full((batch_size,), 0, device=device)  # [0, T-1]
                    loss, losses = loss_func_verbose(diffusion, model, batch, t,
                                                     labels if CONDITIONAL else None)  # diffusion.p_losses(model, batch, t, loss_type=loss_type, labels=labels if conditional else None)
                    lossvaltotal += loss
                    nvalpts += batch_size
                lossvaltotal /= nvalpts

                print('Validation Loss', lossvaltotal.item(), flush=True)
                if (scheduler is not None) and (type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau) and (itn > 16000):
                    scheduler.step(lossvaltotal)

        if (scheduler is not None): #Only steps after one full pass over the training dataloader
            scheduler.step()
        epoch += 1

    ############# End of Train Loop ############
    # Save final model and optimizer state
    if rank==0:
        misc_save_params.update(
            {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'itn': itn,
             'epoch': epoch})
        if ema_handler is not None and ema_handler.state:
            misc_save_params.update({'model_ema_state_dict': ema_handler.model_ema.state_dict()})

        torch.save(misc_save_params, resdir + 'model.pt')

        if CONDITIONAL:
            NSAMP = 20
            rng = np.random.default_rng()
            lidx = rng.choice(1000, 5, replace=False)
            samplabels = np.zeros((NSAMP, model_kwargs['conditional_dim']), dtype=np.float32)
            for si in range(NSAMP):
                samplabels[si, :] = traindata.labels[lidx[si // 4], labels_subset]
            samplabels = torch.from_numpy(samplabels)
            samplabels = samplabels.to(device)
            samples = diffusion.sample(model, image_size=image_size, batch_size=NSAMP, channels=channels, labels=samplabels,
                                       return_all_timesteps=False)
            invtsamples = inverse_transforms(torch.tensor(samples)).detach().cpu().numpy()
            np.savez(resdir + f'samples_final.npz', params_normed=samplabels.detach().cpu().numpy(), samples=invtsamples)

        else:
            NSAMP = 10
            samples = diffusion.sample(model, image_size=image_size, batch_size=NSAMP, channels=channels,
                                       return_all_timesteps=False)
            invtsamples = inverse_transforms(torch.tensor(samples)).numpy()
            np.savez(resdir + f'samples_final.npz', samples=invtsamples)

    dist.destroy_process_group()
    return

if __name__ == '__main__':
    #DDP stuff
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--global-batch-size", type=int, default=40) #20x2
    args = parser.parse_args()
    main(args)
