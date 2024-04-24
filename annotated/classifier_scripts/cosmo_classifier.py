import os
import numpy as np
import torch
import sys
import pickle
import yaml
import wandb
import shutil
import datetime

#import matplotlib.pyplot as plt
from kymatio.torch import Scattering2D
from torch import nn
from astropy.io import fits
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


sys.path.append(os.path.join(os.getcwd(), '../'))
import hf_diffusion as hfd
import classifier_architecture
from classifier_architecture import ScModel, ScModelTwoLayer
torch.set_default_dtype(torch.float)

DEBUG=False
dt = datetime.datetime.now()
name = f'WSTClassifierRun_{dt.month}-{dt.day}_{dt.hour}-{dt.minute}'
diffusion_dir = '/n/holylfs05/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/diffusion-models-for-cosmological-fields/'

resdir = os.path.join(diffusion_dir, f'annotated/results/noisyclassifier/{name}/')

print(name)

'''
class ScModel(nn.Module):
    def __init__(self, J, L, max_order, shape, len_coeff, output_activation=nn.Identity(), device='cpu'):
        super().__init__()
        self.J = J
        self.L = L
        self.max_ord = max_order
        self.shape = shape
        self.S = Scattering2D(J=J, L=L, max_order=max_order, shape=shape)
        if device=='cuda':
            self.S = self.S.cuda()
        self.linear = nn.Linear(len_coeff, 6)
        self.out_act = output_activation

    def forward(self, x):
        #print(x.device, self.S.device)
        coeffs = self.S(x)
        coeffs_spav = coeffs.mean(dim=(-2, -1), dtype=torch.float)
        out = self.linear(coeffs_spav)
        return self.out_act(out)
'''


def main1():
    # Basic model to check whether a wavelet scattering transform+MLP can regress against fields' parameters
    device = 'cpu'
    NTRAINFIELDS = 6000
    fielddir = os.path.join(os.getcwd(), '../CMD/data_processed/')
    DATAPATH = os.path.join(fielddir, 'LogMaps_Mcdm_IllustrisTNG_LH_z=0.00_Nx64_train.npy')
    labels_path = os.path.join(fielddir, 'params_IllustrisTNG.txt')
    #y_train = np.vstack([params[:NTRAINFIELDS,...]]*6)

    if 'fits' in DATAPATH:
        with fits.open(DATAPATH) as hdul:
            imgmemmap = hdul[0].data
    else:
        imgmemmap = np.load(DATAPATH, mmap_mode='r')
    image_size = imgmemmap[0].shape[-1]
    NTRAIN = imgmemmap.shape[0]
    BATCH_SIZE = 40
    epochs = 4
    RANGE_MIN, RANGE_MAX = torch.tensor(imgmemmap.min()), torch.tensor(imgmemmap.max())
    transforms, inverse_transforms = hfd.get_minmax_transform(RANGE_MIN, RANGE_MAX)


    traindata = hfd.CustomTensorDataset(imgmemmap, transforms=transforms, labels_path=labels_path)
    dataloader = hfd.DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True)

    J = 4
    L = 8
    max_order = 2
    S = Scattering2D(J=J, L=L, max_order=max_order, shape=(image_size, image_size))
    lencoeff = S(traindata[0][0]).shape
    lencoeff = lencoeff[1]

    model = ScModel(J=J, L=L, max_order=max_order, shape=(image_size, image_size), len_coeff=lencoeff)
    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch, labels = batch
            batch = torch.squeeze(batch, 1) #removing channel dim
            batch = batch.to(device)
            labels =labels.to(device)
            labels_pred = model(batch)
            loss = F.mse_loss(labels_pred, labels)
            if step % 20 == 0:
                print("Loss:", loss.item())
            loss.backward()
            optimizer.step()
    return

def examine_wavelets_with_noise(Numimages, Noise_timesteps, betas):
    # Basic model to check whether a wavelet scattering transform+MLP can regress against the fields' parameters
    fielddir = os.path.join(os.getcwd(), '../../data_processed/')
    DATAPATH = os.path.join(fielddir, 'LogMaps_Mcdm_IllustrisTNG_LH_z=0.00_Nx64_train.npy')
    imgmemmap = np.load(DATAPATH, mmap_mode='r')
    image_size = imgmemmap[0].shape[-1]

    RANGE_MIN, RANGE_MAX = torch.tensor(imgmemmap.min()), torch.tensor(imgmemmap.max())
    transforms, inverse_transforms = hfd.get_minmax_transform(RANGE_MIN, RANGE_MAX)

    traindata = hfd.CustomTensorDataset(imgmemmap, transforms=transforms)

    J = 4
    L = 8
    max_order = 1
    S = Scattering2D(J=J, L=L, max_order=max_order, shape=(image_size, image_size))
    lencoeff = S(traindata[0][0]).shape
    lencoeff = lencoeff[1]

    model = ScModel(J=J, L=L, max_order=max_order, shape=(image_size, image_size), len_coeff=lencoeff)
    rng = np.random.default_rng(seed=23)
    imgidx = rng.choice(len(traindata), Numimages, replace=False)
    trimgs = imgmemmap[imgidx]

    #diff
    diff = hfd.Diffusion(betas)
    wst_for_each_n = []
    for t in Noise_timesteps:
        time = torch.full((trimgs.shape[0],), t, dtype=int)
        noisybatch = diff.q_sample(torch.tensor(np.expand_dims(trimgs, 1)), time)
        wst_for_each_n.append(model.S(noisybatch.squeeze(1)).mean(dim=(-2, -1)))
    return diff.sqrt_one_minus_alphas_cumprod[Noise_timesteps], wst_for_each_n


def main2(config_dict):
    # Basic model to check whether a wavelet scattering transform+MLP can regress against fields' parameters
    #get_dataloaders
    ### get training data and image_size
    DATAPATH = config_dict['data']['path']
    BATCH_SIZE = config_dict['train']['batch_size']
    NEPOCHS = config_dict['train']['epochs']

    imgmemmap = np.load(DATAPATH, mmap_mode='r')
    labels_all = np.loadtxt(config_dict['data']['labels'], dtype=np.float32)
    params_subset = np.array([int(elem) for elem in config_dict['data']['labels_subset']])
    assert 'Log' not in DATAPATH

    randflips_t, _ = hfd.get_data_transforms(imgmemmap, config_dict)
    transforms = Compose([lambda t: torch.log10(t), randflips_t])

    #train
    trainfrac = 0.7
    NPARAMS_ALL = len(labels_all)
    print('Total number of parameters=', NPARAMS_ALL)
    Ntrain_params = int(trainfrac*NPARAMS_ALL)
    Ntrain_fields = 15*Ntrain_params
    trainx = imgmemmap[:Ntrain_fields]
    trainy = labels_all[:Ntrain_params]
    traindata = hfd.CustomTensorDataset(trainx, transforms=transforms,
                        labels_data=trainy, labels_subset=params_subset,
                        labels_normalize=bool(config_dict['data']['labels_normalize']))
    dataloader = DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True)


    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device='cpu'
    image_size = trainx.shape[-1]

    J = int(config_dict['wavelets']['J'])
    L = int(config_dict['wavelets']['L'])
    max_order = int(config_dict['wavelets']['Order'])
    S = Scattering2D(J=J, L=L, max_order=max_order, shape=(image_size, image_size))
    if device=='cuda':
        S = S.cuda()
        samp = traindata[0][0]
        samp = samp.to(device)
    lencoeff = S(samp).shape
    lencoeff = lencoeff[1]
    
    config_dict['wavelets'].update({'lencoeff': lencoeff})

    if not DEBUG: wandb.init(project='diffmod_classifier', config=config_dict, name=name)
    model = ScModel(J=J, L=L, max_order=max_order, shape=(image_size, image_size), len_coeff=lencoeff, device=device)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=float(config_dict['train']['learning_rate']))
    itn = 0
    for epoch in range(NEPOCHS):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch, labels = batch
            batch = torch.squeeze(batch, 1) #removing channel dim
            batch = batch.to(device)
            labels = labels.to(device)
            #print(batch.device)
            labels_pred = model(batch)[:, params_subset]
            loss = F.mse_loss(labels_pred, labels)
            if step % 20 == 0:
                print("Loss:", loss.item())
            #print(batch.dtype, labels.dtype, labels_pred.dtype)
            loss.backward()
            optimizer.step()
            if not DEBUG:  wandb.log({"loss": loss.item(), "iter": itn, "epoch": epoch})

            if itn % 4000 == 0:
                misc_save_params = {'epoch': epoch, 'itn': itn, 'model_state_dict': model.state_dict(),
                                         'optimizer_state_dict': optimizer.state_dict(), 'wavelets': config_dict['wavelets']}
                torch.save(misc_save_params, resdir + f'checkpoint_{itn}.pt')
            itn += 1

        #normalize back to correct range
        minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])[params_subset]
        maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])[params_subset]

        labels_pred_renorm = labels_pred.detach().cpu().numpy() *(maximum - minimum) + minimum
        labels_truth_renorm = labels.detach().cpu().numpy()*(maximum - minimum) + minimum

        mafr, mfr = 100*np.mean(np.abs((labels_pred_renorm - labels_truth_renorm)/labels_truth_renorm)), 100*np.mean((labels_pred_renorm - labels_truth_renorm)/labels_truth_renorm)
        if not DEBUG: wandb.log({"loss": loss.item(), "Mean Frac %age Err": mfr, "Mean Abs %age Err": mafr,  "iter": itn, "epoch": epoch})
    return


def main3(config_dict):
    # Model to check whether a wavelet scattering transform+MLP can regress against fields' parameters
    # get_dataloaders
    ### get training data and image_size
    DATAPATH = config_dict['data']['path']
    BATCH_SIZE = config_dict['train']['batch_size']
    NEPOCHS = config_dict['train']['epochs']

    imgmemmap = np.load(DATAPATH, mmap_mode='r')
    labels_all = np.loadtxt(config_dict['data']['labels'], dtype=np.float32)
    params_subset = np.array([int(elem) for elem in config_dict['data']['labels_subset']])
    assert 'Log' not in DATAPATH

    randflips_t, _ = hfd.get_data_transforms(imgmemmap, config_dict)
    transforms = Compose([lambda t: torch.log10(t), randflips_t])

    # train
    trainfrac = 0.7
    NPARAMS_ALL = len(labels_all)
    print('Total number of parameters=', NPARAMS_ALL)
    Ntrain_params = int(trainfrac * NPARAMS_ALL)
    Ntrain_fields = 15 * Ntrain_params
    trainx = imgmemmap[:Ntrain_fields]
    trainy = labels_all[:Ntrain_params]
    traindata = hfd.CustomTensorDataset(trainx, transforms=transforms,
                                        labels_data=trainy, labels_subset=params_subset,
                                        labels_normalize=bool(config_dict['data']['labels_normalize']))
    dataloader = DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    image_size = trainx.shape[-1]

    #model details
    ##wavelets
    J = int(config_dict['wavelets']['J'])
    L = int(config_dict['wavelets']['L'])
    max_order = int(config_dict['wavelets']['Order'])
    S = Scattering2D(J=J, L=L, max_order=max_order, shape=(image_size, image_size))
    if device == 'cuda':
        S = S.cuda()
    samp = traindata[0][0]
    samp = samp.to(device)
    lencoeff = S(samp).shape
    lencoeff = lencoeff[1]

    config_dict['wavelets'].update({'lencoeff': lencoeff})

    ##MLP details
    skip = config_dict['model']['skip']
    with_errors = config_dict['model']['with_errors']
    if not DEBUG: wandb.init(project='diffmod_classifier', config=config_dict, name=name)
    model = getattr(classifier_architecture, config_dict['model']['name'])(J=J, L=L, max_order=max_order,
            shape=(image_size, image_size), len_coeff=lencoeff, device=device, skip=skip, with_error=with_errors)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=float(config_dict['train']['learning_rate']))
    itn = 0
    for epoch in range(NEPOCHS):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch, labels = batch
            batch = torch.squeeze(batch, 1)  # removing channel dim
            batch = batch.to(device)
            labels = labels.to(device)
            # print(batch.device)
            labels_pred = model(batch)[:, params_subset]
            sigma_pred = model(batch)[:, 6+params_subset]
            loss1 = F.mse_loss(labels_pred, labels, reduction='none')
            loss1 = torch.log(loss1.sum(dim=0)).sum()
            loss2 = F.mse_loss((labels_pred - labels)**2, sigma_pred**2, reduction='none')
            loss2 = torch.log(loss2.sum(dim=0)).sum()
            loss = loss1 + loss2

            #loss = F.mse_loss(labels_pred, labels)
            if step % 20 == 0:
                print("Loss:", loss.item())
            # print(batch.dtype, labels.dtype, labels_pred.dtype)
            loss.backward()
            optimizer.step()
            if not DEBUG:  wandb.log({"loss": loss.item(), "iter": itn, "epoch": epoch})

            if itn % 4000 == 0:
                misc_save_params = {'epoch': epoch, 'itn': itn, 'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(), 'wavelets': config_dict['wavelets']}
                torch.save(misc_save_params, resdir + f'checkpoint_{itn}.pt')
            itn += 1

        # normalize back to correct range
        minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])[params_subset]
        maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])[params_subset]

        labels_pred_renorm = labels_pred.detach().cpu().numpy() * (maximum - minimum) + minimum
        labels_truth_renorm = labels.detach().cpu().numpy() * (maximum - minimum) + minimum
        sigma_pred_renorm = sigma_pred.detach().cpu().numpy() * (maximum-minimum)

        mafr, mfr = 100 * np.mean(np.abs((labels_pred_renorm - labels_truth_renorm) / labels_truth_renorm)), \
                    100 * np.mean(
            (labels_pred_renorm - labels_truth_renorm) / labels_truth_renorm)
        z_sc = (labels_pred_renorm - labels_truth_renorm) / sigma_pred_renorm
        if not DEBUG: wandb.log(
            {"loss": loss.item(), "loss1": loss1.item(), "loss2": loss2.item(), "Mean Frac %age Err": mfr,
             "Mean Abs %age Err": mafr, "iter": itn, "epoch": epoch, "z": z_sc})
    return


if __name__=='__main__':
    '''
    noiselevel, wavelets_list = examine_wavelets_with_noise(10, [0, 20, 50, 100, 400, 500, 800, 1000, 1500, 1999], torch.linspace(1e-4, 0.02, 2000))
    wavelets_list = np.stack(wavelets_list)
    #iws = [2, 3, 9, 17, 25, 33]
    iws = [2, 3, 4, 5, 30, 31, 32, 33]
    plt.figure()
    for wi in iws:
        plt.plot(noiselevel, wavelets_list.mean(1)[:, wi], '*-', label='WC#'+str(wi))
    plt.xlabel('Noise')
    plt.ylabel('WC')
    #plt.yscale('log')
    plt.legend()
    plt.show()
    print(10)
    '''
    with open(sys.argv[1], 'r') as stream:
        config_dict = yaml.safe_load(stream)
    SEED = int(config_dict['seed'])
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    os.mkdir(resdir)
    shutil.copy(sys.argv[1], resdir+sys.argv[1][sys.argv[1].rindex('/')+1:])

    main3(config_dict)
    print(3)



