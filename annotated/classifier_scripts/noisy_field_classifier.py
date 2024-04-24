import optuna
import yaml
import wandb
import datetime
import pickle
import shutil

import torch.nn as nn
from functools import partial
from astropy.io import fits
from torchvision.transforms import Compose
from torch.utils.data import DataLoader

from classifier_architecture import *

InfNets_dir = '/n/holylfs05/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/data/InferenceNets/'
diffusion_dir = '/n/holylfs05/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/diffusion-models-for-cosmological-fields/'
sys.path.append(InfNets_dir)
import IN_utils
#import data

sys.path.append('../')
import hf_diffusion
from hf_diffusion import Diffusion, get_data_transforms, CustomTensorDataset


with open(sys.argv[1], 'r') as stream:
    config_dict = yaml.safe_load(stream)

SEED = int(config_dict['seed'])
torch.manual_seed(SEED)
np.random.seed(SEED)

DEBUG= False


dt = datetime.datetime.now()
name = f'NoisyClassifierRun_{dt.month}-{dt.day}_{dt.hour}-{dt.minute}'
resdir = os.path.join(diffusion_dir, f'annotated/results/noisyclassifier/{name}/')

if not DEBUG: wandb.init(project='diffmod_classifier', config=config_dict, name=name)
print(name)

timesteps = int(config_dict['diffusion']['timesteps'])
epochs = int(config_dict['train']['epochs'])
beta_schedule_key = config_dict['diffusion']['beta_schedule']
DATAPATH = config_dict['data']['path']
BATCH_SIZE = int(config_dict['train']['batch_size'])

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


#Trains a classifier on the noisy fields p_phi(y|x, t)
def train(model, traindataloader, valdataloader, optimizer, scheduler, params_subset, epochs, sampler, misc_save_params, start_itn=0, start_epoch=0):

    def get_validation_loss():
        print('Compute validation')

        model.eval()
        # do validation: cosmo alone & all params
        valid_loss1, valid_loss2 = torch.zeros(len(params_subset)).to(device), torch.zeros(len(params_subset)).to(device)
        valid_loss, points = 0.0, 0
        for step, batch in enumerate(valdataloader):
            with torch.no_grad():
                batch, labels = batch
                bs = batch.shape[0]  # batch size
                batch = batch.to(device)  # maps
                y = labels.to(device)  # parameters
                time_validation = torch.zeros(batch.shape[0], dtype=int, device=device)
                p = model(batch, time_validation)  # NN output
                y_NN = p[:, params_subset]  # posterior mean
                e_NN = p[:, posterr_subset]  # posterior std

                loss1 = torch.mean((y_NN - y) ** 2, axis=0)
                loss2 = torch.mean(((y_NN - y) ** 2 - e_NN ** 2) ** 2, axis=0)
                valid_loss1 += loss1 * bs
                valid_loss2 += loss2 * bs
                points += bs
        valid_loss = torch.log(valid_loss1 / points) + torch.log(valid_loss2 / points)
        valid_loss = torch.mean(valid_loss).item()
        return valid_loss

    itn = start_itn
    epoch = start_epoch
    posterr_subset = np.array([i+6 for i in params_subset], dtype=int)

    #initial valid loss on noiseless fields
    inivl = get_validation_loss()
    print('Initial valid loss = %.3e' % inivl)

    if not DEBUG:  wandb.log({"val-loss": inivl, "iter": itn, "epoch": epoch})
    while epoch<epochs:
        print('Epoch: ', epoch)
        train_loss1, train_loss2 = torch.zeros(len(params_subset)).to(device), torch.zeros(len(params_subset)).to(
            device)
        train_loss, points = 0.0, 0

        model.train()
        for step, batch in enumerate(traindataloader):  # Step: each pass over a batch
            optimizer.zero_grad() #prevents gradient accumulation
            batch, labels = batch
            batch = batch.to(device)
            batch_size = batch.shape[0]
            t = sampler.get_timesteps(batch_size, itn) #[0, T]
            noisy_batch = diffusion.q_sample_incl_t0(x_start=batch, t=t, noise=None)
            y = labels.to(device) #should only have length = numparams to train on

            #loss for param_inference
            prediction = model(noisy_batch, t) #t's range goes from [0, T] (T+1 possible inputs incl x0)
            y_NN = prediction[:, params_subset]
            std_NN = prediction[:, posterr_subset]
            loss1 = torch.mean((y_NN - y) ** 2, axis=0)
            loss2 = torch.mean(((y_NN - y) ** 2 - std_NN ** 2) ** 2, axis=0)
            loss = torch.mean(torch.log(loss1) + torch.log(loss2))

            train_loss1 += loss1 * batch_size
            train_loss2 += loss2 * batch_size
            points += batch_size

            loss.backward()
            optimizer.step()
            scheduler.step()


            if step % 100 == 0:
                print("Loss:", loss.item())
            if not DEBUG:   wandb.log({"loss": loss.item(), "iter": itn, "epoch": epoch})

            if itn % 4000 == 0:
                misc_save_params.update({'epoch': epoch, 'itn': itn, 'model_state_dict': model.state_dict(),
                                         'optimizer_state_dict': optimizer.state_dict(), 'lr_schedule': scheduler})
                torch.save(misc_save_params, resdir + f'checkpoint_{itn}.pt')
            itn += 1

        train_loss = torch.log(train_loss1/points) + torch.log(train_loss2/points)
        train_loss = torch.mean(train_loss).item()
        valid_loss = get_validation_loss()
        if not DEBUG:  wandb.log({"train-loss": train_loss, "val-loss": valid_loss, "iter": itn, "epoch": epoch})
        if train_loss is np.nan: return np.nan
        print('Epoch: %03d TrainLoss: %.3e ValidLoss: %.3e ' % (epoch, train_loss, valid_loss), end='')
        epoch += 1
    return itn, epoch



if __name__=='__main__':
    #get the best architecture, hyperparameters from optuna for that field
    fdatabase  = 'sqlite://///n/holylfs05/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/data/InferenceNets/Weights/IllustrisTNG_o3_Mcdm_all_steps_500_500_o3.db'
    study_name = 'wd_dr_hidden_lr_o3'
    best = IN_utils.best_params_database(study_name, fdatabase) # get best trial number
    trial_number = best[0]

    study = optuna.load_study(study_name=study_name, storage=fdatabase)
    trial = study.trials[trial_number]
    print("Trial number:  number {}".format(trial.number))
    print("Loss:          %.5e"%trial.value)
    print("Params: ")

    for key, value in trial.params.items():
        print("{}: {}".format(key, value))

    fweights = InfNets_dir + f'Weights/weights_IllustrisTNG_Mcdm_{best[0]}_all_steps_500_500_o3.pt'
    BATCH_SIZE = int(config_dict['train']['batch_size'])
    MAXLR = trial.params['lr']
    DR = trial.params['dr']
    WD = trial.params['wd']
    HIDDEN = trial.params['hidden']
    BETA1, BETA2 = 0.5, 0.999
    #TODO: Decide if you wanna use your own batch size or their best hyperparam


    #get the architecture, and modify accordingly to accomodate time as an input
    model = model_o3_err(trial.params['hidden'], trial.params['dr'], 1)
    model = nn.DataParallel(model)
    model.to(device=device)
    network_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters in the model = %d'%network_total_params)

    #start with the weights of the pretrained network
    if os.path.exists(fweights):
        model.load_state_dict(torch.load(fweights, map_location=torch.device(device)), strict=False)
        print('Loading weights for the layers that already existed in the network')
    else:
        raise Exception('file with weights not found!!!')

    #sampler: weighting the sampler so that the probability of choosing each timestep from [1, T] is sqrt(1-alphas_cumprod)
    custom_weights = torch.cat([torch.tensor(np.array([1.0])), diffusion.sqrt_alphas_cumprod])
    sampler = hf_diffusion.TimestepSampler(sampler_type='custom_weights', timesteps=timesteps+1, device=device, custom_weights=custom_weights) #+1 because you want [0, T] which is T+1 possible versions of the data dbn. Earlier 0 corresponded to x1. Now it corresponds to x0 (no noise)

    #get_dataloaders
    ### get training data and image_size
    if 'fits' in DATAPATH:
        with fits.open(DATAPATH, memmap=True) as hdul:
            imgmemmap = hdul[0].data
    else:
        imgmemmap = np.load(DATAPATH, mmap_mode='r')
    labels_all = np.loadtxt(config_dict['data']['labels'])
    params_subset = np.array([int(elem) for elem in config_dict['data']['labels_subset']])
    assert 'Log' not in DATAPATH

    #transforms: Sequential(Log, RandomFlipRotate, StandardScaling/Minmax)
    with open(config_dict['data']['normfile'], 'rb') as f:
        normdict = pickle.load(f)
        scaledict = normdict[config_dict['data']['normkey']]
        mean, std = scaledict['mean'], scaledict['std']

    randflips_t, _ = get_data_transforms(imgmemmap, config_dict)
    transforms = Compose([lambda t: torch.log10(t), randflips_t])

    #train
    trainfrac = 0.7
    NPARAMS_ALL = len(labels_all)
    print('Total number of parameters=', NPARAMS_ALL)
    Ntrain_params = int(trainfrac*NPARAMS_ALL)
    Ntrain_fields = 15*Ntrain_params
    trainx = imgmemmap[:Ntrain_fields]
    trainy = labels_all[:Ntrain_params]
    traindata = CustomTensorDataset(trainx, transforms=transforms,
                        labels_data=trainy, labels_subset=params_subset,
                        labels_normalize=bool(config_dict['data']['labels_normalize']))
    traindataloader = DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True)

    #validation
    valfrac = 0.15
    val_params_upp = int((trainfrac+valfrac)*NPARAMS_ALL)
    val_fields_upp = 15*val_params_upp
    valx = imgmemmap[Ntrain_fields:val_fields_upp]
    valy = labels_all[Ntrain_params:val_params_upp]
    valdata = CustomTensorDataset(valx, transforms=transforms, labels_data=valy,
                labels_subset=params_subset, labels_normalize=bool(config_dict['data']['labels_normalize']))
    print(valdata[3][0].shape)
    valdataloader = DataLoader(valdata, batch_size=BATCH_SIZE, shuffle=False)

    #optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAXLR, weight_decay=WD,
                                  betas=(BETA1, BETA2))
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-9,
                        max_lr=MAXLR, cycle_momentum=False, step_size_up=500, step_size_down=500)

    #saving
    os.mkdir(resdir)
    shutil.copy(sys.argv[1], resdir+sys.argv[1][sys.argv[1].rindex('/')+1:])
    misc_save_params = {"schedule": beta_schedule_key, "schedule_args": beta_args, "betas": betas}
    end_itn, end_epoch = train(model, traindataloader, valdataloader, optimizer, scheduler, params_subset,
                            epochs, sampler, misc_save_params, start_itn=0, start_epoch=0)
    misc_save_params.update({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                             'itn': end_itn, 'epoch': end_epoch})
    torch.save(misc_save_params, resdir+'model.pt')

    '''
    Differences between your and their multifield dataset / training code:
    * their image transforms: Smooth fields -> Log10 -> StandardScale -> RandFlipRot and store the rotated copies in mem
    * their label transforms: map to [0, 1] (same as yours)
    * they smooth the raw fields at a pixel scale of 2 (but apparently not the norm ones?)
    '''
