import os
import numpy as np

import evaluate
import torch
if torch.cuda.is_available():
    device='cuda'
else:
    device= 'cpu'
torch.set_default_device('cpu')
torch.set_default_dtype(torch.float32)


NSAMPS = 405
BSIZE=50
test_par = np.vstack([[0.3, 0.8, 1.0, 1.0, 1.0, 1.0]]*NSAMPS)

sdpath = '/n/holylfs06/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/diffusion-models-for-cosmological-fields/annotated/results/samples_exps/Run_10-30_2-32/checkpoint_260000.pt'
labels_subset = np.array([0, 1])
param_batches = torch.split(torch.tensor(test_par), BSIZE, dim=0)
seeds = np.arange(len(param_batches))
cvdir = '/n/holylfs06/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/diffusion-models-for-cosmological-fields/annotated/results/checkpoint_samples/Run_10-30_2-32/CV/'

for s in seeds:
    pb = param_batches[s]
    print('Seed:', s, pb.shape[0])
    resdict = evaluate.save_samples_from_checkpoints(pb.cpu().numpy(), [sdpath], s, labels_subset, 'cuda', 
                Nx=256, get_transform=True, savepath=os.path.join(cvdir, f'seed_{s}.pkl'))