import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import os
import sys
import argparse


import evaluate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



def combine_pickles_into_one_file(picklelist, final_savepath):
    '''
    Combines a list of pickles into one pickle.
    :param picklelist: list of pickle paths
    :param final_savepath: path to save combined pickle
    :return:
    '''
    assert len(picklelist)>0
    assert os.path.isdir(os.path.dirname(final_savepath))
    combined_dict = {}
    models = None
    labels_subset = None
    params_combined = []
    samples_combined = []
    for p in picklelist:
        assert os.path.isfile(p)
        with open(p, 'rb') as f:
            d = pickle.load(f)
        params_combined.append(d['params'])
        samples_combined.append(d['samples'][0].numpy())
        if models is None:
            models = d['models']
            labels_subset = d['labels_subset']
        else:
            assert np.all(np.array(models)==np.array(d['models']))
            assert np.all(labels_subset==d['labels_subset'])
    params_final = np.vstack(params_combined)
    samples_final = np.vstack(samples_combined)
    combined_dict['params'] = params_final
    combined_dict['samples'] = [samples_final]

    with open(final_savepath, 'wb') as f:
        pickle.dump(combined_dict, f)
    return


def sample_one_checkpoint_batched(args):
    '''
    Generates samples for a single checkpoint.
    :param args:
    :return:
    '''
    device = 'cuda:0'
    #device = f'cuda:{args.device_id}'
    print(f'Running on device: {0} of {torch.cuda.device_count()} devices')
    sdpath = args.checkpoint #'../diffusion-models-for-cosmological-fields/annotated/results/samples_exps/Run_4-26_16-42/checkpoint_224000.pt'
    B = args.num_samples
    if args.dataset=='train':
        params = evaluate.get_train_cosmo_params(Bsize=args.num_params, seed=args.param_seed)
    elif args.dataset=='validation':
        params = evaluate.get_validation_cosmo_params(Bsize=args.num_params, seed=args.param_seed)
    else:
        raise NotImplementedError()
    param_inputs = np.repeat(params, B, axis=0)
    SEED = args.sample_seed
    labels_subset = np.array([0, 1])
    Nx = args.img_size
    ddp = args.ddp

    #make sample directory
    savedir = args.savedir
    assert os.path.isdir(savedir), savedir #...samples_dir/Run_5-7_0-50
    subdir = os.path.join(savedir, args.dataset)
    os.makedirs(subdir, exist_ok=True) #...samples_dir/Run_5-7_0-50/train
    ckpnumber = int(sdpath.split('_')[-1].split('.')[0])
    ckpdir = os.path.join(subdir, f'checkpoint_{ckpnumber}')
    os.makedirs(ckpdir, exist_ok=False) #...samples_dir/Run_5-7_0-50

    #save args as dictinary
    argdict = vars(args)
    pickle.dump(argdict, open(os.path.join(ckpdir, 'args.pkl'), 'wb'))

    NBATCH = int(param_inputs.shape[0]/args.batchsize)
    if NBATCH==0: NBATCH+=1
    print('Splitting into {} batches'.format(NBATCH))
    params_batched = np.array_split(param_inputs, NBATCH, axis=0)
    picklelist = []
    
    for b, pbatch in enumerate(params_batched):
        savepath_batch = os.path.join(ckpdir, f'sampled_batch{b}.pkl') if NBATCH>1 else os.path.join(ckpdir, 'samples.pkl')
        picklelist.append(savepath_batch)
        _ = evaluate.save_samples_from_checkpoints(pbatch, [sdpath], SEED, labels_subset, device, Nx=Nx,
                        savepath=savepath_batch, get_transform=True, ddp=ddp)
    if NBATCH>1:
        combine_pickles_into_one_file(picklelist, final_savepath=os.path.join(ckpdir, 'samples.pkl'))
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--savedir", type=str, required=True)
    parser.add_argument("--batchsize", type=int, default=60) #20x2
    parser.add_argument("--num_params", type=int, default=50)
    parser.add_argument("--param_seed", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--sample_seed", type=int, default=20)
    parser.add_argument("--dataset", type=str, choices=['train', 'validation', 'test'], default='train')
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--ddp", default=False, action="store_true")
    #parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()
    sample_one_checkpoint_batched(args)
