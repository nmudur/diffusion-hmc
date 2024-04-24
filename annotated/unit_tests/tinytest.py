import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import os
import sys
import argparse


import evaluate

def sample_one_checkpoint_batched(args):
    '''
    Generates samples for a single checkpoint.
    :param args:
    :return:
    '''
    device = f'cuda:{args.device_id}'
    print(f'Running on device: {args.device_id} of {torch.cuda.device_count()} devices')
    sdpath = args.checkpoint #'../diffusion-models-for-cosmological-fields/annotated/results/samples_exps/Run_4-26_16-42/checkpoint_224000.pt'
    B = args.num_samples
    if args.dataset=='train':
        params = evaluate.get_train_cosmo_params(Bsize=args.num_params, seed=args.param_seed)
    elif args.dataset=='validation':
        params = evaluate.get_validation_cosmo_params(Bsize=args.num_params, seed=args.param_seed)
    else:
        raise NotImplementedError()
    print(f'First {args.dataset} param')
    print(params[0, :])
    param_inputs = np.repeat(params, B, axis=0)
    SEED = args.sample_seed
    labels_subset = np.array([0, 1])
    Nx = 256
    ddp = args.ddp

    NBATCH = int(param_inputs.shape[0]/args.batchsize)
    if NBATCH==0: NBATCH+=1
    print('Splitting into {} batches'.format(NBATCH))
    return

#TODO: THis doesn't ensure you get the SAME parameters eg: if the seed is different in subsequent runs??
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--savedir", type=str, required=True)
    parser.add_argument("--batchsize", type=int, default=60) #20x2
    parser.add_argument("--num_params", type=int, default=50)
    parser.add_argument("--param_seed", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=20)
    parser.add_argument("--dataset", type=str, choices=['train', 'validation', 'test'], default='train')
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--ddp", default=False, action="store_true")
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()
    sample_one_checkpoint_batched(args)
