from hf_diffusion import *

def get_default_model_kwargs(image_size, channels, config_dict):
    dim_mults = tuple([int(elem) for elem in config_dict['architecture']['dim_mults']])
    dim = image_size if 'unet_dim' not in config_dict['architecture'] else int(config_dict['architecture']['unet_dim'])
    model_kwargs = {"dim": dim, "channels": channels, "dim_mults": dim_mults, "use_convnext": False} #this has been updated post Oct1 to make unet_dim=dim
    model_kwargs.update({'init_conv': True if 'init_conv' not in config_dict['architecture'].keys() else bool(config_dict['architecture']['init_conv'])})
    if bool(config_dict['data']['conditional']):
        conditional_dim = 6 if config_dict['data']['labels_subset']=='all' else len(config_dict['data']['labels_subset'])
        model_kwargs.update({'conditional_dim': int(conditional_dim)})
    if 'time_embed_dim' in config_dict['architecture'].keys():
        model_kwargs.update({'time_embed_dim': int(config_dict['architecture']['time_embed_dim'])})
    if 'base2fourier' in config_dict['architecture'].keys():
        model_kwargs.update({'base2fourier': True})
        model_kwargs.update(config_dict['architecture']['base2fourier'])
    if 'addcond2img' in config_dict['architecture'].keys():
        model_kwargs.update({'addcond2img': config_dict['architecture']['addcond2img']})

    if config_dict['architecture']['model']=='explicitconditional':
        model_kwargs.pop('use_convnext')
        if 'non_linear_conditional_embedding' in config_dict['architecture']:
            model_kwargs.update({'non_linear_conditional_embedding': bool(config_dict['architecture']['non_linear_conditional_embedding'])})
        if 'use_cond_dim_for_block' in config_dict['architecture']:
            model_kwargs.update({'use_cond_dim_for_block': bool(config_dict['architecture']['use_cond_dim_for_block'])})
        if 'pk_embedding' in config_dict['architecture']: 
            model_kwargs.update({'pk_embedding': bool(config_dict['architecture']['pk_embedding'])})
            model_kwargs.update({'image_size': int(config_dict['architecture']['image_size'])})
        if 'circular_conv' in config_dict['architecture']:
            model_kwargs.update({'circular_convolution': bool(config_dict['architecture']['circular_conv'])})

    return model_kwargs

def get_defaults_config_dict(config_dict):
    if 'conditional' not in config_dict['data']:
        config_dict['data'].update({'conditional': False})
    if 'sampler_args' not in config_dict['diffusion'].keys():
        config_dict['diffusion'].update({'sampler_args': {'sampler_type': 'uniform'}})
    if 'transforms' not in config_dict['data'].keys():
        config_dict['data'].update({'transforms': 'minmax'}) #by default: scales the minmax of the training data to [-1, 1]
    return config_dict


def get_data_transforms(imgmemmap, config_dict):
    '''
    Note: This is only for transforms that are the SAME function for the ENTIRE dataset. NOT imagewise transforms that are pegged to the pixel range of each image.
    :param imgmemmap:
    :param config_dict:
    :return:
    TODO: Check that the discrepancy: that transforms has rotate and inverse transforms doesn't, shouldn't really affect anyting.
    '''
    if config_dict['data']['transforms'] == 'minmax':
        print('Minmax scaling to -1, 1')
        with open(config_dict['data']['normfile'], 'rb') as f:
            normdict = pickle.load(f)
            scaledict = normdict[config_dict['data']['normkey']]
            rangemin, rangemax = scaledict['rangemin'], scaledict['rangemax']
        print('RangeMin={:.3f}, RangeMax={:.3f}'.format(rangemin, rangemax))
        RANGE_MIN, RANGE_MAX = torch.tensor(rangemin), torch.tensor(rangemax)
        transforms, inverse_transforms = get_minmax_transform(RANGE_MIN, RANGE_MAX)
    elif config_dict['data']['transforms']=='None':
        print('No prior data transformation applied')
        transforms, inverse_transforms = nn.Identity(), nn.Identity()
    elif config_dict['data']['transforms']=='center':
        print('Center so mean 0 but no multiplicative scaling')
        RANGE_MIN, RANGE_MAX = torch.tensor(imgmemmap.min()), torch.tensor(imgmemmap.max())
        transforms, inverse_transforms = get_center_transform(RANGE_MIN, RANGE_MAX)
    elif config_dict['data']['transforms'] == 'minmax+randfliprot':
        print('RandomFlipRot followed by Minmax scaling to -1, 1')
        with open(config_dict['data']['normfile'], 'rb') as f:
            normdict = pickle.load(f)
            scaledict = normdict[config_dict['data']['normkey']]
            rangemin, rangemax = scaledict['rangemin'], scaledict['rangemax']
        print('RangeMin={:.3f}, RangeMax={:.3f}'.format(rangemin, rangemax))
        RANGE_MIN, RANGE_MAX = torch.tensor(rangemin), torch.tensor(rangemax)
        transforms, inverse_transforms = get_minmax_transform(RANGE_MIN, RANGE_MAX)
        transforms = get_all_random_rots_flips(postrot_transforms=transforms)
    elif config_dict['data']['transforms']=='standardscale+randfliprot':
        print('RandomFlipRot followed by StandardScale with respect to the full dataset')
        with open(config_dict['data']['normfile'], 'rb') as f:
            normdict = pickle.load(f)
            scaledict = normdict[config_dict['data']['normkey']]
            mean, std = scaledict['mean'], scaledict['std']
        print('Mean={:.3f}, Std={:.3f}'.format(mean, std))
        transforms, inverse_transforms = get_meanstd_transform(mean, std)
        #randfliprot
        transforms = get_all_random_rots_flips(postrot_transforms = transforms)
    elif config_dict['data']['transforms']=='imgwisestandardscale+randfliprot':
        print('RandomFlipRot followed by StandardScale with respect to the mean, std of each image')
        transforms, inverse_transforms = get_all_random_rots_flips(postrot_transforms = None), nn.Identity()
        #This code only rotates or flips the image -- you'll have to implement the standard scaling elsewhere.
    else:
        raise NotImplementedError()
    return transforms, inverse_transforms


def get_scheduler(config_dict, optimizer):
    if 'scheduler' in config_dict['train'].keys():
        scheduler_kw = config_dict['train']['scheduler_args']
        scheduler_kw['optimizer'] = optimizer
        if config_dict['train']['scheduler'] == 'ReduceLROnPlateau': #additional preproc to convert yaml strs to float
            scheduler_kw['factor'] = float(scheduler_kw['factor'])
            scheduler_kw['threshold'] = float(scheduler_kw['threshold'])
        scheduler = getattr(torch.optim.lr_scheduler, config_dict['train']['scheduler'])(**scheduler_kw)
    else:
        scheduler=None
    return scheduler


class EMAUpdater():
    def __init__(self, model_ema, start_update_epoch, update_interval_iter, ema_decay):
        self.model_ema = model_ema
        self.start_update_epoch = start_update_epoch
        self.update_interval = update_interval_iter
        self.epoch_tracker = 0 #Don't * really * need this
        self.ema_decay = ema_decay
        self.state = False #False if OFF

    def update(self, model, epoch):
        self.epoch_tracker = epoch

        if (self.state==False) and (self.epoch_tracker>=self.start_update_epoch): #to turn ON
            print('Switching on model_ema')
            self.state = True
            self.iter_tracker = 0 #iterations since EMAUpdater was switched on
            with torch.no_grad():
                for p_ema, p in zip(self.model_ema.parameters(), model.parameters()):
                    p_ema.copy_(p)
                for b_ema, b in zip(self.model_ema.buffers(), model.buffers()):
                    b_ema.copy_(b)

        elif self.state==True: #Already ON
            if (self.iter_tracker % self.update_interval)==0:
                with torch.no_grad():
                    for p_ema, p in zip(self.model_ema.parameters(), model.parameters()):
                        p_ema.copy_(p.lerp(p_ema, self.ema_decay)) #p_update + 0.99*(p_ema - p_update) = 0.01*p_update + 0.99*p_ema
                    for b_ema, b in zip(self.model_ema.buffers(), model.buffers()):
                        b_ema.copy_(b)
                self.iter_tracker+=1
            else:
                self.iter_tracker += 1
        else:#Not yet reached late enough epoch
            pass
        return




