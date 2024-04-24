import torch
import pickle
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from torch import nn

#data transformations
class IdentityTransform:
    def __call__(self, x):
        return x

class RotateTransform:
    def __init__(self, angle: int):
        self.angle = angle

    def __call__(self, x):
        return transforms.functional.rotate(x, self.angle)

def get_minmax_transform(rangemin, rangemax):
    transforms = Compose([lambda t: (t - rangemin) / (rangemax - rangemin) * 2 - 1])
    invtr = Compose([lambda t: (t + 1)/2 * (rangemax - rangemin) + rangemin])
    return transforms, invtr


def get_center_transform(rangemin, rangemax):
    mean_data = (rangemin+rangemax)/2
    transforms = Compose([lambda t: t - mean_data])
    invtr = Compose([lambda t: t + mean_data])
    return transforms, invtr


def get_meanstd_transform(mean, std):
    transforms = Compose([lambda t: (t - mean)/std])
    invtr = Compose([lambda t: mean + (t*std)])
    return transforms, invtr


def get_all_random_rots_flips(postrot_transforms=None):
    '''
    :param postrot_transforms: tr. tr is a transform applied AFTER randt. The effective transform is thus
        trim = tr(randt(img)), and the effective inverse transform is just invtr(trim), since we don't know what the randomly rotated inverse was.
        tr is usually the minmax.
        It's fine to leave invtr untrouched since that's only relevnt at sampling time.
    :return:
    '''
    tlistall = [IdentityTransform(), transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0),
                transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0)]),
                RotateTransform(90), RotateTransform(270),
                transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), RotateTransform(270)]),
                transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), RotateTransform(90)])]
    randt = transforms.RandomChoice(tlistall)
    if postrot_transforms is not None:
        randt = transforms.Compose([randt, postrot_transforms])
    return randt


def denormalize(params_normed, labels_subset=np.array([0, 1])):
    '''
    :param params_normed: Array with Nfieldsx|Labels_Subset|
    :param labels_subset: Labels subset
    :return:
    '''
    params_NN, errors_NN = params_normed
    assert len(labels_subset)==params_NN.shape[1]
    minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])[labels_subset]
    maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])[labels_subset]
    params_NN = params_NN * (maximum - minimum) + minimum
    if errors_NN is not None:
        errors_NN = errors_NN * (maximum - minimum)
    return params_NN, errors_NN

def normalize(params_NN, labels_subset):
    '''
    :param params_NN: Array with Nfieldsx|Labels_Subset|
    :param labels_subset: Labels subset
    :return:
    '''
    assert params_NN.shape[1] == len(labels_subset)
    minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])[labels_subset]
    maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])[labels_subset]
    params_out = (params_NN - minimum) / (maximum - minimum)
    return params_out.astype(np.float32)




class CustomTensorDataset(Dataset):
    r"""Dataset wrapping tensors.
    memmap is assumed to have shape N_trainxHxW

    Each sample will be retrieved by indexing tensors along the first dimension.
    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    NOT implementing imgwise standard scaling here because then the 'true' range will be lost.
    """

    def __init__(self, memmap, transforms=None, labels_path=None, labels_subset=None, labels_normalize=False, labels_data=None,
                 subset_type='train') -> None:
        '''
        :param memmap:
        :param transforms:
        :param labels_path:
        :param labels_subset:
        :param labels_normalize:
        :param labels_data: Should be None if labels_path is present
        '''
        self.subset_type = subset_type
        if self.subset_type=='train':
            self.memmap = memmap
        else:
            self.memmap=memmap['fields']

        self.transforms = transforms
        self.image_size = self.memmap[0].shape[-1]
        self.conditional = True
        if labels_path is not None:
            assert labels_data is None
            self.labels = np.loadtxt(labels_path, dtype=np.float32)
        elif labels_data is not None:
            #assert labels_path is None
            self.labels = labels_data
            assert len(memmap)/15 == len(labels_data)
        elif 'params' in memmap: #from an npz file which had 'fields', 'params'
            self.labels = memmap['params']
            assert len(self.memmap)/15 == len(self.labels)
        else:
            self.conditional = False

        if self.conditional:
            if labels_subset is None:
                self.labels_subset = np.arange(6)
                print('Conditioning with respect to all parameters')
            else:
                self.labels_subset = labels_subset
                print('Conditioning with respect to parameters at positions', self.labels_subset)

        if labels_normalize:
            print('Normalizing labels before conditioning')
            self.labels_normalize= True
            self.params_minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5], dtype=np.float32)[self.labels_subset]
            self.params_maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0], dtype=np.float32)[self.labels_subset]
        else:
            self.labels_normalize = False

    def __getitem__(self, index):
        #pre = self.tensor[index]
        pre = torch.from_numpy(np.array(self.memmap[index]).astype(np.float32)).view((1, self.image_size, self.image_size))
        if self.transforms:
           pre= self.transforms(pre)
        if self.conditional:
            #return pre, self.labels[(index%9000)//15] #WARNING: only valid for the Nx=64 train cosmology dset, and assumes the dataset was augmented
            labels_normed = self.labels[index//15, self.labels_subset]
            if self.labels_normalize:
                labels_normed = (labels_normed - self.params_minimum)/(self.params_maximum - self.params_minimum)
            return pre, labels_normed #this assumes that the dataset read in was not augmented and any augmentations were applied in self.transforms
        else:
            return pre


    def __len__(self):
        return self.memmap.shape[0]



