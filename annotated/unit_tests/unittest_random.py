import torch
import numpy as np

SEED = 3
torch.manual_seed(SEED)
np.random.seed(SEED)

def misc_random_func():
    noise = torch.randn(size=(3, 4))
    return noise

if __name__=='__main__':
    #call1
    n1 = misc_random_func()
    #call2
    n2 = misc_random_func()
    #n1 and n2 are different but the same everytime you rerun the file.
    print('3')