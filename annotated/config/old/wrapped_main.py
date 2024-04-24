import torch
print(torch.__version__)
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import denoising_diffusion_pytorch.continuous_time_gaussian_diffusion as ctgd
import os

DATAPATH = "../data_processed/LogMaps_Mcdm_IllustrisTNG_LH_z=0.00_Nx64_train.npy"

class CustomTensorDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, tensor, transforms=None) -> None:
        self.tensor = tensor
        self.transforms = transforms

    def __getitem__(self, index):
        pre = self.tensor[index]
        if self.transforms:
            return self.transform(pre)

    def __len__(self):
        return self.tensor.size(0)

'''def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss
'''

if __name__=='__main__':
    model = Unet(dim=64, dim_mults=(1, 2, 4), channels=1)
    diff = GaussianDiffusion(model, image_size=64, timesteps=500, loss_type='l2')
    train_images = torch.tensor(np.load(DATAPATH))
    traindata = CustomTensorDataset(train_images.view((-1, 1, train_images.shape[-2], train_images.shape[-1])))


    timesteps = 200
    BATCH_SIZE = 32

    dataloader = DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True)

    trainer = Trainer(
        diff,
        tuple([traindata, dataloader]),
        train_batch_size=32,
        train_lr=8e-5,
        train_num_steps=500,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True  # turn on mixed precision
    )
    trainer.train()

    '''epochs = 5
    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (BATCH_SIZE,)).long()

            loss = diff.p_losses(model, batch, t, loss_type="huber")

            if step % 100 == 0:
                print("Loss:", loss.item())

            loss.backward()
            optimizer.step()
    '''
    print(45)
