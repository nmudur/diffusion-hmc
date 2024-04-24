
#https://huggingface.co/blog/annotated-diffusion

import os
import wandb
import yaml
import torch
import datetime
import numpy as np

import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, CenterCrop, Resize
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import hf_diffusion
from hf_diffusion import *

os.environ[
    "PATH"] += os.pathsep + '/Users/nayantaramudur/opt/anaconda3/envs/torch_gpu/lib/python3.8/site-packages/graphviz/'

with open("params.yaml", 'r') as stream:
    ydict = yaml.safe_load(stream)


config_dictionary = ydict
dt = datetime.datetime.now()
name = f'Run_{dt.month}-{dt.day}_{dt.hour}-{dt.minute}'
wandb.init(project='diffmod_cosmo0', job_type='unconditional',
           config=config_dictionary, name=name)


timesteps = int(wandb.config['diffusion']['timesteps'])
epochs = int(wandb.config['train']['epochs'])
beta_schedule_key = wandb.config['diffusion']['beta_schedule']
DATAPATH =wandb.config['data']['path']


BATCH_SIZE = int(wandb.config['train']['batch_size'])
dim_mults = tuple([int(elem) for elem in wandb.config['architecture']['dim_mults']])
LR = float(wandb.config['train']['learning_rate'])
if torch.cuda.is_available(): 
    device = 'cuda'
else: 
    device='cpu'
print(device)
beta_schedule = getattr(hf_diffusion, beta_schedule_key)
betas = beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
print('Beta shape', betas.shape)

# calculations for diffusion q(x_t | x_0) and others
alphas_cumprod = torch.cumprod(alphas, axis=0) #alpha_bar
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
# x_t = sqrt_alphas_cumprod* x_0 + sqrt_one_minus_alphas_cumprod * eps_t
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


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
            return self.transforms(pre)

    def __len__(self):
        return self.tensor.size(0)



# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    #L_CE <= L_VLB ~ Sum[eps_t - MODEL(x_t(x_0, eps_t), t) ]
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_t, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()
    return loss


def train(model, dataloader, optimizer, timesteps, epochs, loss_type="huber"):

    '''
    #alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) #needed where?
    #sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    #posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    '''

    for epoch in range(epochs):  # Epochs: number of full passes over the dataset
        print('Epoch: ', epoch)
        for step, batch in enumerate(dataloader):  # Step: each pass over a batch
            optimizer.zero_grad() #prevents gradient accumulation

            batch_size = batch.shape[0]
            batch = batch.to(device)
            #print('Data device', next(model.parameters()).device)


            # Algorithm 1 line 3: sample t uniformly for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, batch, t, loss_type=loss_type)

            if step % 100 == 0:
                print("Loss:", loss.item())
                wandb.log({"loss": loss.item(), "epoch": epoch}) #Add an example sample later

            loss.backward()
            optimizer.step()
    return


@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 but save all images:

@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

if __name__ == '__main__':

    #SinusoidalPositionEmbeddings
    '''
    tt = torch.arange(32)
    out = SinusoidalPositionEmbeddings(1000)(tt)

    plt.figure()
    idxl = [1, 5, 10]
    [plt.plot(np.arange(32), out[idx, :].numpy(), label=idx) for idx in idxl]
    plt.show()
    print(1)
    '''

    #Block
    '''inp = torch.arange(32)*0.1
    inp = inp.view(1, 2, 4, 4)
    out = Block(2, 6, groups=3)(inp)
    '''

    #Attention
    '''
    inp = torch.arange(32) * 0.1
    inp = inp.view(1, 2, 4, 4)
    out = Attention(2)(inp)
    print(121)
    '''
    #Dunno what this is
    '''
    image_size = 128
    transform = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),  # turn into Numpy array of shape HWC, divide by 255
        Lambda(lambda t: (t * 2) - 1),

    ])

    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    x_start = transform(image).unsqueeze(0)
    x_start.shape



    '''
    t1 = torch.randn((3, 1))
    t2 = torch.randn((1, 4))
    print(torch.allclose(t1*t2, torch.matmul(t1, t2)))
    imgnp = np.load(DATAPATH)
    train_images = torch.from_numpy(imgnp).float() #crash here only with torch_gpu environment when you don't write this as torch.from_numpy

    image_size = train_images.shape[-1]
    channels = 1
    wandb.config['data'].update({'image_size': image_size, 'channels': channels})

    RANGE_MIN, RANGE_MAX = train_images.min(), train_images.max()
    transforms = Compose([lambda t: (t - RANGE_MIN) / (RANGE_MAX - RANGE_MIN) * 2 - 1])
    traindata = CustomTensorDataset(train_images.view((-1, 1, train_images.shape[-2], train_images.shape[-1])),
                                    transforms=transforms)

    dataloader = DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True)

    if wandb.config['architecture']['model']=='baseline':
        model = Unet(
            dim=image_size,
            channels=channels,
            dim_mults=dim_mults, use_convnext=False
        )
        model.to(device)
    else:
        raise NotImplementedError()

    if wandb.config['train']['optimizer']=='Adam':
        optimizer = Adam(model.parameters(), lr=LR)
    else:
        raise NotImplementedError()


    #torch.onnx.export(model, tuple([batch, batchtime]), "models/unet.onnx", input_names=["Image_Batch", "Time_Batch"], output_names=["Output"])

    train(model, dataloader, optimizer, timesteps=timesteps, epochs=epochs, loss_type="huber")

    #sample from trained model
    samples = sample(model, image_size=image_size, batch_size=10, channels=channels)
    #np.save('results/test_samples.npy', samples[-1])
    plt.figure()
    c = plt.imshow(samples[-1][4].reshape((image_size, image_size)), origin='lower')
    plt.colorbar(c)
    plt.savefig(f'results/samples_exps/{name}.png')
    plt.show()
    torch.save({'model_state_dict': model.state_dict()}, f'results/samples_exps/{name}.pt')
    wandb.log({"sample": wandb.Image(f'results/samples_exps/{name}.png')})

    #print(model)





