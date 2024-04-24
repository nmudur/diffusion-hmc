import torch
import numpy as np

from inspect import isfunction
from functools import partial
from torch import nn, einsum
from einops import rearrange
import pickle
import torch.nn.functional as F

#from .diffusion import Diffusion

### All NN models and helper functions that are used in hf_diffusion and main

#helper functions and transforms

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)



#blocks
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim, padding_mode='zeros'):
    return nn.Conv2d(dim, dim, 4, 2, 1, padding_mode=padding_mode)

class SinusoidalPositionEmbeddings(nn.Module):
    #embeds time in the phase
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :] #t1: [40, 1], t2: [1, 32]. Works on cpu, not on mps
        #^ is matmul: torch.allclose(res, torch.matmul(t1.float(), t2)): True when cpu
        #NM: added float for mps
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings #Bx64


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, padding_mode='zeros'):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, conditional=False, conditional_emb_dim=None, padding_mode='zeros'):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )
        self.conditional = conditional
        if conditional:
            self.conditional_emb_dim = time_emb_dim if conditional_emb_dim is None else conditional_emb_dim
            self.cond_mlp = (
                nn.Sequential(nn.SiLU(), nn.Linear(self.conditional_emb_dim, dim_out))
                if conditional
                else None
            )

        self.block1 = Block(dim, dim_out, groups=groups, padding_mode=padding_mode)
        self.block2 = Block(dim_out, dim_out, groups=groups, padding_mode=padding_mode)
        self.res_conv = nn.Conv2d(dim, dim_out, 1, padding_mode=padding_mode) if dim != dim_out else nn.Identity()


    def forward(self, x, time_emb=None, label_emb=None, *misc_args):
        #print(x.shape, time_emb is None, label_emb is None)
        if self.conditional:
            assert label_emb is not None
        
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h
        
        if self.conditional:   
            label_emb = self.cond_mlp(label_emb)
            h = rearrange(label_emb, "b c -> b c 1 1") + h
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, padding_mode='zeros'):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False, padding_mode=padding_mode)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1, padding_mode=padding_mode)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k) #summing over d=dim_head
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, padding_mode='zeros'):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1, padding_mode=padding_mode),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

#https://github.com/google-research/vdm/blob/0c60a8979491f56e32f7ed3bca22bd5b506d0fdb/model_vdm.py
class Base2FourierFeatures():
    def __init__(self, start=0, stop=8, step=1):
        self.start = start
        self.stop = stop
        self.step = step
        self.fourier_channels = 2*len(torch.arange(self.start, self.stop+1, self.step))
        print(f'Appending {self.fourier_channels} Fourier Features with start={self.start}, stop={self.stop}')

    def __call__(self, x):
        #assert x[1].shape==1 #Implemented only for a single channel rn
        freqs = torch.arange(self.start, self.stop+1, self.step, dtype=x.dtype, device=x.device) #n-1 in the paper
        w = 2. ** freqs * 2 * np.pi
        w = torch.tile(w[None, :], (1, x.shape[1])) #-1 -> 1, 1F
        h = x.repeat_interleave(freqs.shape[0], dim=1) #B|C*F|HW
        h = w[:, :, None, None] * h
        h = torch.cat([torch.sin(h), torch.cos(h)], dim=1) #B|2*F|HW, sin(fi) then cos(fi)
        return h


#unet
class Unet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            with_time_emb=True,
            resnet_block_groups=8, 
            use_convnext=True,
            convnext_mult=2,
            init_conv=True,
            final_conv=True,
            conditional_dim=None,
            time_embed_dim=None,
            marginal_prob_std=None,
            cond_final_conv=False,
            add_conditioning=False,
            base2fourier=False, b2fstart=0, b2fend=8, b2fstep=1):
        super().__init__()

        # determine dimensions
        self.channels = channels
        time_dim = default(time_embed_dim, dim*4)
        if conditional_dim is not None:
            self.conditional = True
            self.conditional_dim = conditional_dim #6
            self.conditional_embedding = nn.Linear(self.conditional_dim, time_dim) #just an FC layer taking the parameter vector and putting it in a |time_dim| space
        else:
            self.conditional = False

        self.b2fbool = False
        if base2fourier:
            self.b2fbool = True
            self.base2fourierlayer = Base2FourierFeatures(b2fstart, b2fend, b2fstep)

        effchannels = channels*(1 + self.base2fourierlayer.fourier_channels) if self.b2fbool else channels
        if init_conv: 
            init_dim = default(init_dim, dim // 3 * 2)
            self.init_conv = nn.Conv2d(effchannels, init_dim, 7, padding=3) #ch_in, ch_out, size
        else:
            init_dim = effchannels #if no_init conv, channels_in = 1


        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            #time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.cond_final_conv = False
        if final_conv and cond_final_conv:
            print('here')
            self.cond_final_conv = True
            cond_block_klass = partial(ResnetBlock, groups=resnet_block_groups, conditional=True)
            self.final_conv = nn.ModuleList([cond_block_klass(dim, dim, time_emb_dim=time_dim), nn.Conv2d(dim, out_dim, 1)])
        elif final_conv:
            self.final_conv = nn.Sequential(
                block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
            )
        else:
            raise NotImplementedError #keep final Conv2D for now
            self.final_conv = nn.Conv2d(dim, out_dim, 1)

        self.addcond2img = False
        if add_conditioning:
            self.addcond2img = True
            self.condemb1 = nn.Linear(self.conditional_dim, 1)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, time, label=None):
        #x: BCHW: B x 1 x Nx x Nx
        #label: Bxconditional_dim(=Bx6)
        if self.b2fbool:
            z_f = self.base2fourierlayer(x)
            x = torch.cat([x, z_f], dim=1)

        try:
            x = self.init_conv(x) #? Bx42x64x64, NMEDIT
        except AttributeError:
            pass #x=x, no init_conv


        t = self.time_mlp(time) if exists(self.time_mlp) else None #MPS breaks here!
        if self.conditional:
            label_emb = self.conditional_embedding(label)
            t += label_emb

        # TODO: pretty sure this isn't what you had in mind
        if self.addcond2img:
            lemb1 = self.condemb1(label) #B, 1
            x += lemb1[:, None, None, :] #B, LE -> B, C=1, 1, 1

        #BX256
        h = []

        # downsample
        ctr = 0
        for block1, block2, attn, downsample in self.downs:
            #print(x.shape, ctr)
            x = block1(x, t)
            x = block2(x, t) #Bx64x64x64, Bx128x32x32, Bx256x16x16 when no init_conv
            x = attn(x)
            h.append(x)
            x = downsample(x)
            ctr+=1

        # bottleneck: Bx256x16x16
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1) #residual??
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        if self.cond_final_conv:
            mod= self.final_conv[0]
            x = mod(x, None, label_emb)
            mod = self.final_conv[1]
            x = mod(x)
        else:
            x = self.final_conv(x)
        #print('Finished pass')
        # For the 'SDE' based models you might want to normalize the output
        # of the UNet by the marginal_prob_std = \sqrt{\lambda(t)}
        if self.marginal_prob_std is not None:
            x = x/self.marginal_prob_std(t)[:, None, None, None]
        return x


#unet with explicit conditioning on parameters and power spectrum of input
class Log10Pk_Compute():
    def __init__(self, Nx, kbin_smoothing=0.5, device='cpu'):
        self.smoothed = kbin_smoothing
        if Nx==64:
            pknormfile= '/n/holylfs05/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/notebooks/Tmp/3_25/pknorms64.pkl'
        else:
            assert Nx==256
            pknormfile = '/n/holylfs05/LABS/finkbeiner_lab/Users/nmudur/project_dirs/CMD_2D/notebooks/Tmp/3_25/pknorms.pkl'
        pknormdict = pickle.load(open(pknormfile, 'rb'))
        self.lminpk, self.lmaxpk = torch.tensor(pknormdict['lminpk'], device=device), torch.tensor(pknormdict['lmaxpk'], device=device)

    def __call__(self, inputimg):
        #assert x[1].shape==1 #Implemented only for a single channel rn
        #Compute power spectrum
        assert len(inputimg.shape)==4
        assert inputimg.shape[1]==1
        Nx = inputimg.shape[-1]
        kvals = torch.arange(Nx/2)
        rfftimg = torch.fft.rfft2(inputimg)/Nx**2
        rimpf = torch.abs(rfftimg)**2
        x, y = torch.meshgrid(torch.arange(Nx), torch.arange(Nx))
        x, y = x[:, :rfftimg.shape[-1]], y[:, :rfftimg.shape[-1]]
        rR = np.sqrt(x**2+y**2) #|k|
        filt = lambda r: rimpf[:, :, (rR >= r - self.smoothed) & (rR < r + self.smoothed)].mean(dim=-1, keepdim=True)
        pkmean = torch.cat([filt(k) for k in kvals], dim=2)
        assert pkmean.shape==(inputimg.shape[0], inputimg.shape[1], len(kvals))
        #Compute the embedding
        logpk = torch.log10(pkmean)
        #Normalize to 0,1 in the minmax range for each kbin
        normedlogpk = (logpk - self.lminpk)/(self.lmaxpk - self.lminpk)
        return normedlogpk[:, 0]



class PkAwareResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    def __init__(self, dim, dim_out, *, pk_dim=128, time_emb_dim=None, groups=8, conditional=False, conditional_emb_dim=None):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )
        self.conditional = conditional
        if conditional:
            self.conditional_emb_dim = time_emb_dim if conditional_emb_dim is None else conditional_emb_dim
            self.cond_mlp = (
                nn.Sequential(nn.SiLU(), nn.Linear(self.conditional_emb_dim, dim_out))
                if conditional
                else None
            )
        self.pk_mlp = nn.Sequential(nn.SiLU(), nn.Linear(pk_dim, dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, label_emb=None, log10pk=None):
        # print(x.shape, time_emb is None, label_emb is None)
        if self.conditional:
            assert label_emb is not None

        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        if self.conditional:
            label_emb = self.cond_mlp(label_emb)
            h = rearrange(label_emb, "b c -> b c 1 1") + h

        #add pk
        pk_emb = self.pk_mlp(log10pk)
        h = rearrange(pk_emb, "b c -> b c 1 1") + h
        h = self.block2(h)
        return h + self.res_conv(x)


class UnetExplicitConditional(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            resnet_block_groups=8,
            init_conv=True,
            conditional_dim=None,
            use_cond_dim_for_block=True,
            time_embed_dim=None,
            pk_embedding=False,
            base2fourier=False, b2fstart=0, b2fend=8, b2fstep=1, image_size=None,
            non_linear_conditional_embedding=False,
            circular_convolution=False):
        '''
        This model makes all the blocks explicitly conditional on the labels \theta, in addition to being conditional on E(t) + E(\theta)
        :param dim: Unet Dim (64)
        :param init_dim: If init_conv #channels before downsampling in Unet
        :param out_dim: Out channel dimensions if different from channels. Default: channels
        :param dim_mults: [1, 2, 4, 8]
        :param channels:
        :param resnet_block_groups:
        :param init_conv: Convolve before downsampling or not.
        :param conditional_dim: Number of cosmological parameters to use. len(labels_subset)
        :param use_cond_dim_for_block: Whether the blocks take as input E(\theta) [True] eg: 256 or just \theta (2)
        :param time_embed_dim: Time, Conditional Embedding Dimension
        :param pk_embedding: Whether to include an embedding for the LogPowspec of the input.
        :param base2fourier:
        :param b2fstart:
        :param b2fend:
        :param b2fstep:
        :param non_linear_conditional_embedding: The 'more complex' conditional embedding. Only matters for the E(\theta) that is added to time_embedding.
        :param circular_convolution: If true then padding_mode='circular' in all convolutions in this architecture. False to make it backwards compatible with default.
        '''
        super().__init__()
        # determine dimensions
        self.channels = channels
        time_dim = default(time_embed_dim, dim*4)
        self.use_cond_dim_for_block = use_cond_dim_for_block
        if conditional_dim is not None:
            self.conditional = True
            self.conditional_dim = conditional_dim #6
            if non_linear_conditional_embedding:
                self.conditional_embedding = nn.Sequential(nn.Linear(self.conditional_dim, 5*self.conditional_dim),
                                                           nn.GELU(), nn.Linear(5*self.conditional_dim, time_dim))
            else:
                self.conditional_embedding = nn.Linear(self.conditional_dim, time_dim) #just an FC layer taking the parameter vector and putting it in a |time_dim| space
        else:
            self.conditional = False

        self.pk_conditional= False
        if pk_embedding:
            self.pk_conditional = pk_embedding
            assert image_size is not None


        self.b2fbool = False
        if base2fourier:
            self.b2fbool = True
            self.base2fourierlayer = Base2FourierFeatures(b2fstart, b2fend, b2fstep)

        if circular_convolution:
            padding_mode='circular'
        else:
            padding_mode='zeros'

        effchannels = channels*(1 + self.base2fourierlayer.fourier_channels) if self.b2fbool else channels
        if init_conv:
            init_dim = default(init_dim, dim // 3 * 2)
            self.init_conv = nn.Conv2d(effchannels, init_dim, 7, padding=3, padding_mode=padding_mode) #ch_in, ch_out, size
        else:
            init_dim = effchannels #if no_init conv, channels_in = 1

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        if self.pk_conditional:
            block_klass = partial(PkAwareResnetBlock, pk_dim=int(image_size/2), groups=resnet_block_groups, conditional=True,
                              conditional_emb_dim=self.conditional_dim if use_cond_dim_for_block else None)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups, conditional=True,
                              conditional_emb_dim=self.conditional_dim if use_cond_dim_for_block else None, padding_mode=padding_mode) #TODO

        # time embeddings
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out, padding_mode=padding_mode))),
                        Downsample(dim_out, padding_mode=padding_mode) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, padding_mode=padding_mode)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in, padding_mode=padding_mode))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.ModuleList([block_klass(dim, dim, time_emb_dim=time_dim), nn.Conv2d(dim, out_dim, 1, padding_mode=padding_mode)])

    def forward(self, x, time, label=None):
        #x: BCHW: B x 1 x Nx x Nx
        #label: B x Conditional_dim(=Bx6)
        if self.b2fbool:
            z_f = self.base2fourierlayer(x)
            x = torch.cat([x, z_f], dim=1)

        try:
            x = self.init_conv(x) #? Bx42x64x64, NMEDIT
        except AttributeError:
            pass #x=x, no init_conv

        t = self.time_mlp(time) if exists(self.time_mlp) else None #MPS breaks here!
        if self.conditional:
            label_emb = self.conditional_embedding(label)
            t += label_emb
        log10pk = None
        if self.pk_conditional:
            log10pk = Log10Pk_Compute(x.shape[-1], device=x.device)(x)


        #BX256
        h = []

        # downsample
        ctr = 0
        block_label_input = label if self.use_cond_dim_for_block else label_emb
        for block1, block2, attn, downsample in self.downs:
            #print(x.shape, ctr)

            x = block1(x, t, block_label_input, log10pk)
            x = block2(x, t, block_label_input, log10pk) #Bx64x64x64, Bx128x32x32, Bx256x16x16 when no init_conv
            x = attn(x)
            h.append(x)
            x = downsample(x)
            ctr+=1

        # bottleneck: Bx256x16x16
        x = self.mid_block1(x, t, block_label_input, log10pk)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, block_label_input, log10pk)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1) #residual??
            x = block1(x, t, block_label_input, log10pk)
            x = block2(x, t, block_label_input, log10pk)
            x = attn(x)
            x = upsample(x)
        
        mod= self.final_conv[0]
        x = mod(x, t, block_label_input, log10pk)
        mod = self.final_conv[1]
        x = mod(x)
        return x


#### Latent Diffusion Model (VDM-like) #########
class ScalingEncoder(nn.Module):
    def __init__(self):
        super(ScalingEncoder, self).__init__()

    def forward(self, x):
        mean, std = torch.mean(x, dim=(2, 3), keepdim=True), torch.std(x, dim=(2, 3), keepdim=True, unbiased=True) #BC
        return (x - mean) / std

class DeterministicDecoder(nn.Module):
    def __init__(self, conditional_dim, mid_dim=5, channels=1, loss_type='mse', freeze=False):
        super(DeterministicDecoder, self).__init__()
        self.channels = channels
        self.shiftscalefunc = nn.Sequential(nn.Linear(conditional_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, 2*channels))
        if loss_type=='mse':
            self.loss = F.mse_loss
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            print('Freezing decoder parameters')

    def forward(self, z, labels):
        #returns x = shift(theta) + scale(theta)*z
        shiftscale = self.shiftscalefunc(labels)
        shiftscale = torch.reshape(shiftscale, (-1, self.channels, 2)) #BC
        shift, scale = shiftscale[...,0][:, :, None, None], shiftscale[...,1][:, :, None, None]
        return shift+scale*z

    #mildly weird choice of making the loss an attribute of the decoder
    def loss(self, x, x_recon):
        #returns a loss between x and decoded z: reconstruction loss
        return self.loss(x_recon, x)


def get_encoder(emodel, ekwargs, ckpt=None):
    if emodel=='imgwisestandardscale':
        assert ekwargs == {}
        enc = ScalingEncoder()
        return enc
    else:
        raise NotImplementedError()
    if ckpt is not None:
        print(f'Loading encoder from checkpoint {ckpt}')
        ckpt = torch.load(ckpt, map_location='cpu')
        enc.load_state_dict(ckpt['encoder_state_dict'])
    return

def get_decoder(demodel, dekwargs, ckpt=None):
    if demodel=='deterministic+cond':
        dec = DeterministicDecoder(**dekwargs)
    else:
        raise NotImplementedError()
    if ckpt is not None:
        print(f'Loading decoder from checkpoint {ckpt}')
        ckpt = torch.load(ckpt, map_location='cpu')
        dec.load_state_dict(ckpt['decoder_state_dict'])
    return dec

def get_model(model, mokwargs, ckpt=None):
    if model=='explicitconditional':
        model = UnetExplicitConditional(**mokwargs)
    elif model=='baseline':
        model = Unet(**mokwargs)
    else:
        raise NotImplementedError()
    if ckpt is not None:
        print(f'Loading model from checkpoint {ckpt}')
        ckpt = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(ckpt)
    return model

#Encoder+Decoder+ScoreModel combined
#ported from https://github.com/google-research/vdm/blob/main/model_vdm.py
class LatentDiffusionModule(nn.Module):
    def __init__(self, encoder_config, decoder_config, model_config, diffusion_config, timestep_sampler, loss_kwargs, device='cpu'):
        super(LatentDiffusionModule, self).__init__()
        self.Encoder = get_encoder(encoder_config['encoder_model'], encoder_config['encoder_kwargs'], ckpt=encoder_config['ckpt'])
        self.Decoder = get_decoder(decoder_config['decoder_model'], decoder_config['decoder_kwargs'], ckpt=decoder_config['ckpt'])
        self.Model = get_model(model_config['model'], model_config['model_kwargs'])
        self.device=device
        self.Encoder.to(device)
        self.Decoder.to(device)
        self.Model.to(device)

        diffkw = diffusion_config['diffusion_kwargs']
        self.Diffusion = Diffusion(**diffkw)
        self.sampler = timestep_sampler
        self.loss_kwargs = loss_kwargs

    def forward(self, x, labels, itn, custom_t=None):
        '''
        :param x: x is the field WITHOUT normalization and WITHOUT noise added
        :param labels: Should already have been normalized
        :return:
        '''
        #encoder: transform x to diff space
        z = self.Encoder(x)
        #reconstruction loss
        x_recon = self.Decoder(z, labels)
        loss_recon = self.Decoder.loss(x, x_recon)

        #diffusion loss
        batch_size = x.shape[0]
        if custom_t is None:
            t = self.sampler.get_timesteps(batch_size, itn).to(torch.long) # [0, T-1]
        else:
            t = custom_t
        diff_loss = self.Diffusion.p_losses(self.Model, z, t, loss_type=self.loss_kwargs['diff_loss_type'], labels=labels)

        #prior loss
        var1 = self.Diffusion.sqrt_one_minus_alphas_cumprod[-1]**2 #final var
        prior_loss = 0.5* (((torch.tensor(1.0) - var1) * z**2) + var1 - torch.tensor(1.0) - torch.log(var1)).mean()


        if self.sampler.type == 'loss_aware':
            with torch.no_grad():
                loss_timewise = self.Diffusion.timewise_loss(self.Model, z, t,
                                            loss_type=self.loss_kwargs['diff_loss_type'], labels=labels)
                self.sampler.update_history(t, loss_timewise)

        #sum losses
        loss = diff_loss + self.loss_kwargs['recon_weight']*loss_recon + self.loss_kwargs['prior_weight']*prior_loss
        return {'loss': loss, 'diffusion_loss': diff_loss, 'recon_loss': loss_recon, 'prior_loss': prior_loss}

    @torch.no_grad()
    def sample(self, labels, p_sample_args):
        z0_samps = self.Diffusion.p_sample_loop_mem_efficient(self.Model, labels=labels, **p_sample_args)
        z0_samps = torch.tensor(z0_samps).to(self.device) 
        x_samps = self.Decoder(z0_samps, labels)
        return x_samps


if __name__ =='__main__':
    import yaml
    import main_helper
    with open('../config/e1_nx64.yaml', 'r') as stream:
        config_dict = yaml.safe_load(stream)
    Nx= 64
    model_kwargs = main_helper.get_default_model_kwargs(Nx, 1, config_dict)
    model = UnetExplicitConditional(**model_kwargs)
    B=5
    xt = torch.randn((B, 1, Nx, Nx))
    time = torch.randint(low=0, high=999, size=(B,))
    label = torch.randn((B, 2))
    ytest = model(xt, time, label)
    print('ch')
    #test b2f
    import matplotlib.pyplot as plt
    '''
    B = 5
    x = torch.randn(B, 2, 256, 256)
    b2fl = Base2FourierFeatures(6, 8, 1)
    fout = b2fl(x)
    h = torch.concat([x, fout], dim=1)
    print(3)
    
    #test model output has output shape BCHW without additional channels
    Nx = 256
    model = Unet(dim=64, dim_mults=[1, 2, 4, 8], channels=1, use_convnext=False, init_conv=False,
                conditional_dim=2, base2fourier=True, b2fstart=6, b2fend=8, b2fstep=1)
    #model = Unet(dim=64, dim_mults=[1, 2, 4, 8], channels=1, use_convnext=False, init_conv=False,
    #                conditional_dim = 2)
    
    buff = torch.randn((B, 1, Nx, Nx))
    tbu = torch.randint(0, 1000, (B,))
    labe = torch.rand(B, 2)
    buout = model(buff, tbu, labe)
    print('r3')

    '''
    '''
    #Look at which channels look meaningful
    #Nx=256
    trainfields = pickle.load(open('../../../fields/trainfields.pkl', 'rb'))
    #genfields = pickle.load(open('../../../fields/noguid_3ckp.pkl', 'rb'))
    train_main = torch.from_numpy(trainfields[0][1]).unsqueeze(1)
    #Nx=64
    #trainfields = np.load('../../testmodels/normfields64.npy')
    #train_main = torch.from_numpy(trainfields)
    #gen_main = genfields['samples'][1][:15]
    b2fl = Base2FourierFeatures(-3, 2, 1)
    freqs = torch.arange(b2fl.start, b2fl.stop + 1, b2fl.step)
    trainff = b2fl(train_main)
    #genff = b2fl(gen_main)
    Nx = train_main.shape[-1]
    #Nx=256: -3, 2
    #Nx=64, -1, 4??
    for i in range(trainff.shape[0]):
        fig, ax = plt.subplots(nrows=2, ncols=len(freqs)+1)
        for r in range(len(freqs)):
            ax[0, r].imshow(trainff[i, r])
            ax[0, r].set_title('Freq={} k~{:.0f}'.format(freqs[r], Nx*(2.0**freqs[r])))
            ax[1, r].imshow(trainff[i, len(freqs)+r])
            ax[1, r].set_title('Freq={} k~{:.0f}'.format(freqs[r], Nx * (2.0 ** freqs[r])))
            #ax[r, 1].set_title('Gen Filter{} k={:.1f}'.format(r, Nx*(2.0**freqs[r//len(freqs)])))
        ax[0, -1].imshow(train_main[i, 0])
        ax[0, -1].set_title('Train Field')
        plt.delaxes(ax[1, -1])
        #ax[-1, 1].imshow(gen_main[i, 0])
        fig.suptitle('True Image and Fourier Features: Sine in Upper Row, Cosine in Bottom Row')
        plt.show()
    '''
    #Unit test pytorch pk and numpy consistency: Moved to notebook 3_25
    '''
    trainfields = pickle.load(open('../../../fields/trainfields.pkl', 'rb'))
    input_image = trainfields[0][1][:4]
    pkcomp = Log10Pk_Compute()
    kvals, pklist = evaluate.get_powspec_for_samples([input_image])
    pknumpy = pklist[0]
    pkpt = pkcomp(torch.tensor(input_image).unsqueeze(1))
    

    
    plt.figure()
    plt.plot(kvals, kvals**2*)
    
    '''
    print(4)

