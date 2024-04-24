# Code ported from https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py
import abc
import torch
import functools
import numpy as np

from . import sde_helper

class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False, cond_kwargs=None):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn
    self.cond_kwargs = cond_kwargs

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the predictor.
    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.
    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass

class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps=None):
      super().__init__()
      self.sde = sde
      self.score_fn = score_fn
      self.snr = snr
      self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t):
      """One update of the corrector.
      Args:
        x: A PyTorch tensor representing the current state
        t: A PyTorch tensor representing the current time step.
      Returns:
        x: A PyTorch tensor of the next state.
        x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
      """
      pass

class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False, cond_kwargs=None):
    super().__init__(sde, score_fn, probability_flow, cond_kwargs)

  def update_fn(self, x, t, debug=False):
    dt = -1. / self.rsde.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, t)
    if self.cond_kwargs is not None:
        if 'grad_likelihood' in self.cond_kwargs:
            t_int = (t/self.sde.T*(self.sde.N-1)).to(int)
            grad_lik = self.cond_kwargs['grad_likelihood'](t_int)(x) #fn(x)
            constraint_term = - diffusion[:, None, None, None]**2*grad_lik
            print(torch.abs(drift).mean()/np.prod(drift.shape), torch.abs(constraint_term).mean()/np.prod(constraint_term.shape))
            rescale_factor = 1e-1*(torch.abs(drift).mean()/np.prod(drift.shape)) / (torch.abs(constraint_term).mean()/np.prod(constraint_term.shape))
            print('Rescale Factor', rescale_factor)
            drift = drift + rescale_factor*constraint_term
        elif 'likelihood' in self.cond_kwargs:
            t_int = (t / self.sde.T * (self.sde.N - 1)).to(int)
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                likfunc = self.cond_kwargs['likelihood'](t_int)(x_in) #fn(x)
                grad_lik = torch.autograd.grad(likfunc, x_in)[0]
            constraint_term = - diffusion[:, None, None, None]**2*grad_lik
            print(torch.abs(drift).mean()/np.prod(drift.shape), torch.abs(constraint_term).mean()/np.prod(constraint_term.shape))
            rescale_factor = 1e-1*(torch.abs(drift).mean()/np.prod(drift.shape)) / (torch.abs(constraint_term).mean()/np.prod(constraint_term.shape))
            print('Rescale Factor', rescale_factor)
            drift = drift + rescale_factor*constraint_term
        else:
            raise NotImplementedError
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
    if debug:
        return x, x_mean, drift-constraint_term, constraint_term
    else:
        return x, x_mean

class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps=None):
    pass

  def update_fn(self, x, t):
    return x, x


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda', cond_kwargs={}, debug=False):
  """Create a Predictor-Corrector (PC) sampler.
  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.
    cond_kwargs: Controllable generation kwargs
  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  def predictor_update_fn(x, t, model):
      '''
      :param x:
      :param t:
      :param model:
      :returns: Predictor.update_fn(x, t|model=model). Equivalent of "shared_predictor_update_fn" in sde_sampling code.
      '''
      score_func = sde_helper.get_score_fn(sde, model, train=False, continuous=continuous)
      if predictor is None:
          raise NotImplementedError
      else:
          predictor_obj = predictor(sde, score_func, probability_flow, cond_kwargs)
          return predictor_obj.update_fn(x, t, debug=debug)

  def corrector_update_fn(x, t, model):
      '''
      :param x:
      :param t:
      :param model:
      :returns: Corrector.update_fn(x, t|model=model). Equivalent of "shared_corrector_update_fn" in sde_sampling code.
      '''
      score_func = sde_helper.get_score_fn(sde, model, train=False, continuous=continuous)
      if corrector is None:
          raise NotImplementedError
      else:
          corrector_obj = corrector(sde, score_func, probability_flow)
          return corrector_obj.update_fn(x, t)

  def pc_sampler(model, return_all_timesteps=False):
    """ The PC sampler funciton.
    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      x = sde.prior_sampling(shape).to(device)
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
      all_samplest = []
      for i in range(sde.N):
        if i%100==0:
            print('Timestep', i)
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        x, x_mean = corrector_update_fn(x, vec_t, model=model)
        if debug:
            x, x_mean, drift, constraint = predictor_update_fn(x, vec_t, model=model)
        else:
            x, x_mean = predictor_update_fn(x, vec_t, model=model)
        if return_all_timesteps:
            if debug:
                all_samplest.append([x, x_mean, drift, constraint])
            else:
                all_samplest.append(inverse_scaler(x))
      if return_all_timesteps:
          return all_samplest, sde.N * (n_steps + 1)
      else:
        return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

  return pc_sampler


def get_pc_inpainter(sde, predictor, corrector, inverse_scaler, 
                      probability_flow=False, continuous=False,
                     denoise=True, eps=1e-5):
  """Create an image inpainting function that uses PC samplers.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
    corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for the corrector.
    n_steps: An integer. The number of corrector steps per update of the corrector.
    probability_flow: If `True`, predictor solves the probability flow ODE for sampling.
    continuous: `True` indicates that the score-based model was trained with continuous time.
    denoise: If `True`, add one-step denoising to final samples.
    eps: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.
  Returns:
    An inpainting function.
  """
  # Define predictor & corrector
  # Create predictor & corrector update functions
  def predictor_update_fn(x, t, model):
      '''
      :param x:
      :param t:
      :param model:
      :returns: Predictor.update_fn(x, t|model=model). Equivalent of "shared_predictor_update_fn" in sde_sampling code.
      '''
      score_func = sde_helper.get_score_fn(sde, model, train=False, continuous=continuous)
      if predictor is None:
          raise NotImplementedError
      else:
          predictor_obj = predictor(sde, score_func, probability_flow, None)
          return predictor_obj.update_fn(x, t)

  def corrector_update_fn(x, t, model):
      '''
      :param x:
      :param t:
      :param model:
      :returns: Corrector.update_fn(x, t|model=model). Equivalent of "shared_corrector_update_fn" in sde_sampling code.
      '''
      score_func = sde_helper.get_score_fn(sde, model, train=False, continuous=continuous)
      if corrector is None:
          raise NotImplementedError
      else:
          corrector_obj = corrector(sde, score_func, probability_flow)
          return corrector_obj.update_fn(x, t)

  def get_inpaint_update_fn(update_fn):
    """Modify the update function of predictor & corrector to incorporate data information."""

    def inpaint_update_fn(model, data, mask, x, t):
      with torch.no_grad():
        vec_t = torch.ones(data.shape[0], device=data.device) * t
        x, x_mean = update_fn(x, vec_t, model=model)
        masked_data_mean, std = sde.marginal_prob(data, vec_t)
        masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
        x = x * (1. - mask) + masked_data * mask
        x_mean = x * (1. - mask) + masked_data_mean * mask
        return x, x_mean

    return inpaint_update_fn

  projector_inpaint_update_fn = get_inpaint_update_fn(predictor_update_fn)
  corrector_inpaint_update_fn = get_inpaint_update_fn(corrector_update_fn)

  def pc_inpainter(model, data, mask):
    """Predictor-Corrector (PC) sampler for image inpainting.
    Args:
      model: A score model.
      data: A PyTorch tensor that represents a mini-batch of images to inpaint.
      mask: A 0-1 tensor with the same shape of `data`. Value `1` marks known pixels,
        and value `0` marks pixels that require inpainting.
    Returns:
      Inpainted (complete) images.
    """
    with torch.no_grad():
      # Initial sample
      x = data * mask + sde.prior_sampling(data.shape).to(data.device) * (1. - mask)
      timesteps = torch.linspace(sde.T, eps, sde.N)
      for i in range(sde.N):
        if i%100==0:
            print('Timestep', i) 
        t = timesteps[i]
        x, x_mean = corrector_inpaint_update_fn(model, data, mask, x, t)
        x, x_mean = projector_inpaint_update_fn(model, data, mask, x, t)

      return inverse_scaler(x_mean if denoise else x)

  return pc_inpainter
