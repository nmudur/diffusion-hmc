import torch


#Code based on https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py

def get_mse_model(truth, model_out, mask_params):
    '''
    :param truth:
    :param model_out:
    :param mask_params: mask selecting params under consideration
    :return:
    '''
    #assert truth.shape[-1]==6, print('Wrong GT Label Shape', truth.shape[-1])
    assert (model_out.shape[-1]%6==0)
    diff = (truth - model_out[:, :6][:, mask_params]) #Bx2
    return (diff**2).mean(dim=-1) #Bx2-> MSE B

def cond_fn(x, t, y, mask_params, classifier, classifier_scale):
    #Need to write this so that you have a metric that needs to be MAXIMIZED (gradient ascent)
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        model_out = classifier(x_in, t)
        mse = get_mse_model(y, model_out, mask_params)
        msesum = -mse.sum() #Added - sign here
        batched_jac = torch.autograd.grad(msesum, x_in)[0]
        #print(msesum, batched_jac.mean())
        return batched_jac*classifier_scale

def cond_fn_time_independent(x, y, mask_params, classifier, classifier_scale):
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        model_out = classifier(x_in)
        mse = get_mse_model(y, model_out, mask_params)
        msesum = mse.sum()
        batched_jac = torch.autograd.grad(msesum, x_in)[0]
        return batched_jac*classifier_scale

def correct_using_gradients_alone(morphed_data, cond_fn, classifier_args, num_iterations, learning_rate, time_dependent=True, verbose=False):
    if time_dependent:
        morphed_input, t, y = morphed_data
    else:
        morphed_input, y = morphed_data
    mask_params, classifier, classifier_scale = classifier_args['mask_params'], classifier_args['classifier'], classifier_args['classifier_scale']
    corr_input = morphed_input
    for i in range(num_iterations):
        if verbose and (i%10==0):
            print('Iteration', i)
        if time_dependent:
            corr_input = corr_input - learning_rate*cond_fn(corr_input, t, y, mask_params, classifier, classifier_scale)
        else:
            corr_input = corr_input - learning_rate*cond_fn_time_independent(corr_input, y, mask_params, classifier, classifier_scale)
    return corr_input

def condition_mean(cond_fn, p_mean_var, x, t, model_kwargs=None):
    """
    Compute the mean for the previous step, given a function cond_fn that
    computes the gradient of a conditional log probability with respect to
    x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
    condition on y.
    This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
    This is being applied to (x_t, t) in guided_diffusion.py i.e. we technically never evaluate this for x_0
    during conditioning.
    Model accepts [0, T] as timesteps and 0 is considered to be the no-noise limit. i.e.
    <beta_timestep_index> + 1 = <classifier_timestep>
    t: beta_timestep_index
    model_kwargs: y, mask_params, classifier, classifier_scale
    """
    rem_keys = ['cond_fn']
    model_kw = {k: model_kwargs[k] for k in set(list(model_kwargs.keys())) - set(rem_keys)}
    gradient = cond_fn(x, t+1, **model_kw)
    new_mean = (
            p_mean_var["mean"].float() - p_mean_var["variance"] * gradient.float() #the original code has + but replacing that with minus to mimic gradient DESCENT (since we wanna minimize MSE)
    )
    return new_mean


