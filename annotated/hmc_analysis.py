"""HMC Analysis utilities."""
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt

def r_hat(matlist): #list of files for each chain
    N, D, M=matlist[0].shape[0], matlist[0].shape[1], len(matlist)
    print (N, D, M)

    ch_mean_list =[np.mean(mat, axis=0) for mat in matlist]         #mean of each chain
    sm_list=[(np.var(mat, axis=0)*N / (N-1)) for mat in matlist]    #var of each chain
    w_list= np.sum(np.array(sm_list), axis=0)/M                     #avg within chain variance, Needs to be adding across chains, not params--check!, shape=D
    all_mean=np.mean(np.array(ch_mean_list), axis=0)                #mean of means of all chains, Needs to be adding across chains, not params--check!
    diff=np.array(ch_mean_list) - all_mean                          #mean of each chain - mean of all

    b_list=np.sum(np.power(diff, 2), axis=0)*N/(M-1) #shape=D       #variance of within chain means
    var_est=w_list*(N-1)/N + b_list/N #shape=D                      #v^
    r_hat=np.sqrt(var_est/w_list) #shape=D
    return r_hat

def effective_sample_size(data): # Copilot
    """Calculate the Effective Sample Size (ESS) of a sequence of samples."""
    n = len(data)
    # Calculate the mean of the data
    mean_data = np.mean(data)
    # Calculate the variance of the data
    var_data = np.var(data, ddof=1)
    # Autocorrelation and ESS calculation
    t = 1
    total_autocorr = 0.0
    while t < n:
        autocorr_t = np.corrcoef(data[:n-t], data[t:])[0, 1]
        if autocorr_t <= 0:
            break
        total_autocorr += autocorr_t
        t += 1
    ess = n / (1 + 2 * total_autocorr)
    return ess, np.sqrt(var_data / ess)

def cov_ellipse(cov, q=None, nsig=None, **kwargs): #Source: https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
    """
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations.
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)  # nstd^2 = Convert %age (q) for 2 dof to the r2 for this ppf.

    val, vec = np.linalg.eigh(cov)
    width, height = 2 * np.sqrt(val[:, None] * r2) 
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    return width, height, rotation

def plot_confidence_ellipses_CL(mat, bool_save, dir_save, title, trueparam):  # Confidence level ellipses
    params = [r'$Omega_m$', r'$\sigma_8$']
    print (params)
    fig, ax = plt.subplots()
    sigcolors=[[0.683, 'red', '-'], [0.954, 'green', '--'], [0.997, 'black', ':']]
    ax.scatter(x=mat[:, 0], y=mat[:, 1], s=0.25, color='blue')
    ax.scatter(trueparam[0], trueparam[1], marker='x', color='k')
    covar=np.cov(mat.T)
    for scomb in sigcolors:
        w, h, r=cov_ellipse(covar, scomb[0])
        ellipse = Ellipse((np.mean(mat[:, 0]), np.mean(mat[:, 1])), width=w, height=h, angle=r, edgecolor=scomb[1], linestyle=scomb[2], fill=False)
        ax.add_patch(ellipse)
    ax.set_xlabel(params[0])
    ax.set_ylabel(params[1])
    fig.suptitle(title)
    if bool_save:
        plt.savefig(dir_save)
    plt.show()
    return