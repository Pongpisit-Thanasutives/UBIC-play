import numpy as np
from scipy.signal import periodogram
from pydmd import DMD, BOPDMD
from pydmd.plotter import plot_eigs, plot_summary, plot_modes_2D
from pydmd.preprocessing.hankel import hankel_preprocessing

def periodogram_f(*arg, **kwargs):
    return periodogram(*arg, **kwargs)[-1]

def convert_to_scientific_notation(num):
    # Convert the number to scientific notation string
    num_part, exp_part = "{:e}".format(num).split('e')
    return float(num_part), int(exp_part)

def sum_squared_residuals(prediction, ground):
    # ssr = np.sum(np.abs(ground - prediction)**2)
    ssr = np.linalg.norm(ground - prediction)**2
    # ssr = float(((prediction - ground)**2).sum(axis=0))
    return ssr

def ssr2llf(ssr, nobs, epsilon=0):
    nobs2 = float(nobs/2.0)
    return -nobs2*(1+np.log(np.pi*ssr/nobs2+epsilon))

def log_like_value(prediction, ground, epsilon=0):
    nobs = len(ground)
    ssr = sum_squared_residuals(prediction, ground)
    return ssr2llf(ssr, nobs, epsilon)

def mbic(prediction, ground, nparams, epsilon=0):
    nobs = len(ground)
    llf = log_like_value(prediction, ground, epsilon)
    return -2*llf + np.log(nobs)*nparams

def power(uu):
    return ((uu*np.conj(uu))/np.prod(uu.shape))

def remove_f(uu, percent, inverse=False):
    if percent <= 0: return uu
    PSD = power(uu).real
    mask = np.ones_like(uu).astype(np.float32)
    if percent > 0:
        mask = (PSD>np.percentile(PSD, percent)).astype(np.float32)
    uuf = uu*mask
    if inverse:
        return ifft(uuf)
    return uuf

def POD(Y, n_modes=None):
    U, S, V = np.linalg.svd(Y, full_matrices=False)
    if n_modes is None: n_modes = len(S)
    return U[:, :n_modes]@np.diag(S[:n_modes])@(V[:n_modes, :])

def ROM(uu, tt, svd_rank, dmd_algo="DMD", num_trials=0, tol=None):
    assert dmd_algo in {"DMD", "BOPDMD"}
    delays = 2
    if dmd_algo == "DMD":
        optdmd = DMD(svd_rank=svd_rank, tlsq_rank=0, exact=True, opt=True)
        delay_optdmd = hankel_preprocessing(optdmd, d=delays)
        delay_optdmd.fit(uu)
    elif dmd_algo == "BOPDMD":
        if tol is None:
            optdmd = BOPDMD(svd_rank=svd_rank, num_trials=num_trials)
        else:
            optdmd = BOPDMD(svd_rank=svd_rank, num_trials=num_trials, varpro_opts_dict={'tol':tol})
        delay_optdmd = hankel_preprocessing(optdmd, d=delays)
        num_t = len(tt)-delays+1
        delay_optdmd.fit(uu, t=tt[:num_t])
    return delay_optdmd.reconstructed_data

