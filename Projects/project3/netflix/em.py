"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, multivariate_normal
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """ 
    n, d = X.shape
    mu, var, p = mixture 
    K = mu.shape[0]
    
    
    delta = X.astype(bool).astype(int)
   
    f = (np.sum(X**2, axis=1)[:,None] + (delta @ mu.T**2) - 2*(X @ mu.T))/(2*var)
    
    pre_exp = (-np.sum(delta, axis=1).reshape(-1,1)/2.0) @ (np.log((2*np.pi*var)).reshape(-1,1)).T
   
    f = pre_exp - f
    
    
    f = f + np.log(p + 1e-16)
    
    # log of normalizing term in p(j|u)
    logsums = logsumexp(f, axis=1).reshape(-1,1)
    log_posts = f - logsums 
    
    log_L = np.sum(logsums, axis=0).item()   
    
    return np.exp(log_posts), log_L




def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    mu_rev, var_rev, p_rev = mixture
    K = mu_rev.shape[0]
    
   
    p_rev = np.sum(post, axis=0)/n
    
    delta = X.astype(bool).astype(int)
    
    denom = post.T @ delta 
    numer = post.T @ X  
    update_indices = np.where(denom >= 1)
    mu_rev[update_indices] = numer[update_indices]/denom[update_indices]     
    denom_var = np.sum(post*np.sum(delta, axis=1).reshape(-1,1), axis=0) 
    
    norms = np.sum(X**2, axis=1)[:,None] + (delta @ mu_rev.T**2) - 2*(X @ mu_rev.T)
   
    var_rev = np.maximum(np.sum(post*norms, axis=0)/denom_var, min_variance)  
    
    return GaussianMixture(mu_rev, var_rev, p_rev)




def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_log_L = None
    new_log_L = None
    
    while old_log_L is None or (new_log_L - old_log_L > 1e-6*np.abs(new_log_L)):
        
        old_log_L = new_log_L
        
        # E-step
        post, new_log_L = estep(X, mixture)
        
        # M-step
        mixture = mstep(X, post, mixture)
            
    return mixture, post, new_log_L



def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
