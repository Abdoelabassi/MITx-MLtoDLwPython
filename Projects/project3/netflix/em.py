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
    mu, var, weight = mixture
    n, _ = X.shape
    k, _ = mu.shape
    prob_mat = np.zeros([n, k])
    prob_all = np.zeros(n)
    for i in range(k):
        prob = weight[i] * multivariate_normal(mu[i], var[i]).pdf(X)
        prob_mat[:,i] = prob
        prob_all += prob
    post = prob_mat / np.tile(prob_all.reshape(n, 1), (1, k))
    log_likelihood = np.sum(np.log(np.sum(prob_mat, axis = 1)))
    return post, log_likelihood




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
    n,d = X.shape
    _, k = post.shape

    n_clusters = np.einsum("ij->j", post)
    weight = n_clusters/n
    mu = post.T @ X/n_clusters.reshape(k,1)
    var = np.zeros(k)

    for j in range(k):
        var[j] = np.sum(post[:,j].T @ (X - mu[j])**2)/(d*n_clusters[j])
    
    return GaussianMixture(mu, var,weight)




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
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
