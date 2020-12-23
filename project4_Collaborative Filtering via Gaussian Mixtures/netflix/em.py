"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
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
    mu, var, pi = mixture
    K = mu.shape[0]

######## Vectorized version to calculate norms ########
    """
Here is the skelton :

  1.create 'non_null_index a array (n,d) with one at every movie rated by the user.
  2.dim = np.sum(non_null_index, axis=1) will give the number of movies rated by users dim (n,)
  3.non_null_index[:,None,:] * mu will have mask non-rated movie's mu to be zero. note the result is an array(n,k,d)
  4.X[:,None,:] - non_null_index[:,None,:] * mu can be squared and summed up along the appropriate axis to give the norm of all the users. Note this will be an array(n,k) - user and feature
  5.when you divide the result of 4 by var[None, :] * 2 you get the result of  𝑙𝑜𝑔(𝑒−‖𝑥−𝜇‖2/2𝜎2)  for all the users and features. an array (n,k) with values of  −‖𝑥−𝜇‖2/2𝜎2 
  6.Use dim[:, None] , np.log(var[None, :] , logp[None, :] - all of which will be (n,k) - appropriately to get a single step norms array (n,k) having the result of  𝑙𝑜𝑔(𝑓𝑋(𝑥|𝜇,𝜎2))  where  𝑓𝑋(𝑥|𝜇,𝜎2)=1(2𝜋𝜎2)𝑑/2𝑒−‖𝑥−𝜇‖2/2𝜎2  for all users, all featuers.
  7.norm_max = np.max(norms, axis=1, keepdims=True) should give the max norm for subtracting before taking np.logsumexp to avoid blow ups. and added the max_norm after taking np.logsumexp.
B y doing it without for loops, this code runs one single e-step in less than .8 sec for 1200 user with 1200 movie in Netflix incomplete data set.
    """
    # Create a delta matrix to indicate where X is non-zero, which will help us pick Cu indices
    delta = X.astype(bool).astype(int)
    # Exponent term: norm matrix/(2*variance)
    f = (np.sum(X**2, axis =1)[:, None] + (delta @ mu.T**2) - 2*(X@mu.T))/ (2*var)
    # Pre-exponent term: A matrix of shape (n, K)
    pre_exp = (-np.sum(delta, axis =1).reshape(-1,1)/2.0) @ (np.log((2*np.pi*var)).reshape(-1,1)).T
    #Put them together
    f = pre_exp -f
####### End vectorized version##############

    f = f + np.log(pi + 1e-16)
    
    #log of normalizing term in p(j|u)
    logsums = logsumexp(f, axis=1).reshape(-1,1) # Store this to calculate log_lh
    log_posts = f- logsums #This is the log of posterior prob. matrix: log(p(j|u))

    log_lh = np.sum(logsums, axis=0).item() # This is the log likelihood

    return np.exp(log_posts), log_lh




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
    n ,d = X.shape
    mu_rev, _, _ = mixture
    K = mu_rev.shape[0]

    #Calculate the revised p(j|i)
    pi_rev = np.sum(post, axis=0)/n

    #Create delta matrix indicating where X is non-zero
    delta = X.astype(bool).astype(int)

    #Update means only when sum_u(p(j|u)*delta(l,Cu)) >= 1
    denom = post.T @ delta  # Denominator (K,d): Only include dims that have information
    numer = post.T @ X  # Numerator (K,d)
    update_indices = np.where(denom >= 1)
    mu_rev[update_indices] = numer[update_indices]/ denom[update_indices] 

    #Update variances
    denom_var = np.sum(post*np.sum(delta, axis =1).reshape(-1,1),axis =0)

######## Vectorized version for norms calc. ########
    norms = np.sum(X**2, axis=1)[:,None] + (delta @ mu_rev.T**2) - 2*(X @ mu_rev.T)

######## End: vectorized version #########
    # Revised var: if var(j) < 0.25, set it = 0.25
    var_rev = np.maximum(np.sum(post*norms, axis=0)/denom_var, min_variance)

    return GaussianMixture(mu_rev, var_rev, pi_rev)


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
    old_log_lh = None
    new_log_lh = None

    #Start the main loop 
    while old_log_lh is None or (new_log_lh - old_log_lh > 1e-6*np.abs(new_log_lh)):
        old_log_lh = new_log_lh

        #E-step
        post, new_log_lh =estep(X, mixture)

        #M-step
        mixture = mstep(X, post, mixture)

    return mixture, post, new_log_lh

def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    X_pred = X.copy()
    mu, _, _ = mixture

    post, _ = estep(X, mixture)

    # Missing entries to be filled
    miss_indices = np.where(X == 0)
    X_pred[miss_indices] = (post @ mu)[miss_indices]

    return X_pred
