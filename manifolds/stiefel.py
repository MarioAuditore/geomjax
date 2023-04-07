"""
Functions for the manifold of 
Orthogonal matrices
"""

# For proper backprop with custom classes
# =======================================
# Source: https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree
# Source: https://www.kaggle.com/code/aakashnain/tf-jax-tutorials-part-10-pytrees-in-jax
from jax import tree_util

# Base of math operations and derivatives
from jax import numpy as jnp
# General functions for manifolds
from geomjax.manifolds.utils import Manifold


def projection(M, S):
    """
    Projection from ambient space to tangent space at x
    M - point on a manifold
    S - vector from ambient space
    """
    W_hat = S @ M.T - M @ M.T @ S @ M.T / 2
    return (W_hat - W_hat.T) @ M


def retraction(M, T):
    u, _, vh = jnp.linalg.svd(M + T, full_matrices=False)
    return u @ vh


def distance(X, Y, base = None):
    '''
    Stiefel distance (geodesic) as 
    Riemannian metric on Stiefel Tangent space

    Params:
    -------
    X, Y : jax.numpy.array
        Objects from the manifold
    base : torch.tensor
        Point from the manifold for local projectiion

    Returns:
    : jax.numpy.array
        Point on a manifold
    '''
    if base == None:
        base = Y

    X_T = projection(X, base)
    Y_T = projection(Y, base)
    return jnp.trace(X_T.T.mm(Y_T))
