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
# 
import numpy as np
# General functions for manifolds
from geomjax.manifolds.utils import Manifold


def generate_orthogonal(n, m):
    if n >= m:
        Q,_ = jnp.linalg.qr(np.random.randn(n, n))
        Q = Q[:m, :]
    else:
        Q,_ = jnp.linalg.qr(np.random.randn(m, m))
        Q = Q[:n, :]
    return Q


def projection_1(M, S):
    """
    Source: Optimization Algorithms on Matrix Manifolds, Absil; item 4.8.1
    Projection from ambient space to tangent space at x
    M - point on a manifold
    S - vector from ambient space
    """
    return (jnp.eye(M.shape[0]) - M @ M.T) @ S + M @ (M.T @ S - S.T @ M) / 2



def projection_2(M, S):
    """
    Projection from ambient space to tangent space at x
    M - point on a manifold
    S - vector from ambient space
    """
    W_hat = S @ M.T - M @ M.T @ S @ M.T / 2
    return (W_hat - W_hat.T) @ M


def retraction_qr(M, T):
    Q,_ = jnp.linalg.qr(M + T)
    return  Q


def retraction_svd(M, T):
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

    # X_T = projection(X, base)
    # Y_T = projection(Y, base)
    # return jnp.trace(X_T.T @ Y_T)
    return jnp.trace(X.T @ Y)


class Stiefel(Manifold):
    """
    SPD - manifold of symmetric positive definite matrices
    """
    def __init__(self, projection = projection_1, retraction = retraction_svd, distance = distance):
        self.projection = projection
        self.retraction = retraction
        self.distance = distance


tree_util.register_pytree_node(Stiefel,
                               Stiefel._tree_flatten,
                               Stiefel._tree_unflatten)
