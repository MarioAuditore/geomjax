"""
Functions for the manifold of 
Orthogonal matrices
"""

# For proper backprop with custom classes
# =======================================
# Source: https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree
# Source: https://www.kaggle.com/code/aakashnain/tf-jax-tutorials-part-10-pytrees-in-jax
from jax import tree_util, random
# Base of math operations and derivatives
from jax import numpy as jnp
# For random generation
from jax import random
# For boost
from jax import jit
# General functions for manifolds
from geomjax.manifolds.utils import Manifold




def generate_orthogonal(key, n, m):
    
    if n >= m:
        Q,_ = jnp.linalg.qr(random.uniform(key, shape=(n, n)))
        Q = Q[:m, :]
        return Q.T
    else:
        Q,_ = jnp.linalg.qr(random.uniform(key, shape=(m, m)))
        Q = Q[:n, :]
        return Q

@jit
def projection_1(M, S):
    """
    Source: Optimization Algorithms on Matrix Manifolds, Absil; item 4.8.1
    Projection from ambient space to tangent space at x
    M - point on a manifold
    S - vector from ambient space
    """
    return (jnp.eye(M.shape[0]) - M @ M.T) @ S + M @ (M.T @ S - S.T @ M) / 2


@jit
def projection_2(M, S):
    """
    Projection from ambient space to tangent space at x
    M - point on a manifold
    S - vector from ambient space
    """
    W_hat = S @ M.T - M @ M.T @ S @ M.T / 2
    return (W_hat - W_hat.T) @ M

@jit
def retraction_qr(M, T):
    Q,_ = jnp.linalg.qr(M + T)
    return  Q

@jit
def retraction_svd(M, T):
    u, _, vh = jnp.linalg.svd(M + T, full_matrices=False)
    return u @ vh

@jit
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
    def __init__(self, projection = projection_1, retraction = retraction_svd, distance = distance, random_generator = generate_orthogonal):
        self.projection = projection
        self.retraction = retraction
        self.distance = distance

        self.key = random.PRNGKey(1234)
        self.random_generator = random_generator


tree_util.register_pytree_node(Stiefel,
                               Stiefel._tree_flatten,
                               Stiefel._tree_unflatten)
