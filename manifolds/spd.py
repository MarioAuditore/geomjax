"""
Functions for the manifold of 
Symmetric Positive Definite matrices
"""

# For proper backprop with custom classes
# =======================================
# Source: https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree
# Source: https://www.kaggle.com/code/aakashnain/tf-jax-tutorials-part-10-pytrees-in-jax
from jax import tree_util, lax

# Base of math operations and derivatives
from jax import numpy as jnp
from jax import scipy as sp
from jax import jit
# General functions for manifolds
from geomjax.manifolds.utils import Manifold
from sklearn.datasets import make_spd_matrix


# Matrix operations
# =================

# Implementation of matrix fractional power for pytorch
# https://discuss.pytorch.org/t/raising-a-tensor-to-a-fractional-power/93655/3
@jit
def matrix_fractional_pow(M, power):
    evals, evecs = jnp.linalg.eigh(M)
    evpow = evals ** power
    return evecs @ jnp.diag(evpow) @ evecs.T


# Matrix logarithm based on same idea as above
@jit
def matrix_log(M):
    evals, evecs = jnp.linalg.eigh(M)
    # to prevent from zeros in log
    evlog = jnp.log(evals + 1e-17)
    return evecs @ jnp.diag(evlog) @ evecs.T
    

@jit
def projection(M, S):
    """
    Projection from ambient space to tangent space at x
    M - point on a manifold
    S - vector from ambient space
    """
    return M @ (S + S.T) @ M / 2


@jit
def retraction(M, T):
    """
    Retraction from tangent space to manifold
    M - point on a manifold
    T - point from tangent space at M
    """
    
    M_half = matrix_fractional_pow(M, 0.5)
    M_neg_half = matrix_fractional_pow(M, -0.5)

    exp_res = sp.linalg.expm(M_neg_half @ T @ M_neg_half)

    return M_half @ exp_res @ M_half

@jit
def affine_invariant_metric(X, Y):    
    X_neg_half = matrix_fractional_pow(X, -0.5)
    return jnp.linalg.norm(matrix_log(X_neg_half @ Y @ X_neg_half))

@jit
def stein_metric(X, Y):
    det_1 = jnp.linalg.det((X + Y) / 2)
    det_2 = jnp.linalg.det(X @ Y)
    return jnp.log(det_1) - jnp.log(det_2) / 2

@jit
def log_euclidean_metric(X, Y):
    log_X = matrix_log(X)
    log_Y = matrix_log(Y)
    return jnp.linalg.norm(log_X - log_Y)


class SPD(Manifold):
    """
    SPD - manifold of symmetric positive definite matrices
    """
    def __init__(self, projection = projection, retraction = retraction, distance = log_euclidean_metric, random_generator = make_spd_matrix):
        self.projection = projection
        self.retraction = retraction
        self.distance = distance

        self.random_generator = random_generator

    def generate(self, *args):
        n = args[0]
        return self.random_generator(n)


tree_util.register_pytree_node(SPD,
                               SPD._tree_flatten,
                               SPD._tree_unflatten)
