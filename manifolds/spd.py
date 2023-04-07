"""
Functions for the manifold of 
Symmetric Positive Definite matrices
"""

# For proper backprop with custom classes
# =======================================
# Source: https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree
# Source: https://www.kaggle.com/code/aakashnain/tf-jax-tutorials-part-10-pytrees-in-jax
from jax import tree_util

# Base of math operations and derivatives
from jax import numpy as jnp
from jax import scipy as sp
# General functions for manifolds
from geomjax.manifolds.utils import Manifold


# Matrix operations
# =================

# Implementation of matrix fractional power for pytorch
# https://discuss.pytorch.org/t/raising-a-tensor-to-a-fractional-power/93655/3
def matrix_fractional_pow(M, power):
    evals, evecs = jnp.linalg.eigh(M)
    evpow = evals ** power
    return evecs @ jnp.diag(evpow) @ evecs.T


# Matrix logarithm based on same idea as above
def matrix_log(M):
    evals, evecs = jnp.linalg.eigh(M)
    evlog = jnp.log(evals)
    return evecs @ jnp.diag(evlog) @ evecs.T


# Matrix exponent
# def matrix_exp(M):
#     evals, evecs = jnp.linalg.eigh(M)
#     evexp = jnp.exp(evals)
#     return evecs @ jnp.diag(evexp) @ evecs.T
    


def projection(M, S):
    """
    Projection from ambient space to tangent space at x
    M - point on a manifold
    S - vector from ambient space
    """
    return M @ (S + S.T) @ M / 2


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


def affine_invariant_metric(X, Y):    
    X_neg_half = matrix_fractional_pow(X, -0.5)
    return jnp.linalg.norm(matrix_log(X_neg_half @ Y @ X_neg_half))


def stein_metric(X, Y):
    det_1 = jnp.linalg.det((X + Y) / 2)
    det_2 = jnp.linalg.det(X @ Y)
    return jnp.log(det_1) - jnp.log(det_2) / 2


def log_euclidean_metric(X, Y):
    log_X = matrix_log(X)
    log_Y = matrix_log(Y)
    return jnp.linalg.norm(log_X - log_Y)


class SPD(Manifold):
    """
    SPD - manifold of symmetric positive definite matrices
    """
    def __init__(self, projection = projection, retraction = retraction, distance = log_euclidean_metric):
        self.projection = projection
        self.retraction = retraction
        self.distance = distance


tree_util.register_pytree_node(SPD,
                               SPD._tree_flatten,
                               SPD._tree_unflatten)
