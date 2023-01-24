"""
Functions for Euclidean space
"""

# Base of math operations and derivatives
from jax import numpy as jnp
# General functions for manifolds
from geomjax.manifolds.utils import Manifold


class Euclidean(Manifold):
    """
    Euclidean space - ordinary space 
    where retraction and projection operations are identity operation
    """
    def __init__(self):
        identity = lambda m, s : s
        self.projection = identity
        self.retraction = identity
        self.distance = euclidean_distance


def euclidean_distance(A, B):
    """
    Standard euclidean distance
    A : point from Euclidean space
    B : point from Euclidean space
    ord : resulting value will be powered by this parameter
    """
    squared_dist = jnp.inner(A - B, A - B).squeeze()
    return jnp.square(squared_dist)
        