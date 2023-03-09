"""
Functions for Euclidean space
"""

# Base of math operations and derivatives
from jax import numpy as jnp
# General functions for manifolds
from geomjax.manifolds.utils import Manifold, tree_util


def euclidean_distance(A, B):
    """
    Standard euclidean distance
    A : point from Euclidean space
    B : point from Euclidean space
    ord : resulting value will be powered by this parameter
    """
    squared_dist = jnp.inner(A - B, A - B).squeeze()
    return jnp.square(squared_dist)



class Euclidean(Manifold):
    """
    Euclidean space - ordinary space 
    where retraction and projection operations are identity operation
    """
    def __init__(self, projection = None, retraction = None, distance = euclidean_distance):
        identity = lambda m, s : s
        
        if projection == None:
            self.projection = identity
        else:
            self.projection = projection

        if retraction == None:
            self.retraction = identity
        else:
            self.retraction = retraction
        
        self.distance = distance

tree_util.register_pytree_node(Euclidean,
                               Euclidean._tree_flatten,
                               Euclidean._tree_unflatten)