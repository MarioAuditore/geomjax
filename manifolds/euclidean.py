"""
Functions for Euclidean space
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


def euclidean_distance(A, B):
    """
    Standard euclidean distance
    A : point from Euclidean space
    B : point from Euclidean space
    ord : resulting value will be powered by this parameter
    """
    return jnp.linalg.norm(A - B)


def retraction(m , s):
    return m + s

class Euclidean(Manifold):
    """
    Euclidean space - ordinary space 
    where retraction and projection operations are identity operation
    """
    def __init__(self, projection = None, retraction = retraction, distance = euclidean_distance, random_generator = None):
        identity = lambda m, s : s
        
        if projection == None:
            self.projection = identity
        else:
            self.projection = projection

        self.retraction = retraction
        
        self.distance = distance

        self.key = None
        self.random_generator = random_generator
    
    # def step_forward(self, base, direction):
    #     """
    #     Optimization step on manifold
    #     base : point from the manifold
    #     direction : gradient descent direction
    #     """
    #     return self.retract(base, base + direction)

tree_util.register_pytree_node(Euclidean,
                               Euclidean._tree_flatten,
                               Euclidean._tree_unflatten)