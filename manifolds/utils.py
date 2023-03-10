"""
General functions for all manifolds
"""

# For proper backprop with custom classes
# =======================================
# Source: https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree
# Source: https://www.kaggle.com/code/aakashnain/tf-jax-tutorials-part-10-pytrees-in-jax
from jax import tree_util

# For defining static arguments and non diff arguments
# ====================================================
# Source: https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html
from functools import partial

# Base of math operations and derivatives
from jax import numpy as jnp
# For batch functions
from jax import vmap, jit




class Manifold():
    """
    Base class for Manifolds.
    """
    def __init__(self, projection, retraction, distance):
        """
        projection : callable
            Local projection from ambient space to tangent space
        retraction : callable
            Retraction from tangent space to manifold
        distance : callable
            Distance function used on the manifold
        """
        self.projection = projection
        self.retraction = retraction
        self.distance = distance

    @jit
    def project(self, M, S):
        """
        Local projection operation on manifold tangent space
        M : point from the manifold
        S : point from ambient space
        """
        return self.projection(M, S)

    @jit
    def retract(self, M, T):
        """
        Retraction operation from tangent space to manifold
        M : point from the manifold
        T : point from tangent space
        """
        return self.retraction(M, T)

    @jit
    def calculate_distance(self, X, Y):
        
        """
        Calculate distance between two points on the manifold
        X : point from the manifold
        Y : point from the manifold
        """
        return self.distance(X, Y)

    @jit
    def step_forward(self, base, direction):
        """
        Optimization step on manifold
        base : point from the manifold
        direction : gradient descent direction
        """
        return self.retract(base, base + direction)
    
    def _tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {'projection': self.projection, 
                    'retraction': self.retraction, 
                    'distance': self.distance}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    

# tree_util.register_pytree_node(Manifold,
#                                Manifold._tree_flatten,
#                                Manifold._tree_unflatten)


@partial(jit, static_argnames=['metric', 'ord'])
def pairwise_distance(Y, X_set, metric, weights = None, ord = 2):
    # make a batch-friendly version of distance function
    batch_distance = vmap(metric, (0, None), 0)
    # calculate pairwise distances
    distances = batch_distance(X_set, Y) ** ord
    # sum them
    if weights == None:
        return jnp.mean(distances)
    else:
        return jnp.mean(jnp.inner(weights, distances).squeeze() / jnp.sum(weights))


