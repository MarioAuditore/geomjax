"""
General functions for all manifolds
"""

# For defining static arguments and non diff arguments
# ====================================================
# Source: https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html
from functools import partial

# Base of math operations and derivatives
from jax import numpy as jnp
# For batch functions
from jax import vmap, jit
from jax import random




class Manifold():
    """
    Base class for Manifolds.
    """
    def __init__(self, projection, retraction, distance, random_generator = None):
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

        self.key = random.PRNGKey(1234)
        self.random_generator = random_generator

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

    
    def generate(self, *args):
        self.key,_ = random.split(self.key)
        return self.random_generator(self.key, *args)

    
    def _tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {'projection': self.projection, 
                    'retraction': self.retraction, 
                    'distance': self.distance,
                    'random_generator': self.random_generator}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)



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
        return jnp.inner(weights, distances).squeeze() 