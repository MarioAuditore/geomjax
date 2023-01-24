"""
General functions for all manifolds
"""

# Base of math operations and derivatives
from jax import numpy as jnp
# For batch functions
from jax import vmap



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

    def project(self, M, S):
        """
        Local projection operation on manifold tangent space
        M : point from the manifold
        S : point from ambient space
        """
        return self.projection(M, S)

    def retract(self, M, T):
        """
        Retraction operation from tangent space to manifold
        M : point from the manifold
        T : point from tangent space
        """
        return self.retraction(M, T)

    def calculate_distance(self, X, Y):
        
        """
        Calculate distance between two points on the manifold
        X : point from the manifold
        Y : point from the manifold
        """
        return self.distance(X, Y)

    def step_forward(self, base, direction):
        """
        Optimization step on manifold
        base : point from the manifold
        direction : gradient descent direction
        """
        return self.retract(base, base + direction)


def pairwise_distance(Y, X_set, manifold, weights = None, ord = 2):
    # make a batch-friendly version of distance function
    batch_distance = vmap(manifold.distance, (0, None), 0)
    # calculate pairwise distances
    distances = batch_distance(X_set, Y) ** ord
    # sum them
    if weights == None:
        return jnp.sum(distances)
    else:
        return jnp.sum(jnp.inner(weights, distances).squeeze())


