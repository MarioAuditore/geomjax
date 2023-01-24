"""
Functions for Hypersphere manifold
"""

# Base of math operations and derivatives
from jax import numpy as jnp
# General functions for manifolds
from geomjax.manifolds.utils import Manifold


class Hypersphere(Manifold):
    """
    Hypersphere - a set of points with equal norms
    """
    def __init__(self):
        self.projection = projection
        self.retraction = retraction
        self.distance = arctan_distance




def projection(x, s):
    """
    Projection from ambient space to tangent space at x
    x - point on a sphere
    s - point from ambient space
    """
    # Get normal vector as radius
    n = x / jnp.linalg.norm(x)
    # Find projection
    return s - jnp.dot(s - x, n) * n 


def retraction(x, a):
    """
    Central projection on sphere surface
    x - point on a sphere
    a - point from tangent space at x
    """
    return a * (jnp.linalg.norm(x) / jnp.linalg.norm(a))
  

def arctan_distance(A, B):
    """
    Vector-form angular distance that mesures 
    the angle between two points on hypersphere.
    More stable all thanks to tan function.
    A : point from sphere
    B : point from sphere
    ord : resulting value will be powered by this parameter
    """
    vector_prod = jnp.cross(A, B)
    val = jnp.linalg.norm(vector_prod) / jnp.dot(A, B)
    return jnp.arctan(val)

def arccos_distance(A, B, ord = 1):
    """
    Vector-form angular distance that mesures 
    the angle between two points on hypersphere.
    Less preferable.
    A : point from sphere
    B : point from sphere
    ord : resulting value will be powered by this parameter
    """
    # Find angle
    angle = jnp.inner(A, B) / (jnp.linalg.norm(A) * jnp.linalg.norm(B))
    # To prevent from nan
    if jnp.allclose(angle, jnp.array([1.0])):
        return 0
    else:
        return jnp.arccos(angle) ** ord