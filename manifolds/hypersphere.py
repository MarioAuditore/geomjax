"""
Functions for Hypersphere manifold
"""

# For proper backprop with custom classes
# =======================================
# Source: https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree
# Source: https://www.kaggle.com/code/aakashnain/tf-jax-tutorials-part-10-pytrees-in-jax
from jax import tree_util, jit

# Base of math operations and derivatives
from jax import numpy as jnp
# General functions for manifolds
from geomjax.manifolds.utils import Manifold

@jit
def projection(x, s):
    """
    Projection from ambient space to tangent space at x
    x - point on a sphere
    s - vector from ambient space
    """
    # Get normal vector as radius
    n = x / jnp.linalg.norm(x)
    # Find projection
    return s - jnp.dot(s - x, n) * n 
    # return s - jnp.dot(s, n) * n 


@jit
def retraction(x, a):
    """
    Central projection on sphere surface
    x - point on a sphere
    a - vector from tangent space at x
    """
    # step forward
    p = x + a
    return p * (jnp.linalg.norm(x) / jnp.linalg.norm(p))
  

@jit
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

@jit
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
    

class Hypersphere(Manifold):
    """
    Hypersphere - a set of points with equal norms
    """
    def __init__(self, projection = projection, retraction = retraction, distance = arctan_distance, random_generator = None):
        self.projection = projection
        self.retraction = retraction
        self.distance = distance

        self.key = None
        self.random_generator = random_generator


tree_util.register_pytree_node(Hypersphere,
                               Hypersphere._tree_flatten,
                               Hypersphere._tree_unflatten)
