"""
====================================
--- Manifold friendly optimisers ---
====================================
"""

# Base of math operations and derivatives
from jax import numpy as jnp
# Custom derivative declaration
from jax import grad, jacobian, jit
# Jax functions for jit
from functools import partial
# Plotting
import matplotlib.pyplot as plt

'''
1) Для производных по аргументам среднего
учесть, что надо брать проекцию после выдаяи неявной производной

2) Сделать все через vmap, чтобы была батчи
'''

class GeometricOptimiser():
    def __init__(self, lr, maxiter, manifold):
        
        self.lr = lr
        self.maxiter = maxiter
        self.manifold = manifold


class GradientDescent(GeometricOptimiser):
    
    @partial(jit, static_argnums=(0,))
    def step(self, param, euclid_grad):
        
        # Tangent projection for Riemannian gradient
        riem_grad = self.manifold.project(param, euclid_grad)
        # Update param
        return self.manifold.step_forward(param, -self.lr * riem_grad)
