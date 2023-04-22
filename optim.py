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
    def __init__(self, lr, maxiter, manifold, lr_reducer = None):
        
        self.lr = lr
        self.maxiter = maxiter
        self.manifold = manifold
        self.lr_reducer = lr_reducer

    def decrease_lr(self):
        if self.lr_reducer:
            return self.lr * self.lr_reducer
        else:
            return self.lr

class GradientDescent(GeometricOptimiser):
    
    @partial(jit, static_argnums=(0,))
    def step(self, param, euclid_grad):
        
        # Tangent projection for Riemannian gradient
        riem_grad = self.manifold.project(param, euclid_grad)
        # Update param
        param_updated = self.manifold.retract(param, -self.lr * riem_grad)
        # Update learning rate
        self.lr = self.decrease_lr()
        # Return result
        return param_updated


class MomentumGrad(GeometricOptimiser):
    
    def __init__(self, lr, gamma, maxiter, manifold, lr_reducer = None):
        
        self.lr = lr
        self.gamma = gamma
        self.maxiter = maxiter
        self.manifold = manifold
        self.lr_reducer = lr_reducer

    @partial(jit, static_argnums=(0,))
    def step(self, param, euclid_grad, momentum=None):
        
        # Tangent projection for Riemannian gradient
        riem_grad = self.manifold.project(param, euclid_grad)
        # Add momentum
        if momentum is None:
            total_grad = riem_grad
        else:
            total_grad = riem_grad + self.gamma * momentum
        # Save momentum
        momentum = total_grad
        # Update param
        param_updated = self.manifold.retract(param, -self.lr * total_grad)
        # Update learning rate
        self.lr = self.decrease_lr()
        # Return result
        return param_updated, momentum
