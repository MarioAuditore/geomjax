"""
====================================
--- Manifold friendly optimisers ---
====================================
"""
# Vectorization
from jax import vmap
# Base of math operations and derivatives
from jax import numpy as jnp
# Custom derivative declaration
from jax import grad, jacobian, jit
# Jax functions for jit
from functools import partial
# Flax functions for parameter manipulation
from flax.core import freeze, unfreeze
# For detecting arrays
from jaxlib.xla_extension import ArrayImpl
# Plotting
import matplotlib.pyplot as plt

'''
1) Для производных по аргументам среднего
учесть, что надо брать проекцию после выдачи неявной производной

2) Сделать все через vmap, чтобы была батчи
'''

class GeometricOptimiser():
    def __init__(self, manifold, lr, lr_schedule = None):
        '''
        lr_schedule = {'freq' : int, 'multiplier': float}
        or lr_schedule = [lr_2, ..., lr_n]
        '''
        
        self.lr = lr
        self.manifold = manifold
        self.lr_schedule = lr_schedule
        self.counter = 0


    def update_lr(self):
        if type(self.lr_schedule) is dict:
            if self.counter % self.lr_schedule['freq'] == 0:
                self.lr *= self.lr_schedule['multiplier']
        else:
            self.lr = self.lr_schedule[counter]

    
    @partial(jit, static_argnums=(0,))
    def init(self, params):
        # re-init iterations counter
        self.counter = 0
        # Check if params are dict
        # if type(params) is dict:
        try:
            state = {}
            # Iterate over each layer
            for layer in params.keys():
                # If layer contains a dict of params
                # if type(params[layer]) is dict:
                try:
                    state[layer] = {}
                    for param in params[layer].keys():
                        state[layer][param] = self.init_state_params(params[layer][param])

                # If layer just stores an array
                # else:
                except:
                    state[layer] = self.init_state_params(params[layer])

        # If params are just an array                        
        # else
        except:
            state = self.init_state_params(params)
        

        return state


    @partial(jit, static_argnums=(0,))
    def update(self, params, euclid_grads, state):
        
        def perform_update(param, euclid_grad, state):
            gradient, state = self.total_grad(param, euclid_grad, state)
            param = self.manifold.retract(param, gradient)
            if self.lr_schedule:
                self.update_lr()
            return param, state

        # if type(params) is dict:
        try:
            params = unfreeze(params)
            for layer in params.keys():
                # if type(params[layer]) is dict:
                try:
                    for param in params[layer].keys():
                        if len(params[layer][param].shape) == 3:
                            params[layer][param], state[layer][param] = vmap(perform_update, (0,0,0),0)(params[layer][param], euclid_grads[layer][param], state[layer][param])
                        else:
                            params[layer][param], state[layer][param] = perform_update(params[layer][param], euclid_grads[layer][param], state[layer][param])
                # else:
                except:
                    if len(params[layer].shape) == 3:
                        params[layer], state[layer] = vmap(perform_update, (0,0,0),0)(params[layer], euclid_grads[layer], state[layer])
                    else:
                        params[layer], state[layer] = perform_update(params[layer], euclid_grads[layer], state[layer])   
        # else:
        except:
            if len(params.shape) == 3:
                params, state = vmap(perform_update, (0,0,0),0)(params, euclid_grads, state)
            else:
                params, state = perform_update(params, euclid_grads, state)   
        
        self.counter += 1
        return params, state


        



'''
Based on: https://medium.com/konvergen/momentum-method-and-nesterov-accelerated-gradient-487ba776c987
'''
class MomentumGrad(GeometricOptimiser):
    
    def __init__(self, manifold, lr = 1e-1, gamma = 1e-1, lr_schedule = None):
        
        self.gamma = gamma
        self.lr = lr
        self.manifold = manifold
        self.lr_schedule = lr_schedule
        self.counter = 0


    @partial(jit, static_argnums=(0,))
    def init_state_params(self, param):
        return {'momentum' : jnp.zeros_like(param)}


    @partial(jit, static_argnums=(0,))
    def total_grad(self, param, euclid_grad, state):
        
        # Tangent projection for Riemannian gradient
        riem_grad = self.manifold.project(param, euclid_grad)
        
        # Add momentum
        total_grad = -self.lr * riem_grad + self.gamma * state['momentum']
        
        # Save momentum
        state['momentum'] = total_grad
        # Return grad and state
        return total_grad, state

    # @partial(jit, static_argnums=(0,))
    # def step(self, param, euclid_grad, state=None):
        
    #     # Tangent projection for Riemannian gradient
    #     riem_grad = self.manifold.project(param, euclid_grad)
        
    #     # Add momentum
    #     if state is None:
    #         state = {}
    #         total_grad = -self.lr * riem_grad
    #     else:
    #         total_grad = -self.lr * riem_grad + self.gamma * state['momentum']
        
    #     # Save momentum
    #     state['momentum'] = total_grad
    #     # Update param
    #     param_updated = self.manifold.retract(param, total_grad)
    #     # Update learning rate
    #     self.decrease_lr()
    #     # Return result
    #     return param_updated, state


# class GradientDescent(GeometricOptimiser):
    
#     @partial(jit, static_argnums=(0,))
#     def step(self, param, euclid_grad, state=None):
        
#         # Tangent projection for Riemannian gradient
#         riem_grad = self.manifold.project(param, euclid_grad)
#         # Update param
#         param_updated = self.manifold.retract(param, -self.lr * riem_grad)
#         # Update learning rate
#         self.decrease_lr()
#         # Return result
#         return param_updated

