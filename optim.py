"""
====================================
--- Manifold friendly optimisers ---
====================================
"""
# Vectorization
from jax import vmap
# Base of math operations and derivatives
from jax import numpy as jnp
# Flax functions for parameter manipulation
from flax.core import unfreeze


'''
1) Для производных по аргументам среднего
учесть, что надо брать проекцию после выдачи неявной производной

2) Сделать все через vmap, чтобы была батчи
'''


class GeometricOptimiser():
    def __init__(self, manifold, lr = 3e-4, scheduler = None):
        '''
        lr_schedule : optax scheduler
        '''
        
        self.lr = lr
        self.manifold = manifold
        self.scheduler = scheduler
        self.counter = 0


    def update_lr(self):
        self.lr = self.scheduler(self.counter)
        self.counter += 1

    
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


    def update(self, params, euclid_grads, state):
        
        # Update is performed via exponential map
        def perform_update(param, euclid_grad, state):
            gradient, state = self.total_grad(param, euclid_grad, state)
            param = self.manifold.retract(param, gradient)
            if self.scheduler:
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
                            params[layer][param], state[layer][param] = vmap(perform_update, (0,0,0), 0)(params[layer][param], euclid_grads[layer][param], state[layer][param])
                        # elif len(params[layer][param].shape) == 4:
                        #     params[layer][param], state[layer][param] = vmap(vmap(perform_update, (0,0,0), 0), (0,0,0), 0)(params[layer][param], euclid_grads[layer][param], state[layer][param])
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
                params, state = vmap(perform_update, (0,0,0), 0)(params, euclid_grads, state)
            else:
                params, state = perform_update(params, euclid_grads, state)   
        
        self.counter += 1
        return params, state


class SGD(GeometricOptimiser):

    def init_state_params(self, param):
        self.counter = 0
        return {}

    
    def total_grad(self, param, euclid_grad, state):
        
        # Tangent projection for Riemannian gradient
        riem_grad = self.manifold.project(param, euclid_grad)

        # Return grad and state
        return -self.lr * riem_grad, state