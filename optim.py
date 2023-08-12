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
            self.lr *= self.lr_schedule

    
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
    
    def __init__(self, manifold, lr = 1e-1, gamma = 0.9, lr_schedule = None):
        
        self.gamma = gamma
        self.lr = lr
        self.manifold = manifold
        self.lr_schedule = lr_schedule
        self.counter = 0


    def init_state_params(self, param):
        self.counter = 0
        return {'momentum' : jnp.zeros_like(param)}


    def total_grad(self, param, euclid_grad, state):
        
        # Tangent projection for Riemannian gradient
        riem_grad = self.manifold.project(param, euclid_grad)
        
        # Add momentum
        # total_grad = riem_grad + self.gamma * state['momentum']
        if self.counter == 0:
            total_grad = riem_grad
        else:
            total_grad = (1 - self.gamma) * riem_grad + self.gamma * state['momentum']
        
        # Save momentum
        state['momentum'] = total_grad
        # Return grad and state
        return -self.lr * total_grad, state



class Adam(GeometricOptimiser):
    '''
    Under cinstruction, currently it's just crap
    '''
    
    def __init__(self, manifold, lr = 1e-1, beta_1 = 0.9, beta_2 = 0.99, eps = 1e-3, lr_schedule = None):
        
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.lr = lr
        self.eps = eps
        self.manifold = manifold
        self.lr_schedule = lr_schedule
        self.counter = 0


    def init_state_params(self, param):
        self.counter = 0
        return {'m' : jnp.zeros_like(param), 
                'v' : jnp.zeros_like(param)}


    def total_grad(self, param, euclid_grad, state):
        
        # Tangent projection for Riemannian gradient
        riem_grad = self.manifold.project(param, euclid_grad)
        
        # Statistics
        if self.counter == 0:
            state['m'] = riem_grad
            state['v'] = riem_grad @ riem_grad.T # попробовать имитировать квадрат
        else:
            state['m'] = self.beta_1 * state['m'] + (1 - self.beta_1) * riem_grad
            state['v'] = self.beta_2 * state['v'] + (1 - self.beta_2) * riem_grad @ riem_grad.T # попробовать имитировать квадрат

        # Bias correction
        if self.counter == 0:
            m_corrected = state['m']
            v_corected = state['v']
        else:
            m_corrected = state['m'] / (1 - self.beta_1 ** self.counter)
            v_corected = state['v'] / (1 - self.beta_2 ** self.counter)

        total_grad = -self.lr * m_corrected / (jnp.sqrt(jnp.linalg.norm(v_corected)) + self.eps)
        
        # print(f'm_corrected: {jnp.linalg.norm(m_corrected)} v_corrected: {jnp.sqrt(jnp.linalg.norm(v_corected))} total_grad {jnp.linalg.norm(total_grad)}')
        
        # Return grad and state
        return total_grad, state
