"""
============================================================
--- Gradient descent based mean with implicit derivative ---
============================================================
"""
# Base of math operations and derivatives
from jax import numpy as jnp
# Custom derivative declaration
from jax import grad, jacobian, custom_vjp
# Random generator
from jax import random
# Jax functions for jit
from functools import partial
# Vectorization
from jax import vmap

import numpy as np

# Implicit derivative
# from jaxopt import implicit_diff

KEY = random.PRNGKey(0)

# Distance function
from geomjax.manifolds.utils import pairwise_distance
# Optimiser
from geomjax.optim import GradientDescent
# Plotting
import matplotlib.pyplot as plt


# @implicit_diff.custom_root(grad(pairwise_distance))
def gradient_descend_weighted_mean(X_set, weights, optimiser, plot_loss_flag=False):
    """
    Weighted mean calculation as an optimization problem:
    find point, which minimises pairwise distances in the given set of points
    """

    # if no weights are provided, they are the same
    if weights == None:
        weights = jnp.ones(X_set.shape[0]) / X_set.shape[0]
        

    # array to plot loss
    if plot_loss_flag:
        plot_loss = []

    # init mean with random element from set and move it a bit
    Y = X_set[np.random.randint(0, X_set.shape[0], (1,))][0] + 1e-7

    for i in range(optimiser.maxiter):

        # calculate loss
        loss = pairwise_distance(Y, X_set, optimiser.manifold.distance, weights)

        euclid_grad = grad(pairwise_distance, argnums=0)(Y, X_set, optimiser.manifold.distance, weights)

        riem_grad = optimiser.manifold.project(Y, euclid_grad)

        Y = optimiser.step(Y, euclid_grad)

        if plot_loss_flag:
            # collect loss for plotting
            plot_loss.append(loss)

    if plot_loss_flag:
        print(f"Total loss: {pairwise_distance(Y, X_set, optimiser.manifold.distance, weights)}")
        fig, ax = plt.subplots()
        ax.plot(plot_loss)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        plt.show()

    return Y



def weighted_mean_implicit_derivative(x, X, w, manifold):
    """
    Computation of the derivative for the weighted mean
    """

    dy = grad(pairwise_distance, argnums=0)
    dy_projected = lambda x, X, w: manifold.project(x, dy(x, X, manifold.distance, w))  # новая строчка (проекция)

    d2yy = jnp.squeeze(jacobian(dy_projected, argnums=0)(x, X, w))
    d2yy_inv = jnp.linalg.inv(d2yy)

    d2xy = jnp.squeeze(jacobian(dy_projected, argnums=1)(x, X, w))  # (2, 22, 2)
    d2wy = jnp.squeeze(jacobian(dy_projected, argnums=2)(x, X, w))  # (2, 22)

    def grad_multiply_inverse(dyy_inv, d_mixed):
      return -dyy_inv @ d_mixed

    grad_multiply_inverse_batch = vmap(grad_multiply_inverse, (None, 1), 1)

    dfdx = grad_multiply_inverse_batch(d2yy_inv, d2xy)
    dfdw = grad_multiply_inverse_batch(d2yy_inv, d2wy)

    return dfdx, dfdw
    # for i in range(X.shape[0]):
    #     # broadcast ainsum опробовать
    #     d2xy = d2xy.at[:, i].set(-d2yy_inv @ d2xy[:, i])  # (2, 2) @ (2, 2)
    #     # приходится добавить x
    #     d2wy = d2wy.at[:, i].set(-d2yy_inv @ d2wy[:, i])

    # return d2xy, d2wy


@partial(custom_vjp, nondiff_argnums=(2,3))
def weighted_mean(X, w, optimiser, plot_loss_flag=False):
    return gradient_descend_weighted_mean(X_set=X, 
                                          weights=w, 
                                          optimiser=optimiser, 
                                          plot_loss_flag=plot_loss_flag)


def weighted_mean_fwd(X, w, optimiser, plot_loss_flag=False):
    # Returns primal output and residuals to be used in backward pass by f_bwd.
    x = weighted_mean(X, w, optimiser, plot_loss_flag)
    return x, (x, X, w, optimiser.manifold)


def weighted_mean_bwd(optimiser, plot_loss_flag, res, g):
    x, X, w, manifold = res  # Gets residuals computed in f_fwd
    grad_X, grad_w = weighted_mean_implicit_derivative(x, X, w, manifold)

    out_X = grad_X.T @ g
    out_w = grad_w.T @ g

    return (out_X.T, out_w)


weighted_mean.defvjp(weighted_mean_fwd, weighted_mean_bwd)
