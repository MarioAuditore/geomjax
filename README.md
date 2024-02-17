# geomjax
This repo contains a library for Geometric Deep Learning based on JAX. It emerged as a result of my research in the field of Deep Learning on Riemannian manifolds. 

It implements SPDNet from [1], riemannian gradient descent and manifolds:
- Hyperspher
- Stiefel manifold (Orthogonal matrices)
- Manifold of Symmetric Positive Definite (SPD) Matrices

What is more important, it contains the implementation of differentiable Frechet mean, originally described in [2] and extended to manifolds in [3]. The idea wast to use the average to extend the capabilities of SPDNet architecture and to implement novel algorithms.

# References

1. A Riemannian Network for SPD Matrix Learning, Zhiwu Huang, 2016, [arxiv](https://arxiv.org/abs/1608.04233)
2. On Differentiating Parameterized Argmin and Argmax Problems with Application to Bi-level Optimization, Stephen Gould, 2016, [arxiv](https://arxiv.org/abs/1607.05447)
3. Differentiating through the Fréchet Mean, Aaron Lou, 2020, [arxiv](https://arxiv.org/abs/2003.00335)
