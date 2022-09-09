Documentation and implementations of the various forms of SVD's derivative in the context of AD.

SVD has a number of variants. Flags - "thin"/"partial" vs "full" and compute only singular values vs the whole factorization. Input types - real vs complex.
These variants all change the derivative. Impacting both its value and its numerical stability. There are different resources documenting the different 
derivative variants. The implementations in well known AD codebases (pytorch, tensorflow, and jax) all have slightly different implementations of the derivative.

This python notebook is an attempt to consolidate documentation of the different cases as well as provide example implementations.

References:
- [Real valued partial SVD](https://j-towns.github.io/papers/svd-derivative.pdf)
- [Real valued full SVD](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf)
- [Complex partial SVD -- Backward pass only](https://arxiv.org/pdf/1909.02659.pdf)
- [Other real differentiability](https://arxiv.org/pdf/1509.07838.pdf)
- [Other complex differentiability](https://giggleliu.github.io/2019/04/02/einsumbp.html)
- [Jax](https://github.com/google/jax/blob/2a00533e3e686c1c9d7dfe9ed2a3b19217cfe76f/jax/_src/lax/linalg.py#L1578)
- [Pytorch forward](https://github.com/pytorch/pytorch/blob/7a8152530d490b30a56bb090e9a67397d20e16b1/torch/csrc/autograd/FunctionsManual.cpp#L3122)
- [Pytorch backward](https://github.com/pytorch/pytorch/blob/7a8152530d490b30a56bb090e9a67397d20e16b1/torch/csrc/autograd/FunctionsManual.cpp#L3228)
- [Tensorflow backward](https://github.com/tensorflow/tensorflow/blob/bbe41abdcb2f7e923489bfa21cfb546b6022f330/tensorflow/python/ops/linalg_grad.py#L815)
