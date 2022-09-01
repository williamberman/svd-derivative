```python
import os

os.environ["CUDA_VISIBLE_DEVICES"]=""

import jax.numpy as jnp
from jax import lax
import jax
from jax import random
import torch
import torch.autograd.functional as TF
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

torch_jvp = TF.jvp
torch_vjp = TF.vjp

jax.config.update('jax_platform_name', 'cpu')
jax.devices()
```

    2022-09-01 16:20:33.734899: E tensorflow/stream_executor/cuda/cuda_driver.cc:271] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
    2022-09-01 16:20:33.751634: E external/org_tensorflow/tensorflow/stream_executor/cuda/cuda_driver.cc:265] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected





    [CpuDevice(id=0)]



## Cases
$$
\newcommand{\tr}{{\mathrm{tr}}}
\newcommand{\d}{{\mathrm{d}}}
\newcommand{\A}{{\mathbf{A}}}
\newcommand{\U}{{\mathbf{U}}}
\renewcommand{\S}{{\mathbf{S}}}
\newcommand{\V}{{\mathbf{V}}}
\newcommand{\F}{{\mathbf{F}}}
\newcommand{\I}{{\mathbf{I}}}
\renewcommand{\P}{{\mathbf{P}}}
\newcommand{\D}{{\mathbf{D}}}
\newcommand{\dA}{{\d\A}}
\newcommand{\dU}{{\d\U}}
\newcommand{\dS}{{\d\S}}
\newcommand{\dV}{{\d\V}}
\newcommand{\dP}{{\d\P}}
\newcommand{\dD}{{\d\D}}
\newcommand{\Ut}{{\U^{\top}}}
\newcommand{\Vt}{{\V^{\top}}}
\newcommand{\Vh}{{\V^{H}}}
\newcommand{\dAt}{{\dA^{\top}}}
\newcommand{\dVt}{{\dV^{\top}}}
\newcommand{\gA}{{\overline{\A}}}
\newcommand{\gAt}{{\gA^{\top}}}
\newcommand{\gU}{{\overline{\U}}}
\newcommand{\gUt}{{\gU^{\top}}}
\newcommand{\gS}{{\overline{\S}}}
\newcommand{\gSt}{{\gS^{\top}}}
\newcommand{\gV}{{\overline{\V}}}
\newcommand{\gVt}{{\gV^{\top}}}
\newcommand{\K}{{\mathbf{K}}}
\newcommand{\Ku}{{\K_u}}
\newcommand{\Kut}{{\Ku^{\top}}}
\newcommand{\Vm}{{\V_m}}
\newcommand{\Vmt}{{\Vm^{\top}}}
\newcommand{\Kv}{{\K_v}}
\newcommand{\Kvt}{{\Kv^{\top}}}
\newcommand{\Vr}{{\V_r}}
\newcommand{\Vrt}{{\Vr^{\top}}}
\newcommand{\Ft}{{\F^{\top}}}
\newcommand{\gVm}{{\gV_m}}
\newcommand{\gVmt}{{\gVm^{\top}}}
\newcommand{\gVr}{{\gV_r}}
$$
The derivative of the SVD operation is determined by

1. Computing the full SVD vs the "thin"/"partial" SVD
2. complex vs real inputs
3. Computing the complete factorization $\U\S\Vh$ vs computing only the singular values $\S$

Computing only the singular values does not change the differential or gradient formulas and can be considered a side case of the full factorization.

## Real valued partial SVD

Reference: https://j-towns.github.io/papers/svd-derivative.pdf

### Forward mode

The differential formulas $\dU$, $\dS$, and $\dV$ in terms of $\dA$, $\U$, $\S$, and $\V$ are found from the chain rule.

##### Chain rule

$\dA = \dU \S \Vt + \U \dS \Vt + \U \S \dVt$

TODO(will) - Is this really the chain rule? Deriving this is a little funky because you end up taking partial derivatives with respect to matrices which I'm not sure how to define. The rule for expressing the differential and the product is in 2.1/2.2 https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf

##### Differential formulas

$\dU = \U ( \F \circ [\Ut \dA \V \S + \S \Vt \dAt \U] ) + (\I_m - \U \Ut ) \dA \V \S^{-1}$

$\dS = \I_k \circ [\Ut \dA \V]$

$\dV = \V (\F \circ [\S \Ut \dA \V + \Vt \dAt \U \S]) + (\I_n - \V \Vt) \dAt \U \S^{-1}$

where

$F_{ij} = \frac{1}{s_j^2 - s_i^2}, i \neq j$

$F_{ij} = 0$ otherwise

$k$ is the rank of $\A$

$\U$ is $m \times k$

$\S$ is $k \times k$

$\Vt$ is $k \times n$

##### Numerical instability
Both $\S^{-1}$ and $\F$ cause numerical instability in $\dU$ and $\dV$.  

$\S^{-1}$ causes numerical instability with zero or very small singular values. The inverse of a diagonal matrix is the element wise reciprocals of the inverting matrix's diagonal. Zero singular values will cause $\S^{-1}$ to have infinite elements. Very small singular values will cause $\S^{-1}$ to have very large floating point elements.

$\F$ divides by the difference of all pairs of singular values. Repeated singular values will cause infinite elements of $\F$. Close singular values will cause very large floating point elements of $\F$.  

##### Computing only singular values
Only compute the differential for $\S$. $\dU$ and $\dV$ are ignored. Because $\dU$ and $\dV$ are not computed, $\F$ and $\S^{-1}$ are not needed, so there is no numerical instability.


```python
def svd_jvp_real_valued_partial(A, dA):    
    U, S_vals, Vt = jnp.linalg.svd(A, compute_uv=True, full_matrices=False)

    S = jnp.diag(S_vals)
    Ut = U.T
    V = Vt.T
    dAt = dA.T

    k = S.shape[0]
    m = A.shape[0]
    n = A.shape[1]

    I_k = jnp.eye(k)
    I_m = jnp.eye(m)
    I_n = jnp.eye(n)

    S_inv = jnp.linalg.inv(S)

    F_ = F(S_vals, k)

    dU = U @ (F_ * ((Ut @ dA @ V @ S) + (S @ Vt @ dAt @ U))) + ((I_m - U @ Ut) @ dA @ V @ S_inv)
    
    # Note that the `I_k *` is extraneous. It zeros out the rest of the matrix besides the diagonal.
    # We only return `dS_vals` which takes only the diagonal of `dS` anyway. 
    dS = I_k * (Ut @ dA @ V)
    dS_vals = jnp.diagonal(dS)
    
    dV = V @ (F_ * (S @ Ut @ dA @ V + Vt @ dAt @ U @ S)) + (I_n - V @ Vt) @ dAt @ U @ S_inv
    
    return (U, S_vals, Vt), (dU, dS_vals, dV.T)

def F(S_vals, k):
    F_i_j = lambda i, j: lax.cond(i == j, lambda: 0., lambda: 1 / (S_vals[j]**2 - S_vals[i]**2))
    F_fun = jax.vmap(jax.vmap(F_i_j, (None, 0)), (0, None))

    indices = jnp.arange(k)
    F_ = F_fun(indices, indices)
    
    return F_
```


```python
def assert_svd_uv(a, b):
    (U, S, Vt), (dU, dS, dVt) = a
    (U_, S_, Vt_), (dU_, dS_, dVt_) = b
    
    assert_allclose(U, U_)
    assert_allclose(S, S_)
    assert_allclose(Vt, Vt_)
    assert_allclose(dU, dU_)
    assert_allclose(dS, dS_)
    assert_allclose(dVt, dVt_)

def check_allclose(l, r):
    return jnp.allclose(jnp.array(l), jnp.array(r), atol=1e-04)

def assert_allclose(l, r):
    assert(check_allclose(l, r))
        
def jax_real_valued_partial(A, dA):
    A = jnp.array(A)
    dA = jnp.array(dA)
    
    return jax.jvp(
        lambda A: jnp.linalg.svd(A, compute_uv=True, full_matrices=False), 
        (A,), 
        (dA,)
    )

def pytorch_real_valued_partial(A, dA):
    A = torch.tensor(A)
    dA = torch.tensor(dA)
    
    return torch_jvp(lambda A: torch.linalg.svd(A, full_matrices=False), A, dA)
    
def check_real_valued_partial(A, dA):
    jax_res = jax_real_valued_partial(A, dA)
    torch_res = pytorch_real_valued_partial(A, dA)
    res = svd_jvp_real_valued_partial(jnp.array(A), jnp.array(dA))
    
    # example == jax == torch
    assert_svd_uv(jax_res, torch_res)
    assert_svd_uv(jax_res, res)
    
    return jax_res, torch_res, res

def check_real_valued_partial_args(key, m, n):
    A = random.normal(key, (m, n))
    dA = jnp.ones_like(A)
            
    A = A.tolist()
    dA = dA.tolist()
    
    return A, dA
  
def check_real_valued_partials():
    key = random.PRNGKey(0)
    
    for m in range(3, 11):
        for n in range(3, 11):
            key, subkey = random.split(key)
            args = check_real_valued_partial_args(subkey, m, n)
            check_real_valued_partial(*args)

check_real_valued_partials()
```

### Reverse mode

##### Chain rule
$$
\newcommand{\termu}{{\mathrm{term}_U}}
\newcommand{\terms}{{\mathrm{term}_S}}
\newcommand{\termv}{{\mathrm{term}_V}}
$$
$ \tr(\gAt \dA) = \tr(\gUt \dU) + \tr(\gSt \dS) + \tr(\gVt \dV) $

##### Gradient formula

The formula for A's gradient has terms $\termu$, $\terms$, and $\termv$ found from the respective trace terms in the chain rule

$\gA = \termu + \terms + \termv$

$\termu = [\U (\F \circ [\Ut \gU - \gUt \U]) \S + (\I_m - \U \Ut) \gU \S^{-1} ] \Vt$

$\terms = \U (\I_k \circ \gS ) \Vt$

$\termv = \U [\S (\F \circ [\Vt \gV - \gVt \V]) \Vt + \S^{-1} \gVt (\I_n - \V \Vt)]$

##### Numerical instability
$\S^{-1}$ and $\F$ cause numerical instability in $\gA$ through $\termu$ and $\termv$. The explanation is the same as in the forward mode.

##### Computing only singular values
When only computing singular values, no $\U$ or $\Vt$ were produced to impact a resulting objective function. $\gU$ and $\gVt$ must then be zero, and so are $\termu$ and $\termv$. This means that $\gA = \terms$, and there is no numerical instability.


```python
def svd_vjp_real_valued_partial(A, U, S_vals, Vt, gU, gS_vals, gVt):
    S = jnp.diag(S_vals)
    gS = jnp.diag(gS_vals)
    
    k = S.shape[0]
    m = A.shape[0]
    n = A.shape[1]
    
    I_m = jnp.eye(m)
    I_k = jnp.eye(k)
    I_n = jnp.eye(n)
    
    V = Vt.T
    Ut = U.T
    gUt = gU.T
    gV = gVt.T
    
    S_inv = jnp.linalg.inv(S)
    
    F_ = F(S_vals, k)
    
    term_U = (U @ (F_ * (Ut @ gU - gUt @ U)) @ S + (I_m - U @ Ut) @ gU @ S_inv) @ Vt
    
    term_S = U @ (I_k * gS) @ Vt
    
    term_V = U @ (S @ (F_ * (Vt @ gV - gVt @ V)) @ Vt + S_inv @ gVt @ (I_n - V @ Vt))
    
    gA = term_U + term_S + term_V
    
    return gA
```


```python
def svd_vjp_real_valued_partial_(A, gU, gS, gVt):
    A = jnp.array(A)
    gU = jnp.array(gU)
    gS = jnp.array(gS)
    gVt = jnp.array(gVt)
    
    U, S, Vt = jnp.linalg.svd(A, compute_uv=True, full_matrices=False)
    
    gA = svd_vjp_real_valued_partial(A, U, S, Vt, gU, gS, gVt)
    
    return gA

def jax_real_valued_partial_vjp(A, gU, gS, gVt):
    A = jnp.array(A)
    gU = jnp.array(gU)
    gS = jnp.array(gS)
    gVt = jnp.array(gVt)    
    
    _, vjp_fun = jax.vjp(
        lambda A: jnp.linalg.svd(A, compute_uv=True, full_matrices=False), 
        A,
    )
    
    gA, = vjp_fun((gU, gS, gVt))
    
    return gA

def pytorch_real_valued_partial_vjp(A, gU, gS, gVt):
    A = torch.tensor(A)
    gU = torch.tensor(gU)
    gS = torch.tensor(gS)
    gVt = torch.tensor(gVt)
    
    _, gA = torch_vjp(lambda A: torch.linalg.svd(A, full_matrices=False), A, (gU, gS, gVt))
    
    return gA
    
def check_real_valued_partial_vjp(A, gU, gS, gVt):
    jax_gA = jax_real_valued_partial_vjp(A, gU, gS, gVt)
    torch_gA = pytorch_real_valued_partial_vjp(A, gU, gS, gVt)
    res_gA = svd_vjp_real_valued_partial_(A, gU, gS, gVt)
    
    # example == jax == torch
    assert_allclose(jax_gA, torch_gA)
    assert_allclose(jax_gA, res_gA)
    
    return jax_gA, torch_gA, res_gA
    
def check_real_valued_partial_vjp_args(key, m, n):
    A = random.normal(key, (m, n))

    k = min(m, n)

    gU = jnp.ones((m, k))
    gS = jnp.ones(k)
    gVt = jnp.ones((k, n))

    A = A.tolist()
    gU = gU.tolist()
    gS = gS.tolist()
    gVt = gVt.tolist()
    
    return A, gU, gS, gVt

def check_real_valued_partials_vjp():
    key = random.PRNGKey(0)
    
    for m in range(3, 11):
        for n in range(3, 11):
            key, subkey = random.split(key)
            args = check_real_valued_partial_vjp_args(subkey, m, n)
            check_real_valued_partial_vjp(*args)

            
check_real_valued_partials_vjp()
```

## Real valued full SVD

Reference: https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf

### Forward mode

The differential formulas are found by a similar process to the partial case.

##### Differential formulas

$ \dU = \U [\F \circ ( \dP_1 \S_1 + \S_1 \dP_1^{\top} ) ] $

$ \Ut \dA \V = \begin{bmatrix} \dP_1 & \dP_2 \end{bmatrix} $

$ \S = \begin{bmatrix} \S_1 & 0 \end{bmatrix} $

$ \dP_1 $ is $m \times m$

$ \S_1 $ is $m \times m$

---

$ \dS = \I \circ [\U^{\top} \dA \V ] $

---

$ \dV = \V \dD $

$ \dD = \begin{bmatrix} \dD_1 & -\dD_2 \\ \dD_2^{\top} & \dD_3 \end{bmatrix} $

$ \dD_1 = \F \circ (\S_1 \dP_1 + \dP_1^{\top} \S_1 ) $

$ \dD_2 = \S_1^{-1} \dP_2 $

$ \dD_3 = \mathbf{0} $

$ \dD $ is $ n \times n $

$ \dD_1 $ is $ m \times m $

$ \dD_2 $ is $ m \times (n - m) $

$ \dD_3 $ is $ (n - m) \times (n - m) $

##### Numerical instability
The full SVD has the same numerical instability as the partial SVD.

Both $\S_1^{-1}$ and $\F$ cause numerical instability in $\dU$ and $\dV$.

Additionally, we know ahead of time $\F$ will contain divisions by zero when $|n - m| \geq 2$ because this guarantees at least two zero valued singular values. 

##### Computing only singular values
Same as in the partial SVD. Only computing the singular values ignores $\dU$ and $\dV$ which means there is no numerical instability.


```python
def svd_jvp_real_valued_full(A, dA):
    m = A.shape[0]
    n = A.shape[1]
    
    if m > n:
        (U, S, Vt), (dU, dS, dVt) = svd_jvp_real_valued_full(A.T, dA.T)
        return (Vt.T, S, U.T), (dVt.T, dS, dU.T)
    
    # m <= n
    
    U, S_vals, Vt = jnp.linalg.svd(A, compute_uv=True, full_matrices=True)
        
    Ut = U.T
    V = Vt.T
    
    I = fill_diagonal(jnp.zeros((m, n)), 1)

    # Can directly create S_1 without creating S which has added zero columns
    # There will be m singular values, so S_1 will be m x m.
    S_1 = jnp.diag(S_vals)
    S_1_inv = jnp.linalg.inv(S_1)
    
    # m is number singular values insted of k as in partial version
    F_ = F(S_vals, m)
    
    ########## dU ###########

    dP = Ut @ dA @ V
    dP_1 = dP[:, :m]
    dP_2 = dP[:, m:]
    dP_1t = dP_1.T
    
    dU = U @ (F_ * (dP_1 @ S_1 + S_1 @ dP_1t))
    
    ########################
    
    ######### dS ##########
    
    dS = I * (Ut @ dA @ V)
    dS_vals = jnp.diagonal(dS)
    
    #######################
    
    ######## dV ##########
    
    dD_1 = F_ * (S_1 @ dP_1 + dP_1t @ S_1)
    dD_2 = S_1_inv @ dP_2
    dD_3 = jnp.zeros((n-m, n-m))
    neg_dD_2 = -dD_2
    dD_2t = dD_2.T
    
    dD_top = jnp.concatenate((dD_1, neg_dD_2), axis=1)
    dD_bottom = jnp.concatenate((dD_2t, dD_3), axis=1)
    
    dD = jnp.concatenate((dD_top, dD_bottom), axis=0)
    
    dV = V @ dD
    
    #####################
    
    return (U, S_vals, Vt), (dU, dS_vals, dV.T)

# modified version of fill_diagonal because jax does not implement it.
# https://github.com/numpy/numpy/blob/v1.23.0/numpy/lib/index_tricks.py#L779-L910
def fill_diagonal(a, val, start=None):
    m, n = a.shape
    
    end = None
    step = a.shape[1] + 1
    
    if start:
        start_row, start_col = start
        start = start_row * n + start_col
        a = a.flatten().at[start:end:step].set(val)
    else:
        a = a.flatten().at[:end:step].set(val)
        
    a = a.reshape((m, n))
    
    return a
```


```python
def svd_jvp_real_valued_full_(A, dA):
    A = jnp.array(A)
    dA = jnp.array(dA)
    
    return svd_jvp_real_valued_full(A, dA)
    
def pytorch_real_valued_full(A, dA):
    A = torch.tensor(A)
    dA = torch.tensor(dA)
    
    return torch_jvp(lambda A: torch.linalg.svd(A, full_matrices=True), A, dA)

def check_real_valued_full(A, dA):
    # NOTE - JAX does not implement full
    torch_res = pytorch_real_valued_partial(A, dA)
    res = svd_jvp_real_valued_partial(jnp.array(A), jnp.array(dA))
    
    assert_svd_uv(torch_res, res)
    
    return torch_res, res

def check_real_valued_full_args(key, m, n):
    A = random.normal(key, (m, n))
    dA = jnp.ones_like(A)
            
    A = A.tolist()
    dA = dA.tolist()
    
    return A, dA
  
def check_real_valued_fulls():
    key = random.PRNGKey(0)
    
    for m in range(3, 11):
        for n in range(3, 11):
            key, subkey = random.split(key)
            args = check_real_valued_full_args(subkey, m, n)
            check_real_valued_full(*args)

check_real_valued_fulls()
```

### Backward mode

The backward mode for the full case is found by the same process as in the partial case, but the complete formula is not documented in the reference. 

##### Gradient formula

$\gA = \termu + \terms + \termv$

$\termu = \U [\Ku + \Kut] \S_1 \Vmt $

$\terms = \U (\I \circ \gS) \Vt$

$\termv = \U \S_1 [\Kv + \Kvt] \Vmt + \U \S_1^{-1} \gVmt \Vr \Vrt - \U \S_1^{-1} \Vmt \gVr \Vrt $

$\Ku = \F \circ (\Ut\gU)$

$\Kv = \F \circ (\Vmt \gVm)$

$ \Vm $ is the left most $m$ columns of $\V $.

$ \gVm $ is the left most $m$ columns of $\gV $.

$ \Vr $ is the right most $n - m$ columns of $\V$.

$ \gVr $ is the right most $n - m$ columns of $\gV$.

##### Numerical instability
$\S^{-1}$ and $\F$ cause numerical instability in $\gA$ through $\termu$ and $\termv$. The explanation is the same as in the forward mode.

##### Computing only singular values
When only computing singular values, no $\U$ or $\Vt$ were produced to impact a resulting objective function. $\gU$ and $\gVt$ must then be zero, and so are $\termu$ and $\termv$. This means that $\gA = \terms$, and there is no numerical instability.


```python
def svd_vjp_real_valued_full(A, U, S_vals, Vt, gU, gS_vals, gVt):
    m = A.shape[0]
    n = A.shape[1]
    
    if m > n:
        gA = svd_vjp_real_valued_full(A.T, Vt.T, S_vals, U.T, gVt.T, gS_vals, gU.T)
        return gA.T
    
    # m <= n
    
    S = fill_diagonal(jnp.zeros((m, n)), S_vals)
    gS = fill_diagonal(jnp.zeros((m, n)), gS_vals)
    S_1 = fill_diagonal(jnp.zeros((m, m)), S_vals)
    S_1t = S_1.T
    S_1_inv = jnp.linalg.inv(S_1)
    
    F_ = F(S_vals, m)
    Ft = F_.T
    
    V = Vt.T
    Ut = U.T
    gSt = gS.T
    gV = gVt.T
    
    Vm = V[:,:m]
    Vr = V[:,m:]
    Vmt = Vm.T
    Vrt = Vr.T
    
    gVm = gV[:,:m]
    gVr = gV[:,m:]
    gVmt = gVm.T
    
    
    ############ term_U ##########
    
    K_u = F_ * (Ut @ gU)
    K_ut = K_u.T
    term_U = U @ (K_u + K_ut) @ S_1 @ Vmt
    
    #############################
    
    ########## term_S ###########
    
    I = fill_diagonal(jnp.zeros((m, n)), 1)
    term_S = U @ (I * gS) @ Vt
    
    #############################
    
    ######### term_V ############

    K_v = F_ * (Vmt @ gVm)
    K_vt = K_v.T
    
    term_V1 = U @ S_1t @ (K_v + K_vt) @ Vmt
    term_V2 = U @ S_1_inv @ gVmt @ Vr @ Vrt
    term_V3 = -U @ S_1_inv @ Vmt @ gVr @ Vrt
    
    term_V = term_V1 + term_V2 + term_V3
    
    #############################
    
    gA = term_U + term_S + term_V
    
    return gA
```


```python
def svd_vjp_real_valued_full_(A, gU, gS, gVt):
    A = jnp.array(A)
    gU = jnp.array(gU)
    gS = jnp.array(gS)
    gVt = jnp.array(gVt)
    
    U, S, Vt = jnp.linalg.svd(A, compute_uv=True, full_matrices=True)
    
    gA = svd_vjp_real_valued_full(A, U, S, Vt, gU, gS, gVt)
    
    return gA
    
def tf_real_valued_full_vjp(A, gU, gS, gVt):
    A = tf.constant(A)
    gU = tf.constant(gU)
    gS = tf.constant(gS)
    gVt = tf.constant(gVt)
    
    with tf.GradientTape() as g:
        g.watch(A)
        S, U, V = tf.linalg.svd(A, full_matrices=True, compute_uv=True)
        Vt = tf.transpose(V)
        
    gA = g.gradient((S, U, Vt), A, (gS, gU, gVt))
        
    return gA

def check_real_valued_full_vjp(A, gU, gS, gVt):
    tf_gA = tf_real_valued_full_vjp(A, gU, gS, gVt)
    res_gA = svd_vjp_real_valued_full_(A, gU, gS, gVt)
    
    assert_allclose(tf_gA, res_gA)
    
    return tf_gA, res_gA

def check_real_valued_full_vjp_args(key, m, n):
    while True:
        key, subkey = random.split(key)
        A = random.normal(key, (m, n))
        
        tf_S, tf_U, tf_V = tf.linalg.svd(tf.constant(A.tolist()), full_matrices=True, compute_uv=True)
        tf_Vt = tf.transpose(tf_V)
        
        jax_U, jax_S, jax_Vt = jnp.linalg.svd(A, full_matrices=True, compute_uv=True)
        
        if check_allclose(jax_U, tf_U) and check_allclose(jax_S, tf_S) and check_allclose(jax_Vt, tf_Vt):
            break

    k = min(m, n)

    gU = jnp.zeros((m, m))
    gS = jnp.zeros(k)
    gVt = jnp.ones((n, n))

    A = A.tolist()
    gU = gU.tolist()
    gS = gS.tolist()
    gVt = gVt.tolist()

    return key, (A, gU, gS, gVt)
  
def check_real_valued_fulls_vjp():
    key = random.PRNGKey(0)
    
    for m in range(2, 11):
        # TF gradient will throw error if abs(n - m) >= 2, so constrain range of dimensions
        min_n = max(m - 1, 2)
        max_n = min(m + 2, 11)
        for n in range(min_n, max_n):
            assert(abs(m - n) < 2)
            key, args = check_real_valued_full_vjp_args(key, m, n)
            check_real_valued_full_vjp(*args)

# NOTE(will) - TF calculates a different gradient for full SVD than Pytorch. 
# Algebraically, I can confirm the gradient used by TF is what is specified in
# the reference by the differential formulas. I'm only checking against TF's
# gradient and will investigate why TF and pytorch calculate different gradients
# in the future.
check_real_valued_fulls_vjp()
```


```python
# TODO complex values
```

Sources:

SVD real differentiability:
- https://j-towns.github.io/papers/svd-derivative.pdf
- https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
- https://arxiv.org/pdf/1509.07838.pdf

SVD complex differentiability:
- https://arxiv.org/pdf/1909.02659.pdf
- https://giggleliu.github.io/2019/04/02/einsumbp.html

Existing implementations:

Jax forward:
- https://github.com/google/jax/blob/2a00533e3e686c1c9d7dfe9ed2a3b19217cfe76f/jax/_src/lax/linalg.py#L1578
- Jax only implements the forward rule because jax can derive the backward rule from the forward rule and vice versa.

Pytorch forward:
- https://github.com/pytorch/pytorch/blob/7a8152530d490b30a56bb090e9a67397d20e16b1/torch/csrc/autograd/FunctionsManual.cpp#L3122

Pytorch backward:
- https://github.com/pytorch/pytorch/blob/7a8152530d490b30a56bb090e9a67397d20e16b1/torch/csrc/autograd/FunctionsManual.cpp#L3228

Tensorflow backward:
- https://github.com/tensorflow/tensorflow/blob/bbe41abdcb2f7e923489bfa21cfb546b6022f330/tensorflow/python/ops/linalg_grad.py#L815

General complex differentiability:
- https://mediatum.ub.tum.de/doc/631019/631019.pdf
