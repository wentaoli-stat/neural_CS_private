# Fisher-score approximation using subscores

> Transcription note: Math is delimited with `$...$` and `$$...$$` for VSCode Markdown rendering. I kept the mathematical notation as close as possible to the handwritten notes. I kept the mathematical notation as close as possible to the handwritten notes. A few symbols are ambiguous in the scan; I marked these with `[unclear]` where necessary.

## Motivation: approximate the full Fisher score using marginal and pairwise composite scores


Consider $Y \sim f(Y;\theta_0)$ where $\theta_0\in\mathbb{R}^p$ is the data-generating parameter. We try to approximate the Fisher score $\nabla_{\theta}\log f(Y;\theta_0)$. 

Existing architectures includes

$$
S_\psi(\theta,\{X_i\}_{i=1}^n),
$$
(This includes $S_\psi(\theta,\{X_i\}_{i=1}^n,\{(X_i,X_j)\}_{i,j=1}^n)$)
Khoo et al 2025 and Jiang et al 2026. 

### New Architecture:  Nonlinear aggregation of marginal and pairwise subscores

Let  $cs_1(Y_i;\theta) = \nabla_{\theta}\log f(Y_i;\theta)$ and $cs_2(Y_i,Y_j;\theta)=\nabla_{\theta}\log f(Y_i,Y_j;\theta)$. The new method is to approximate $\nabla_{\theta}\log f(Y;\theta)$ using 

$$S_{\psi,\phi}(X,\theta)=S_\psi\{\theta,CS_1(X,\theta; \phi), CS_2(X,\theta; \phi)\}\in\mathbb{R}^p,$$

where $CS_1(Y,\theta; \phi) = \sum_{i=1}^n a_{\phi}(cs_1(Y_i;\theta) )$ and $CS_2(Y,\theta; \phi)=\sum_{i<j} b_{\phi}(cs_2(Y_i,Y_j;\theta))$, in an $L^2$ sense.


<!-- $$
S_\psi(\theta,\{a_{\phi}(X_i,\theta)\}_{i=1}^n)
$$ -->

<!-- and

$$
S_\psi(\theta,\{b_{\phi}(X_i,X_j,\theta)\}_{i,j=1}^n,\{a_{\phi}(X_i,\theta)\}_{i=1}^n).
$$ -->
<!-- 
$\sum_{i,j} (x_i * x_j)^{\theta}$

U1=$\{X_i\}$

U2=$\{(X_i,X_j)\}$  -->

The ideal loss would be

$$
\mathcal L
=
\mathbb E_{\theta,X}
\left[
\left\|
S_\psi\{\theta,c_\phi(X;\theta)\}
-
\nabla_{\theta}\log f(X\mid\theta)
\right\|^2
\right].
$$

But $\log f(X\mid\theta,\nu)$ is unknown.

#### Example:
As a motivating example, consider a multivariate $t$ model. The score can be written as

$$
\nabla_{\rho,\nu}\log f(Y;\rho,\nu)
=
\nabla\left\{C_1(\rho,\nu)\log\left(1+\frac{Q(Y;\rho)}{\nu}\right)+C_2(\rho,\nu)\right\}.
$$

Equivalently,

$$
\nabla_{\rho,\nu}\log f(Y_i;\rho,\nu)
=
\nabla\{C_1(\rho,\nu)R(Q_1(Y_i;\rho),\nu)+C_2(\rho,\nu)\},
$$

and for pairs

$$
\nabla_{\rho,\nu}\log f(Y_i,Y_j;\rho,\nu)
\propto
\nabla R(Q_2(Y_i,Y_j;\rho),\nu).
$$

Since

$$
Q(Y;\rho) =
\sum_{i,j} w_{ij}(\rho)
L\{Q_1(Y_i;\rho),Q_2(Y_i,Y_j;\rho)\},
$$

we have $\nabla_{\rho,\nu}\log f(Y;\rho,\nu)$ exactly a nonlinear transformation of $\nabla_{\rho,\nu}\log f(Y_i;\rho,\nu)$ and $\nabla_{\rho,\nu}\log f(Y_i,Y_j;\rho,\nu)$.


### Can not do sampling and data compression simultaneously in CSDM

At a fixed $(\theta_0,\nu_0)$, one would like to use

$$
\mathcal L
=
\mathbb E_{\theta_0,\nu_0}
\left[
\left\|
S_\psi\{\theta_0,\nu_0,c_\phi(X;\theta_0,\nu_0)\}
-
\nabla_{\theta,\nu}\log f(X\mid\theta,\nu)
\right\|^2
\right].
$$

However, using CSDM to minimise this is equivalent to minimising

$$
\mathcal L
=
\mathbb E_{\theta_0,\nu_0}
\left[
\left\|
S_\psi\{\theta_0,\nu_0,c_\phi(X;\theta_0,\nu_0)\}
-
\nabla_{\theta,\nu}\log f(c_\phi(X;\theta_0,\nu_0)\mid\theta,\nu)
\right\|^2
\right].
$$

This is not what we want.

### Raw score-variance objective (Fisher information) is problematic

One might want to optimize the Fisher information

$$
\mathbb E_{\theta_0}
\left\{
S_{\psi,\phi}(X,\theta_0)
S_{\psi,\phi}(X,\theta_0)^T
\right\}.
$$

But unless $S_{\psi,\phi}(X,\theta)$ is a proper statistical score satisfying constraints as in Jiang et al. (2026), for example, using Second Bartlett identity, 

$$
\mathbb E
\left\{
S_{\psi,\phi}(X,\theta)S_{\psi,\phi}(X,\theta)^T
\right\}
=
\mathbb E
\left\{
\nabla_\theta S_{\psi,\phi}(X,\theta)
\right\},
$$

we cannot use this objective. The issue is scaling: the above metric of $C \cdot S_{\psi,\phi}(X,\theta)$ can be made arbitrarily large by increasing $C$.

---

## Godambe information objective

One option is Godambe information:

$$
G(\theta_0)
=
\mathbb E_{\theta_0}
\{\nabla_\theta S_{\psi,\phi}(X,\theta_0)\}^{T}
\operatorname{Var}_{\theta_0}
\{S_{\psi,\phi}(X,\theta_0)\}^{-1}
\mathbb E_{\theta_0}
\{\nabla_\theta S_{\psi,\phi}(X,\theta_0)\}.
$$

This is scale-invariant. So we  maximise $\log\det\{G(\theta_0)+\epsilon I\}.$

If $S_{\psi,\phi}(X,\theta)$ is not unbiased, i.e. $\mathbb E_\theta\{S_{\psi,\phi}(X,\theta)\}\neq 0$, it has an implied class of unbiased estimating equations of the form

$$
S_{\psi,\phi}(X,\beta)
-
\mathbb E_\theta\{S_{\psi,\phi}(X,\beta)\},
\qquad
\beta\in\Theta.
$$

Since

$$
\nabla_\theta
\left[
S_{\psi,\phi}(X,\beta)
-
\mathbb E_\theta\{S_{\psi,\phi}(X,\beta)\}
\right]
=
-
\nabla_\theta
\mathbb E_\theta\{S_{\psi,\phi}(X,\beta)\},
$$

the Godambe information of the unbiased estimating function indexed by $\beta$ is as follows.
$$
G^*(\theta,\beta)
=
\nabla_\theta
\mathbb E_\theta\{S_{\psi,\phi}(X,\beta)\}
\operatorname{Var}_{\theta}
\{S_{\psi,\phi}(X,\beta)\}^{-1}
\nabla_\theta
\mathbb E_\theta\{S_{\psi,\phi}(X,\beta)\}^{T},
$$

though the intuition of $G^*(\theta,\beta)$ for different $\beta$ is not clear.

If $S_{\psi,\phi}(X,\theta)$ is unbiased, Li et al. (2026) show that the Godambe information can also be expressed as above.


So we don't need to debias $S_{\psi,\phi}(X,\theta)$, and the optimization objective is $G^*(\theta_0,\theta_0)$. In practice, $\theta_0$ is replaced by an unbiased estimate $\hat{\theta}$ using the observation.

For amortisation, we would like to have the mapping
$$
\{\psi(\theta),\phi(\theta)\}
=
\arg\max_{\psi,\phi} G^*(\theta,\theta).
$$
(How to train this?)

---
## Benchmarks

For X = {x_i},

M1: MLP with {x_i}

M2: Linear with {x_i} as in Khoo et al 2025

M3: Linearly aggregated batched composite score

M4: MLP with {x_i} under constraints as in Jiang et al 2026 (for amortization, can it learn features such as x^{\theta})

M5: new method
<!-- 
* $U_\phi(X; \theta) = h_{\psi}\left(\sum_i a_\phi\left(\theta, X_i\right) , \sum_{i<j} b_\phi\left(\theta, X_i, X_j\right)\right)$, i.e. using raw data and raw pairs instead of marginal scores and pairwise scores. The benefits include
  
  * (X_i, X_j) can be high-dimensional while $cs_2(X_i, X_j;\theta)$ is p-dimensional
  * $U_{\phi}$ is less robust in extreme values of $(X_i, X_j)$ than the new method
  * The variance matrix in Godambe info may be more stable for the new method than  $U_{\phi}$. 
   -->
Ablations

Try simulation budget from small to large

### Intuitive comparison
*  M1 vs M5: The Gaussian example and Box-Cox transformed Gaussian example show subscores contain features in the Fisher score, and M1 need to learn them from scratch. So M5 has lower inductive bias.
*  M4 vs M5: The multivariate t example shows the Fisher score is a nonlinear combination of these features and hence the subscores.

### Metrics
* Score approximation error, given the true score function, under same simulation budget and as simulation budget increases 
* Godambe information in the form of $R E_{\mathrm{det}}=\left[\frac{\operatorname{det} G\left(\theta_0\right)}{\operatorname{det} I\left(\theta_0\right)}\right]^{1 / d_\theta}$, relative to the Fisher information
* Visualize and compare the historgram/contour plots of 1D/2D Fisher score and approximate scores as random variables  
* With sampling results, compare RMSE of posterior mean, bias, posterior variance and credible interval coverage

### Conceptual examples

#### Gaussian example

#### Box-Cox transformed Gaussian (in Appendix)
See the box_cox_transformed_gaussian_subscore_example.md file 


---
## Numerical Examples

* Vector AR(1) with multivariate t noise where $X_i \in \mathbb{R}^{20}$; $cs_1(X_i,\theta)\in\mathbb{R}^2$ and $cs_2(X_i,X_j,\theta)\in\mathbb{R}^2$ provide low-dimensional features

---
## Further refinements
### Variance inversion in optimization
One issue is that

$$
\operatorname{Var}_{\theta_0}
\{S_{\psi,\phi}(X,\theta_0)\}^{-1}
$$
is delicate in optimisation. 

---
### Over-identification extension

Use $S_{\psi,\phi}(X,\theta)\in\mathbb{R}^q$ where $q>p$, i.e. an over-identified estimating function. Its Godambe information is still $p$-dimensional. 




