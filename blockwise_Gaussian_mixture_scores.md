# Full, Marginal, and Pairwise Scores for Two Blockwise Latent Mixture Models

This note collects the exact full Fisher score, marginal subscore, pairwise subscore, and the nonlinear recovery of the full score for two blockwise latent mixture models.

Throughout, the unknown parameter is $\pi$, while $\tau$ is known.  For each block $k=1,\dots,K$,

$$
Z_k\stackrel{iid}{\sim}\mathrm{Bernoulli}(\pi),
$$

and observations are

$$
Y_{\cdot k}=(Y_{1k},\dots,Y_{nk})^\top.
$$

Different blocks are independent. Consider a large value for $n$ and a fixed value for $K$.

---

# 1. Blockwise Shared Mean-Shift Mixture

## 1.1 Model

Conditional on $Z_k=0$,

$$
Y_{ik}\mid Z_k=0 \stackrel{iid}{\sim} N(0,1),
\qquad i=1,\dots,n.
$$

Conditional on $Z_k=1$,

$$
Y_{ik}\mid Z_k=1 \stackrel{iid}{\sim} N(\tau,1),
\qquad i=1,\dots,n.
$$

Define

$$
\phi_0(y)=\phi(y),
\qquad
\phi_\tau(y)=\phi(y-\tau),
$$

and

$$
r_{ik}
=
\frac{\phi_\tau(Y_{ik})}{\phi_0(Y_{ik})}
=
\exp\left(\tau Y_{ik}-\frac{\tau^2}{2}\right).
$$

The block likelihood ratio is

$$
R_k
=
\prod_{i=1}^n r_{ik}
=
\exp\left(
\tau\sum_{i=1}^n Y_{ik}
-
\frac{n\tau^2}{2}
\right).
$$

---

## 1.2 Exact full Fisher score for $\pi$

The block density is

$$
p_\pi(Y_{\cdot k})
=
(1-\pi)\prod_{i=1}^n\phi_0(Y_{ik})
+
\pi\prod_{i=1}^n\phi_\tau(Y_{ik}).
$$

Equivalently,

$$
p_\pi(Y_{\cdot k})
=
\prod_{i=1}^n\phi_0(Y_{ik})
\left(1-\pi+\pi R_k\right).
$$

The full joint density is

$$
p_\pi(Y)
=
\prod_{k=1}^K p_\pi(Y_{\cdot k}).
$$

Therefore

$$
\log p_\pi(Y)
=
\text{constant}
+
\sum_{k=1}^K
\log(1-\pi+\pi R_k).
$$

The exact Fisher score for $\pi$ is

$$
S_{\mathrm{full}}(Y;\pi)
=
\frac{\partial}{\partial\pi}\log p_\pi(Y)
=
\sum_{k=1}^K
\frac{R_k-1}{1-\pi+\pi R_k}.
$$

Equivalently, define the block posterior regime probability

$$
w_k(Y_{\cdot k})
=
\mathbb P_\pi(Z_k=1\mid Y_{\cdot k})
=
\frac{\pi R_k}{1-\pi+\pi R_k}.
$$

Then

$$
S_{\mathrm{full}}(Y;\pi)
=
\sum_{k=1}^K
\frac{w_k(Y_{\cdot k})-\pi}{\pi(1-\pi)}.
$$

The full score depends on the global within-block likelihood ratio $R_k=\prod_i r_{ik}$.

---

## 1.3 Marginal subscore

The one-dimensional marginal density of $Y_{ik}$ is

$$
p_\pi(Y_{ik})
=
(1-\pi)\phi_0(Y_{ik})
+
\pi\phi_\tau(Y_{ik})
=
\phi_0(Y_{ik})(1-\pi+\pi r_{ik}).
$$

The marginal subscore is

$$
s_{ik}^{(1)}
=
\frac{\partial}{\partial\pi}
\log p_\pi(Y_{ik}).
$$

Thus

$$
s_{ik}^{(1)}
=
\frac{\phi_\tau(Y_{ik})-\phi_0(Y_{ik})}
{(1-\pi)\phi_0(Y_{ik})+\pi\phi_\tau(Y_{ik})}
=
\frac{r_{ik}-1}{1-\pi+\pi r_{ik}}.
$$

Equivalently, with

$$
w_{ik}
=
\mathbb P_\pi(Z_k=1\mid Y_{ik})
=
\frac{\pi r_{ik}}{1-\pi+\pi r_{ik}},
$$

we have

$$
s_{ik}^{(1)}
=
\frac{w_{ik}-\pi}{\pi(1-\pi)}.
$$

The marginal composite score is

$$
CS_1(Y;\pi)
=
\sum_{k=1}^K
\sum_{i=1}^n
s_{ik}^{(1)}.
$$

In general,

$$
CS_1(Y;\pi)\neq S_{\mathrm{full}}(Y;\pi),
$$

unless $n=1$.

---

## 1.4 Pairwise subscore

There are two types of pairwise subscores.

### Same-block pair

For $i\neq j$ within the same block $k$, the pairwise marginal density is

$$
p_\pi(Y_{ik},Y_{jk})
=
(1-\pi)\phi_0(Y_{ik})\phi_0(Y_{jk})
+
\pi\phi_\tau(Y_{ik})\phi_\tau(Y_{jk}).
$$

Equivalently,

$$
p_\pi(Y_{ik},Y_{jk})
=
\phi_0(Y_{ik})\phi_0(Y_{jk})
\left(1-\pi+\pi r_{ik}r_{jk}\right).
$$

The same-block pairwise subscore is

$$
s_{ij,k}^{(2)}
=
\frac{\partial}{\partial\pi}
\log p_\pi(Y_{ik},Y_{jk}).
$$

Hence

$$
s_{ij,k}^{(2)}
=
\frac{r_{ik}r_{jk}-1}
{1-\pi+\pi r_{ik}r_{jk}}.
$$

Equivalently, with

$$
w_{ij,k}
=
\mathbb P_\pi(Z_k=1\mid Y_{ik},Y_{jk})
=
\frac{\pi r_{ik}r_{jk}}{1-\pi+\pi r_{ik}r_{jk}},
$$

we have

$$
s_{ij,k}^{(2)}
=
\frac{w_{ij,k}-\pi}{\pi(1-\pi)}.
$$

The within-block pairwise composite score is

$$
CS_{2,\mathrm{within}}(Y;\pi)
=
\sum_{k=1}^K
\sum_{1\leq i<j\leq n}
s_{ij,k}^{(2)}.
$$

### Cross-block pair

For $k\neq \ell$, the latent variables $Z_k$ and $Z_\ell$ are independent. Therefore

$$
p_\pi(Y_{ik},Y_{j\ell})
=
p_\pi(Y_{ik})p_\pi(Y_{j\ell}).
$$

Hence the cross-block pairwise score is

$$
s_{ik,j\ell}^{(2)}
=
s_{ik}^{(1)}+s_{j\ell}^{(1)},
\qquad k\neq \ell.
$$

Thus cross-block pairwise scores add no new dependence information. The informative pairwise scores are the same-block scores $s_{ij,k}^{(2)}$.

---

## 1.5 Nonlinear recovery of the full score

The basic score map

$$
x\mapsto \frac{x-1}{1-\pi+\pi x}
$$

is invertible for $x>0$. If

$$
s=\frac{x-1}{1-\pi+\pi x},
$$

then

$$
x
=
A_\pi(s)
=
\frac{1+(1-\pi)s}{1-\pi s}.
$$

Therefore, from the marginal subscore,

$$
r_{ik}
=
A_\pi(s_{ik}^{(1)})
=
\frac{1+(1-\pi)s_{ik}^{(1)}}{1-\pi s_{ik}^{(1)}}.
$$

It follows that

$$
R_k
=
\prod_{i=1}^n r_{ik}
=
\prod_{i=1}^n
A_\pi(s_{ik}^{(1)}).
$$

Therefore the block full score is

$$
S_k
=
\frac{R_k-1}{1-\pi+\pi R_k}
=
\frac{
\prod_{i=1}^n A_\pi(s_{ik}^{(1)})-1
}
{
1-\pi+\pi\prod_{i=1}^n A_\pi(s_{ik}^{(1)})
}.
$$

Thus

$$
S_{\mathrm{full}}(Y;\pi)
=
\sum_{k=1}^K
\frac{
\prod_{i=1}^n A_\pi(s_{ik}^{(1)})-1
}
{
1-\pi+\pi\prod_{i=1}^n A_\pi(s_{ik}^{(1)})
}.
$$

So a nonlinear link of the marginal subscores alone can recover the exact full score.

For the pairwise subscore,

$$
r_{ik}r_{jk}
=
A_\pi(s_{ij,k}^{(2)})
=
\frac{1+(1-\pi)s_{ij,k}^{(2)}}{1-\pi s_{ij,k}^{(2)}}.
$$

For $n\geq 2$,

$$
\prod_{1\leq i<j\leq n}
r_{ik}r_{jk}
=
\left(
\prod_{i=1}^n r_{ik}
\right)^{n-1}
=
R_k^{n-1}.
$$

Therefore

$$
R_k
=
\left[
\prod_{1\leq i<j\leq n}
A_\pi(s_{ij,k}^{(2)})
\right]^{1/(n-1)}.
$$

Hence

$$
S_k
=
\frac{
\left[
\prod_{i<j}
A_\pi(s_{ij,k}^{(2)})
\right]^{1/(n-1)}
-1
}
{
1-\pi+\pi
\left[
\prod_{i<j}
A_\pi(s_{ij,k}^{(2)})
\right]^{1/(n-1)}
}.
$$

Thus

$$
S_{\mathrm{full}}(Y;\pi)
=
\sum_{k=1}^K
\frac{
\left[
\prod_{i<j}
A_\pi(s_{ij,k}^{(2)})
\right]^{1/(n-1)}
-1
}
{
1-\pi+\pi
\left[
\prod_{i<j}
A_\pi(s_{ij,k}^{(2)})
\right]^{1/(n-1)}
}.
$$

This shows exact nonlinear recovery from the same-block pairwise subscores when $n\geq 2$.

---

## 1.6 Log-transform representation

Define

$$
\varphi_\pi(s)
=
\log A_\pi(s)
=
\log\left(
\frac{1+(1-\pi)s}{1-\pi s}
\right),
$$

and

$$
h_\pi(u)
=
\frac{e^u-1}{1-\pi+\pi e^u}.
$$

For marginal subscores,

$$
\varphi_\pi(s_{ik}^{(1)})
=
\log r_{ik}.
$$

Hence

$$
\sum_{i=1}^n
\varphi_\pi(s_{ik}^{(1)})
=
\sum_{i=1}^n
\log r_{ik}
=
\log R_k.
$$

Therefore

$$
S_k
=
h_\pi
\left(
\sum_{i=1}^n
\varphi_\pi(s_{ik}^{(1)})
\right).
$$

So

$$
S_{\mathrm{full}}(Y;\pi)
=
\sum_{k=1}^K
h_\pi
\left(
\sum_{i=1}^n
\varphi_\pi(s_{ik}^{(1)})
\right).
$$

For pairwise subscores,

$$
\varphi_\pi(s_{ij,k}^{(2)})
=
\log(r_{ik}r_{jk}).
$$

Hence

$$
\sum_{i<j}
\varphi_\pi(s_{ij,k}^{(2)})
=
(n-1)\log R_k.
$$

Therefore

$$
S_k
=
h_\pi
\left(
\frac{1}{n-1}
\sum_{i<j}
\varphi_\pi(s_{ij,k}^{(2)})
\right).
$$

Thus

$$
S_{\mathrm{full}}(Y;\pi)
=
\sum_{k=1}^K
h_\pi
\left(
\frac{1}{n-1}
\sum_{i<j}
\varphi_\pi(s_{ij,k}^{(2)})
\right).
$$

This is an exact nonlinear composite-score representation.

---

# 2. Blockwise Common-Factor Variance Mixture

## 2.1 Model

Conditional on $Z_k=0$,

$$
Y_{ik}\mid Z_k=0 \stackrel{iid}{\sim} N(0,1),
\qquad i=1,\dots,n.
$$

Conditional on $Z_k=1$,

$$
Y_{ik}=U_k+\varepsilon_{ik},
$$

where

$$
U_k\sim N(0,\tau^2),
$$

and

$$
\varepsilon_{ik}\stackrel{iid}{\sim}N(0,1).
$$

Assume $U_k$ and all $\varepsilon_{ik}$ are mutually independent across $i$ and $k$.

Thus, under $Z_k=0$,

$$
Y_{\cdot k}\sim N(0,I_n),
$$

while under $Z_k=1$,

$$
Y_{\cdot k}\sim N(0,I_n+\tau^2\mathbf 1_n\mathbf 1_n^\top),
$$

where

$$
\mathbf 1_n=(1,\dots,1)^\top.
$$

---

## 2.2 Exact full Fisher score for $\pi$

Define

$$
f_{0,n}(Y_{\cdot k})
=
\phi_n(Y_{\cdot k};0,I_n),
$$

and

$$
f_{1,n}(Y_{\cdot k})
=
\phi_n(Y_{\cdot k};0,I_n+\tau^2\mathbf 1_n\mathbf 1_n^\top).
$$

The block likelihood ratio is

$$
R_{n,k}
=
\frac{f_{1,n}(Y_{\cdot k})}{f_{0,n}(Y_{\cdot k})}.
$$

Using

$$
\left|I_n+\tau^2\mathbf 1_n\mathbf 1_n^\top\right|
=
1+n\tau^2,
$$

and

$$
\left(I_n+\tau^2\mathbf 1_n\mathbf 1_n^\top\right)^{-1}
=
I_n-
\frac{\tau^2}{1+n\tau^2}
\mathbf 1_n\mathbf 1_n^\top,
$$

we obtain

$$
R_{n,k}
=
(1+n\tau^2)^{-1/2}
\exp\left\{
\frac{\tau^2}{2(1+n\tau^2)}
\left(\sum_{i=1}^n Y_{ik}\right)^2
\right\}.
$$

The block density is

$$
p_\pi(Y_{\cdot k})
=
(1-\pi) f_{0,n}(Y_{\cdot k})
+
\pi f_{1,n}(Y_{\cdot k})
=
f_{0,n}(Y_{\cdot k})(1-\pi+\pi R_{n,k}).
$$

The full joint density is

$$
p_\pi(Y)
=
\prod_{k=1}^K p_\pi(Y_{\cdot k}).
$$

Therefore the exact Fisher score is

$$
S_{\mathrm{full}}(Y;\pi)
=
\sum_{k=1}^K
\frac{R_{n,k}-1}{1-\pi+\pi R_{n,k}}.
$$

Equivalently, with

$$
w_k(Y_{\cdot k})
=
\mathbb P_\pi(Z_k=1\mid Y_{\cdot k})
=
\frac{\pi R_{n,k}}{1-\pi+\pi R_{n,k}},
$$

we have

$$
S_{\mathrm{full}}(Y;\pi)
=
\sum_{k=1}^K
\frac{w_k(Y_{\cdot k})-\pi}{\pi(1-\pi)}.
$$

The full score depends on each block through

$$
\left(\sum_{i=1}^n Y_{ik}\right)^2.
$$

---

## 2.3 Marginal subscore

For a single observation $Y_{ik}$, under $Z_k=0$,

$$
Y_{ik}\sim N(0,1),
$$

while under $Z_k=1$,

$$
Y_{ik}=U_k+\varepsilon_{ik}\sim N(0,1+\tau^2).
$$

Define

$$
f_{0,1}(y)=\phi(y;0,1),
$$

and

$$
f_{1,1}(y)=\phi(y;0,1+\tau^2).
$$

The one-dimensional likelihood ratio is

$$
R_{1,ik}
=
\frac{f_{1,1}(Y_{ik})}{f_{0,1}(Y_{ik})}.
$$

Explicitly,

$$
R_{1,ik}
=
(1+\tau^2)^{-1/2}
\exp\left\{
\frac{\tau^2}{2(1+\tau^2)}
Y_{ik}^2
\right\}.
$$

The marginal density is

$$
p_\pi(Y_{ik})
=
qf_{0,1}(Y_{ik})
+
\pi f_{1,1}(Y_{ik})
=
f_{0,1}(Y_{ik})(1-\pi+\pi R_{1,ik}).
$$

The marginal subscore is

$$
s_{ik}^{(1)}
=
\frac{\partial}{\partial\pi}
\log p_\pi(Y_{ik}).
$$

Thus

$$
s_{ik}^{(1)}
=
\frac{R_{1,ik}-1}{1-\pi+\pi R_{1,ik}}.
$$

This subscore contains information about $Y_{ik}^2$, but not directly about cross-products $Y_{ik}Y_{jk}$.

---

## 2.4 Pairwise subscore

There are again two pair types.

### Same-block pair

For $i\neq j$ in the same block $k$, under $Z_k=0$,

$$
(Y_{ik},Y_{jk})^\top\sim N(0,I_2).
$$

Under $Z_k=1$,

$$
(Y_{ik},Y_{jk})^\top
\sim
N(0,I_2+\tau^2\mathbf 1_2\mathbf 1_2^\top).
$$

The alternative covariance matrix is

$$
I_2+\tau^2\mathbf 1_2\mathbf 1_2^\top
=
\begin{pmatrix}
1+\tau^2 & \tau^2\\
\tau^2 & 1+\tau^2
\end{pmatrix}.
$$

Define

$$
R_{2,ij,k}
=
\frac{
\phi_2((Y_{ik},Y_{jk})^\top;0,I_2+\tau^2\mathbf 1_2\mathbf 1_2^\top)
}
{
\phi_2((Y_{ik},Y_{jk})^\top;0,I_2)
}.
$$

Then

$$
R_{2,ij,k}
=
(1+2\tau^2)^{-1/2}
\exp\left\{
\frac{\tau^2}{2(1+2\tau^2)}
(Y_{ik}+Y_{jk})^2
\right\}.
$$

The same-block pairwise subscore is

$$
s_{ij,k}^{(2)}
=
\frac{\partial}{\partial\pi}
\log p_\pi(Y_{ik},Y_{jk}).
$$

Therefore

$$
s_{ij,k}^{(2)}
=
\frac{R_{2,ij,k}-1}
{1-\pi+\pi R_{2,ij,k}}.
$$

This pairwise score contains information about

$$
(Y_{ik}+Y_{jk})^2
=
Y_{ik}^2+Y_{jk}^2+2Y_{ik}Y_{jk}.
$$

Thus it contains cross-product information.

### Cross-block pair

For $k\neq \ell$, the latent variables $Z_k$ and $Z_\ell$ are independent. Hence

$$
p_\pi(Y_{ik},Y_{j\ell})
=
p_\pi(Y_{ik})p_\pi(Y_{j\ell}).
$$

Therefore the cross-block pairwise score is

$$
s_{ik,j\ell}^{(2)}
=
s_{ik}^{(1)}+s_{j\ell}^{(1)},
\qquad k\neq \ell.
$$

Cross-block pairs are redundant for dependence information. The useful pairwise scores are the within-block scores.

---

## 2.5 Nonlinear recovery of the full score

Again use the inverse map

$$
A_\pi(s)
=
\frac{1+(1-\pi)s}{1-\pi s}.
$$

From the marginal subscore,

$$
R_{1,ik}
=
A_\pi(s_{ik}^{(1)}).
$$

From the same-block pairwise subscore,

$$
R_{2,ij,k}
=
A_\pi(s_{ij,k}^{(2)}).
$$

Define the log-transformed subscores

$$
\ell_{ik}^{(1)}
=
\log A_\pi(s_{ik}^{(1)}),
$$

and

$$
\ell_{ij,k}^{(2)}
=
\log A_\pi(s_{ij,k}^{(2)}).
$$

Then

$$
\ell_{ik}^{(1)}
=
-\frac{1}{2}\log(1+\tau^2)
+
\frac{\tau^2}{2(1+\tau^2)}
Y_{ik}^2,
$$

and

$$
\ell_{ij,k}^{(2)}
=
-\frac{1}{2}\log(1+2\tau^2)
+
\frac{\tau^2}{2(1+2\tau^2)}
(Y_{ik}+Y_{jk})^2.
$$

Therefore,

$$
Y_{ik}^2
=
\frac{2(1+\tau^2)}{\tau^2}
\left\{
\ell_{ik}^{(1)}
+
\frac{1}{2}\log(1+\tau^2)
\right\},
$$

and

$$
(Y_{ik}+Y_{jk})^2
=
\frac{2(1+2\tau^2)}{\tau^2}
\left\{
\ell_{ij,k}^{(2)}
+
\frac{1}{2}\log(1+2\tau^2)
\right\}.
$$

The full block likelihood ratio requires

$$
B_k
=
\left(\sum_{i=1}^n Y_{ik}\right)^2.
$$

Use the identity

$$
\sum_{i<j}(Y_{ik}+Y_{jk})^2
=
(n-2)\sum_{i=1}^n Y_{ik}^2
+
\left(\sum_{i=1}^n Y_{ik}\right)^2.
$$

Hence

$$
B_k
=
\sum_{i<j}(Y_{ik}+Y_{jk})^2
-
(n-2)\sum_{i=1}^n Y_{ik}^2.
$$

Substituting the recovered quantities gives

$$
B_k
=
\frac{2(1+2\tau^2)}{\tau^2}
\sum_{i<j}
\left\{
\ell_{ij,k}^{(2)}
+
\frac{1}{2}\log(1+2\tau^2)
\right\}
-
(n-2)
\frac{2(1+\tau^2)}{\tau^2}
\sum_{i=1}^n
\left\{
\ell_{ik}^{(1)}
+
\frac{1}{2}\log(1+\tau^2)
\right\}.
$$

Once $B_k$ is recovered, the full block likelihood ratio is

$$
R_{n,k}
=
(1+n\tau^2)^{-1/2}
\exp\left\{
\frac{\tau^2}{2(1+n\tau^2)}
B_k
\right\}.
$$

Then the block full score is

$$
S_k
=
\frac{R_{n,k}-1}{1-\pi+\pi R_{n,k}}.
$$

Finally,

$$
S_{\mathrm{full}}(Y;\pi)
=
\sum_{k=1}^K S_k.
$$

Thus the full Fisher score is recovered exactly by a nonlinear link of the marginal and same-block pairwise subscores.

---

## 2.6 nCS-style representation

Define

$$
\varphi_1(s)
=
\log A_\pi(s)
+
\frac{1}{2}\log(1+\tau^2),
$$

and

$$
\varphi_2(s)
=
\log A_\pi(s)
+
\frac{1}{2}\log(1+2\tau^2).
$$

Then

$$
\varphi_1(s_{ik}^{(1)})
=
\frac{\tau^2}{2(1+\tau^2)}
Y_{ik}^2,
$$

and

$$
\varphi_2(s_{ij,k}^{(2)})
=
\frac{\tau^2}{2(1+2\tau^2)}
(Y_{ik}+Y_{jk})^2.
$$

Define the blockwise transformed summaries

$$
U_{1,k}
=
\sum_{i=1}^n
\varphi_1(s_{ik}^{(1)}),
$$

and

$$
U_{2,k}
=
\sum_{1\leq i<j\leq n}
\varphi_2(s_{ij,k}^{(2)}).
$$

Then

$$
B_k
=
\frac{2(1+2\tau^2)}{\tau^2}U_{2,k}
-
(n-2)
\frac{2(1+\tau^2)}{\tau^2}U_{1,k}.
$$

Define

$$
R(u_1,u_2)
=
(1+n\tau^2)^{-1/2}
\exp\left[
\frac{\tau^2}{2(1+n\tau^2)}
\left\{
\frac{2(1+2\tau^2)}{\tau^2}u_2
-
(n-2)
\frac{2(1+\tau^2)}{\tau^2}u_1
\right\}
\right],
$$

and

$$
H_{\pi,\tau,n}(u_1,u_2)
=
\frac{R(u_1,u_2)-1}{1-\pi+\pi R(u_1,u_2)}.
$$

Then

$$
S_{\mathrm{full}}(Y;\pi)
=
\sum_{k=1}^K
H_{\pi,\tau,n}
\left(
\sum_{i=1}^n
\varphi_1(s_{ik}^{(1)}),
\sum_{i<j}
\varphi_2(s_{ij,k}^{(2)})
\right).
$$

This is an exact nonlinear composite-score representation.

---

# 3. Summary

For the shared mean-shift mixture,

$$
Y_{ik}\mid Z_k=1\sim N(\tau,1),
$$

the block likelihood ratio depends on

$$
\sum_{i=1}^n Y_{ik}.
$$

A nonlinear transformation of marginal scores recovers each $\log r_{ik}$, so summing transformed marginal scores recovers $\log R_k$. Pairwise transformed scores also recover $\log R_k$ after averaging by $n-1$.

For the common-factor variance mixture,

$$
Y_{ik}\mid Z_k=1=U_k+\varepsilon_{ik},
$$

the block likelihood ratio depends on

$$
\left(\sum_{i=1}^n Y_{ik}\right)^2.
$$

Marginal subscores recover $Y_{ik}^2$, while pairwise subscores recover $(Y_{ik}+Y_{jk})^2$. Combining them nonlinearly recovers

$$
\left(\sum_{i=1}^n Y_{ik}\right)^2,
$$

and hence recovers the full Fisher score.

Therefore, in both models, the linear composite score may be insufficient, but a nonlinear link applied to marginal and same-block pairwise subscores can recover the exact full Fisher score.
