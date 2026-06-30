# 3. Hidden Markov Common-Factor Variance Mixture

This example is the most natural bridge from the two blockwise iid latent-mixture examples to a realistic market-stress time-series model.

The independent block indicators $Z_k\stackrel{iid}{\sim}\mathrm{Bernoulli}(\pi)$ are replaced by a two-state Markov chain $B_t\in\{0,1\}, \qquad P(B_t=b\mid B_{t-1}=a)=P_{ab}, \qquad t=1,\ldots,T.$ At each time $t$, observe a cross-sectional block $Y_t=(Y_{t1},\ldots,Y_{tn})^\top.$ The emission distribution is the common-factor variance mixture from Example 2: $Y_t\mid B_t=0\sim N(0,I_n),$ and $Y_t\mid B_t=1\sim N(0,I_n+\tau^2\mathbf 1_n\mathbf 1_n^\top).$ Here $B_t=1$ represents a persistent common-factor or stress state.

---

## 3.1 Local emission likelihood ratio

Define $f_0(Y_t)=\phi_n(Y_t;0,I_n),$ and $f_1(Y_t)=\phi_n(Y_t;0,I_n+\tau^2\mathbf 1_n\mathbf 1_n^\top).$ The time-local emission likelihood ratio is $R_t=\frac{f_1(Y_t)}{f_0(Y_t)}.$ As in Example 2, $R_t = (1+n\tau^2)^{-1/2} \exp\left\{ \frac{\tau^2}{2(1+n\tau^2)} \left(\sum_{i=1}^n Y_{ti}\right)^2 \right\}.$ Equivalently, $L_t=\log R_t = -\frac12\log(1+n\tau^2) + \frac{\tau^2}{2(1+n\tau^2)} \left(\sum_{i=1}^n Y_{ti}\right)^2.$ Thus the Markov model preserves the same local nonlinear recovery problem as Example 2, but adds sequential aggregation over time.

---

## 3.2 Marginal and pairwise subscores

For each time $t$, define marginal and within-block pairwise subscores exactly as in Example 2: $s_{ti}^{(1)} = \frac{R_{1,ti}-1}{1-\pi+\pi R_{1,ti}},$ where $R_{1,ti} = (1+\tau^2)^{-1/2} \exp\left\{ \frac{\tau^2}{2(1+\tau^2)}Y_{ti}^2 \right\},$ and $s_{tij}^{(2)} = \frac{R_{2,tij}-1}{1-\pi+\pi R_{2,tij}},$ where $R_{2,tij} = (1+2\tau^2)^{-1/2} \exp\left\{ \frac{\tau^2}{2(1+2\tau^2)}(Y_{ti}+Y_{tj})^2 \right\}.$ Here the symbol $\pi$ is used only as a reference mixing probability defining the local subscore map. In the HMM model, the actual state probabilities are determined by the Markov transition matrix and by the observed sequence.

As before, define $A_\pi(s)=\frac{1+(1-\pi)s}{1-\pi s},$ and $\ell_{ti}^{(1)}=\log A_\pi(s_{ti}^{(1)}), \qquad \ell_{tij}^{(2)}=\log A_\pi(s_{tij}^{(2)}).$ Then the transformed marginal subscores recover $Y_{ti}^2$, and the transformed pairwise subscores recover $(Y_{ti}+Y_{tj})^2$.

Therefore the block quantity
$$
B_t^\star = \left(\sum_{i=1}^nY_{ti}\right)^2
$$
can be recovered from transformed local subscores by
$$
B_t^\star = \frac{2(1+2\tau^2)}{\tau^2} \sum_{i<j} \left\{ \ell_{tij}^{(2)}+\frac12\log(1+2\tau^2) \right\} - (n-2) \frac{2(1+\tau^2)}{\tau^2} \sum_i \left\{ \ell_{ti}^{(1)}+\frac12\log(1+\tau^2) \right\}.
$$
Hence nCS can recover $L_t = -\frac12\log(1+n\tau^2) + \frac{\tau^2}{2(1+n\tau^2)}B_t^\star.$

---

## 3.3 Full observed likelihood and HMM score

The observed likelihood is $p(Y_{1:T}) = \sum_{b_{1:T}} \mu_{b_1}f_{b_1}(Y_1) \prod_{t=2}^T P_{b_{t-1}b_t}f_{b_t}(Y_t).$ Using the local log-likelihood ratio $L_t$, one may take the emission log-potentials as $e_t(0)=0, \qquad e_t(1)=L_t.$ The forward recursion is $\alpha_1(b)=\log \mu_b+e_1(b),$ and $\alpha_t(b) = e_t(b)+ \log\sum_a \exp\{\alpha_{t-1}(a)+\log P_{ab}\}.$ The backward recursion is initialized by $\beta_T(b)=0,$ and $\beta_{t-1}(a) = \log\sum_b \exp\{\log P_{ab}+e_t(b)+\beta_t(b)\}.$ These give the smoothing probabilities $\gamma_t(b)=P(B_t=b\mid Y_{1:T}),$ and $\xi_t(a,b)=P(B_{t-1}=a,B_t=b\mid Y_{1:T}).$ For a transition parameter, for example $p=P_{11}$ with $P_{10}=1-p$, the full score is $S_p^{\mathrm{full}} = \sum_{t=2}^T \left\{ \frac{\xi_t(1,1)}{p} - \frac{\xi_t(1,0)}{1-p} \right\}.$ Therefore Example 3 has an exact structured nonlinear representation:
$$
\{s_{ti}^{(1)},s_{tij}^{(2)}\} \quad\longrightarrow\quad L_t \quad\longrightarrow\quad \{\gamma_t,\xi_t\} \quad\longrightarrow\quad S_{\mathrm{full}}.
$$
This is the analytical prototype of a sequential nCS method.

---

## 3.4 Why this example follows naturally from Examples 1 and 2

Example 1 shows that nCS should transform local subscores into log-likelihood-ratio evidence before summing.

Example 2 shows that, when the latent state changes cross-sectional dependence, marginal and pairwise transformed subscores are needed to recover common-factor evidence.

Example 3 adds only one new feature: temporal persistence. The local block evidence is still recovered in the same way, but the final score now requires HMM smoothing rather than independent blockwise summation.

The linear composite score loses information because it compresses the local subscores as
$$
\sum_t\sum_i s_{ti}^{(1)} + \sum_t\sum_{i<j}s_{tij}^{(2)},
$$
whereas the full score uses the sequence $L_1,\ldots,L_T,$ inside the nonlinear forward-backward recursion.
