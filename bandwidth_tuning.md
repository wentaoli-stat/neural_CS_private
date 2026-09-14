# Bandwidth Tuning

## Tuning for Fisher-Score Approximation

Let \(M_e \in \mathbb{R}^{p\times p}\) denote the estimated covariance shape of the pilot error \(\widehat\beta(X)-\theta\), where \(\theta \in \mathbb{R}^p\) is the generating parameter and \(\widehat\beta(X)\) is the pilot estimator. Parameterize the tube covariance as

\[
\Sigma_q(c)=c^2M_e,\qquad c>0,
\]

so that the anisotropic shape is determined by the empirically estimated pilot uncertainty and only the scalar width \(c\) needs tuning. This avoids the exponentially growing search over diagonal or full covariance matrices noted as problematic by Khoo et al. The pilot-error geometry also makes the tube directly relevant to the off-diagonal displacement encountered at deployment.

For each candidate \(c\), train the surrogate score \(\widehat S_c(X,\beta)\), where \(\beta\in\mathbb{R}^p\) is the score-evaluation point. A principled validation criterion is the **diagonal Fisher-score matching risk**, evaluated using independent draws \(\beta\sim\lambda_{\rm val}\) and \(X\sim p(\cdot\mid\beta)\), where \(\lambda_{\rm val}\) is a validation distribution over evaluation points. Up to a constant independent of \(c\), the unavailable Fisher-score MSE

\[
\mathbb E\left\|
\widehat S_c(X,\beta)-\nabla_\beta\log p(X\mid\beta)
\right\|^2
\]

can be estimated by

\[
\widehat R_{\rm score}(c)
=
\mathbb E
\left[
\|\widehat S_c(X,\beta)\|^2
+
2\,\operatorname{div}_\beta \widehat S_c(X,\beta)
+
2\,\widehat S_c(X,\beta)^\top
\nabla_\beta\log\lambda_{\rm val}(\beta)
\right],
\]

where

\[
\operatorname{div}_\beta\widehat S_c
=
\operatorname{tr}
\left\{
\frac{\partial\widehat S_c(X,\beta)}
{\partial\beta^\top}
\right\}.
\]

Select

\[
\widehat c_{\rm score}
=
\arg\min_c\widehat R_{\rm score}(c).
\]

This directly targets approximation of the diagonal Fisher score rather than downstream data fit, unlike the heuristic of Khoo et al. Its main computational cost is differentiation with respect to \(\beta\), which propagates through the local scores and may therefore require second derivatives of the local log likelihoods. Because this is needed only on a validation set rather than throughout training, it may nevertheless be practical.

---

## Tuning for Posterior Inference

Posterior tuning should instead target the quality of the **final posterior conditional on the pilot-score summary**. For candidate tube width \(c\), define

\[
Z_c(X)
=
\left\{
\widehat\beta(X),
\widehat S_c\bigl(X,\widehat\beta(X)\bigr)
\right\},
\]

and let \(q_{\phi,c}(\theta\mid Z_c)\) denote the neural posterior estimator with trainable parameters \(\phi\).

The primary tuning criterion should be **held-out prior-predictive negative log probability**. Using validation draws

\[
\theta_i\sim\pi,\qquad X_i\sim p(\cdot\mid\theta_i),
\]

where \(\pi\) is the prior, select the NPE hyperparameters, checkpoint, and—if desired—the tube width according to

\[
(\widehat c_{\rm post},\widehat\phi)
=
\arg\min_{c,\phi}
\frac1{N_{\rm val}}
\sum_{i=1}^{N_{\rm val}}
-\log q_{\phi,c}
\left\{
\theta_i\mid Z_c(X_i)
\right\}.
\]

This criterion simultaneously rewards a summary \(Z_c\) that retains more information about \(\theta\) and an NPE that accurately represents the posterior conditional on that summary. Consequently, \(\widehat c_{\rm post}\) need not equal the width \(\widehat c_{\rm score}\) that minimizes Fisher-score approximation error.

Posterior log score should be accompanied by **calibration diagnostics**, rather than replaced by them. In particular, simulation-based calibration, empirical credible-interval coverage, or a joint multivariate method such as TARP can check whether posterior uncertainty is calibrated; coverage alone should not be optimized because an overly diffuse posterior can achieve nominal coverage.

Finally, full NPE training for every candidate \(c\) can be reduced using a cheap **one-step screening criterion**. Let

\[
\Psi_c(X,\beta)
=
\widehat S_c(X,\beta)
-
E_\beta\{\widehat S_c(X,\beta)\}
\]

be the centered surrogate score, and let \(H_c(\beta)\in\mathbb{R}^{p\times p}\) be its local sensitivity matrix. Form

\[
\widehat\theta_{1,c}
=
\widehat\beta(X)
+
H_c\{\widehat\beta(X)\}^{-1}
\Psi_c\{X,\widehat\beta(X)\}.
\]

Candidate widths can first be ranked by the validation error of \(\widehat\theta_{1,c}\), for example

\[
E\left[
\left\|
M_e^{-1/2}
(\widehat\theta_{1,c}-\theta)
\right\|^2
\right].
\]

Only the best few values of \(c\) then need full NPE training and comparison by held-out posterior log score and calibration.
