# Methods

## 1. Overview

We study whether nonlinear calibration of tractable local likelihood scores can
retain information that is otherwise lost during pooling. The main comparison
is deliberately narrow. Linear and the Nonlinear gate use the same simulator,
local scores, pooling rule, block readout, training datasets, optimization
budget, and posterior estimator. Their defining difference is the local map
applied before pooling.

The methodology is presented in two parts. Section 2 gives the general method
without reference to a particular simulator. Section 3 defines the
supplementary direct posterior baseline used for the two mixture experiments.
Sections 4 to 6 then instantiate the same framework in three experiments: a
Gaussian mean shift mixture, a Gaussian common factor mixture, and a Smith max
stable rainfall model.

The first two experiments are controlled methodological benchmarks. Their
likelihood structure is analytically tractable, so the source of any nonlinear
advantage can be studied directly. The third experiment transfers the method
to a spatial extreme value setting in which only bivariate likelihood scores
are tractable.

This sequence has a specific purpose. Model 1 isolates the simplest loss of
information caused by pooling a nonlinear score link. Model 2 retains a scalar
estimand but requires two local channels to represent dependence inside a
block. The max stable experiment then asks whether the same local calibration
principle remains useful when neither an analytic complete score nor an inverse
link is available. The experiments therefore form a progression from
mechanism, to structural stress, to scientific application.

## 2. General methodology

### 2.1 Parameters and notation

Let

$$
\boldsymbol\theta\in\Theta\subseteq\mathbb R^d
$$

denote the scientific parameter of interest. Computation is performed in an
unconstrained coordinate
$$
\boldsymbol\beta=r(\boldsymbol\theta)\in\mathbb R^d.
$$

A score evaluation point, also called a reference point or anchor, is denoted
by $\boldsymbol\beta_0$. A parameter drawn from its local proposal is denoted
by $\widetilde{\boldsymbol\beta}$. This notation separates the scientific
parameter, the computational coordinate, the score evaluation point, and the
random perturbation. For the two mixture experiments, $d=1$,
$\theta=\pi$, and

$$
\beta=\operatorname{logit}(\pi),
\qquad
\pi=\operatorname{sigmoid}(\beta).
$$

For the max stable experiment, $d=3$, and the transformation from the
bounded parameter to $\boldsymbol\beta$ is given in Section 6. Bold notation is
used for vectors. In the scalar experiments the bold symbols reduce to their
scalar counterparts.

The original local FSM formulation denotes the current optimization iterate by
$\boldsymbol\theta_t$ and a proposal draw by $\boldsymbol\theta$. Our
method learns one field over many evaluation points instead of fitting a new
model at each sequential iterate. We therefore use
$\boldsymbol\beta_0$ rather than $\boldsymbol\theta_t$, and reserve
$\boldsymbol\theta$ for the scientific parameter on its original scale.

The transformation to $\boldsymbol\beta$ has three purposes. It converts a
constrained parameter space into an unconstrained Euclidean space, allowing
Gaussian perturbations without boundary clipping. It permits pilot optimization
with ordinary gradients and curvature while preventing invalid probabilities
or covariance parameters. It also guarantees that every finite transformed
value maps back to a valid scientific parameter. Because the map is invertible,
no parameter information is discarded, and posterior summaries can be returned
to the original scientific scale.

For a mixing probability, the logit is particularly convenient because it maps
$(0,1)$ bijectively to $\mathbb R$. It also gives a simple likelihood score:
the posterior probability of the active component minus the mixing probability.
This identity is derived explicitly in Sections 4 and 5.

The score is learned with respect to $\beta$, not directly with respect to
$\pi$. The $\beta$ score remains bounded because it is a difference of two
probabilities. By the chain rule, the corresponding $\pi$ score contains the
factor $\{\pi(1-\pi)\}^{-1}$, which becomes poorly scaled near the boundary.
Using $\beta$ therefore improves numerical stability while preserving the same
underlying parameter information.

We use $\ell$ for a local log likelihood ratio, $L$ for a complete block
log likelihood ratio, $\boldsymbol s$ for a local score, and
$\boldsymbol S$ for a complete data score. The index $k$ always denotes a
block. The index $j$ denotes a local contribution inside a block.

### 2.2 Amortized forward score matching

Stage 1 uses forward score matching, abbreviated FSM, to learn a score field
over reference points. A reference $\boldsymbol\beta_0$ is sampled from a
fixed design distribution. A perturbed coordinate is then drawn from

$$
\widetilde{\boldsymbol\beta}\mid\boldsymbol\beta_0
\sim
\mathcal N(\boldsymbol\beta_0,\boldsymbol\Sigma_q),
$$

and a complete dataset is generated from

$$
Y\sim p(\,\cdot\mid\widetilde{\boldsymbol\beta}).
$$

Amortization is needed because Stage 2 evaluates the score at a different
pilot for every dataset. A model trained at one fixed reference point would
have to be refitted whenever that evaluation point changed. Sampling reference
points throughout the design region instead trains one reusable field,

$$
(Y,\boldsymbol\beta_0)
\longmapsto
\widehat{\boldsymbol S}(Y,\boldsymbol\beta_0),
$$

so the Stage 1 cost is paid once and the frozen field can be queried at every
dataset specific pilot.

The Gaussian proposal is used because its score with respect to the reference
is known even when the simulator likelihood score is unavailable. This makes
it possible to learn from simulations without differentiating the simulator
or evaluating the complete likelihood. Its covariance controls locality.
Smaller variance reduces smoothing but increases the variance of the FSM
target; larger variance stabilizes the target but produces more smoothing. The
proposal widths below are fixed compromises between these effects.

The FSM target is the score of the Gaussian proposal with respect to its
reference:

$$
\boldsymbol t(\widetilde{\boldsymbol\beta},\boldsymbol\beta_0)
=
\nabla_{\boldsymbol\beta_0}
\log q(\widetilde{\boldsymbol\beta}\mid\boldsymbol\beta_0)
=
\boldsymbol\Sigma_q^{-1}
(\widetilde{\boldsymbol\beta}-\boldsymbol\beta_0).
$$

The score network $\widehat{\boldsymbol S}_{\psi}$ is trained by minimizing

$$
\mathcal L_{\mathrm{FSM}}(\psi)
=
\mathbb E
\left[
\left\|
\widehat{\boldsymbol S}_{\psi}(Y,\boldsymbol\beta_0)
-
\boldsymbol t(\widetilde{\boldsymbol\beta},\boldsymbol\beta_0)
\right\|_2^2
\right].
$$

Squared error is chosen because its population minimizer is the conditional
mean of the accessible proposal target. The conditional expectation identity
below then converts that regression problem into the desired smoothed
likelihood score. A generic prediction loss would not provide this identity.

The population minimizer is

$$
\widehat{\boldsymbol S}^{*}(Y,\boldsymbol\beta_0)
=
\mathbb E
\left[
\boldsymbol t(\widetilde{\boldsymbol\beta},\boldsymbol\beta_0)
\mid Y,\boldsymbol\beta_0
\right]
=
\nabla_{\boldsymbol\beta_0}
\log
\int
p(Y\mid\widetilde{\boldsymbol\beta})
q(\widetilde{\boldsymbol\beta}\mid\boldsymbol\beta_0)
\,d\widetilde{\boldsymbol\beta}.
$$

Thus, at nonzero proposal width, FSM targets the score of a Gaussian smoothed
likelihood rather than the exact likelihood score at the reference point.
Exact complete likelihood scores, when available, are reserved for evaluation
after training. They are not used in the training loss, hyperparameter
selection, or checkpoint selection.

This separation is required for a meaningful likelihood free comparison. An
exact score used during training or selection would give the tractable mixture
experiments information that is unavailable in the spatial application. It
would also turn the exact reference into part of the method rather than an
independent diagnostic.

### 2.3 Local composite score representation

Write a complete dataset as $K$ blocks. Within block $k$, suppose that a
collection of tractable likelihood components provides local score tokens

$$
\boldsymbol s_{kj}(\boldsymbol\beta_0)
=
\left.
\nabla_{\boldsymbol\beta}
\log f_{kj}(Y_{kj}\mid\boldsymbol\beta)
\right|_{\boldsymbol\beta=\boldsymbol\beta_0}.
$$

The index $j$ may identify a single observation, an observation pair, or a
spatial pair. Several types of token may be retained as separate channels.
Every normalization constant is estimated from the Stage 1 training bank and
then frozen.

Local composite scores provide a middle ground between an unavailable complete
likelihood and unstructured raw observations. They remain tractable in all
three experiments, retain known conditional or spatial structure, and reduce
the amount of structure that the neural network must discover from simulations.
Estimating normalization constants from training data only prevents validation
or test information from entering preprocessing.

Let $\boldsymbol x_{kj}$ denote the coordinate in which the local map is
applied. It is an invertible affine transformation of the local score, or the
raw local score itself. This distinction is stated explicitly for each
experiment. The general architecture is

$$
\widehat{\boldsymbol S}(Y,\boldsymbol\beta_0)
=
\sum_{k=1}^{K}
\rho_{\omega}
\left[
\operatorname{Pool}_{j}
\left\{
\phi_{\eta}
(\boldsymbol x_{kj},\overline{\boldsymbol\beta}_0)
\right\},
\overline{\boldsymbol\beta}_0
\right].
$$

Here $\overline{\boldsymbol\beta}_0$ is the standardized reference coordinate.
The final implementations pass only the local score coordinate and the
reference coordinate to the local map. In particular, pair distance and pair
direction are not gate inputs in the max stable experiment. Distance is used
only to assign spatial pairs to fixed pooling bins. The function
$\rho_{\omega}$ is a shared nonlinear MLP. Its output is a contribution to
the complete data score, and these contributions are summed across blocks.
When several score channels are present, each channel is pooled separately and
the pooled values are concatenated before entering $\rho_{\omega}$.

Pooling converts a collection of local contributions into a fixed dimensional
representation. A mean is used within a channel so that its scale is not
determined only by the number of tokens. A sum is used across independent
blocks because likelihood scores add across independent observations. The
shared readout allows a nonlinear conversion from pooled evidence to a block
score while enforcing permutation invariance. In the spatial experiment the
ordered distance bins preserve coarse design information without conditioning
the local gate on pair geometry.

The block sum enforces permutation invariance across blocks. The pooling rule
enforces the relevant invariance inside each block. The complete likelihood
score is never supplied as an input to this architecture.

The additive block form is motivated by conditional independence: at a fixed
parameter, exact log likelihood scores add across independent blocks or years.
It also makes the model valid for arbitrary permutations of those replicates.
There is, however, a deliberate approximation at finite proposal width. The
Gaussian smoothing integral uses one common perturbed parameter for all blocks
and can induce interactions between blocks in the population FSM target. The
additive architecture cannot represent every such interaction. It is retained
as a transparent structural bias and is shared by Linear and the Nonlinear
gate, so the comparison isolates the local pooling bottleneck rather than
changing the complete data architecture.

### 2.4 Linear and Nonlinear gate local maps

The Linear arm uses the identity local map

$$
\phi_{L}(\boldsymbol x,\overline{\boldsymbol\beta}_0)
=
\boldsymbol x.
$$

The term Linear refers only to this identity map. The readout
$\rho_{\omega}$ remains nonlinear. Linear therefore means that no additional
nonlinear transformation is applied before pooling.

We use **Nonlinear gate** as the displayed method name in all three
experiments. The historical implementation keys are `radial` in Model 1,
`shared_radial` in Model 2, and `positive_anchor` in the max stable experiment.
The word positive below describes the constraint on the multiplier, not a
separate method name.

The Nonlinear gate arm uses

$$
\phi_{N}(\boldsymbol x,\overline{\boldsymbol\beta}_0)
=
\boldsymbol x
\odot
\boldsymbol m_{\eta}
(\boldsymbol x,\overline{\boldsymbol\beta}_0),
\qquad
\boldsymbol m_{\eta}>\boldsymbol 0,
$$

where positivity is understood component by component. The multiplier is
produced by an MLP followed by a softplus transformation. In scalar notation,

$$
m_{\eta}(x,\overline{\beta}_0)
=
\operatorname{softplus}
\{g_{\eta}(x,\overline{\beta}_0)\}.
$$

Softplus is used because it is smooth, strictly positive, and has a finite bias
that produces multiplier one exactly. It is less prone to explosive multipliers
than an unrestricted exponential output. SiLU activations are used inside the
gate and readout because they are smooth and provide stable gradients for a
continuously indexed score field.

The reference coordinate is included because the meaning of a local score
depends on the parameter value at which it is evaluated. In the mixture models
the link from a log likelihood ratio to a score changes directly with the
reference point. Conditioning the gate on this coordinate allows one amortized
calibration to adapt across the entire training region rather than learning one
compromise transformation.

This form changes the magnitude of the chosen local coordinate without changing
its sign. Multiplication by $\boldsymbol x$ also guarantees that a zero local
coordinate remains zero.

These restrictions are intentional. In the raw score implementations,
coordinatewise positivity preserves the direction of local score evidence and
zero preservation prevents the gate from creating evidence in a coordinate
whose score is zero. Model 1 applies the same multiplier form to a standardized
score coordinate; there the sign and zero statements refer to that frozen
feature coordinate rather than directly to the raw score. In either case the
constraint is weaker than specifying the correct nonlinear map, so it remains
usable when no analytic inverse is known.

The final layer of $g_{\eta}$ has zero weights and bias

$$
b_{\mathrm{id}}
=
\operatorname{softplus}^{-1}(1)
=
\log(e-1).
$$

Consequently,

$$
\boldsymbol m_{\eta}=\boldsymbol 1,
\qquad
\phi_{N}=\phi_{L}
$$

at initialization. The Nonlinear gate function class therefore contains the
Linear starting point exactly. The max stable implementation divides the
softplus output by the same initialized scalar so that this equality also holds
exactly in floating point arithmetic.

Identity initialization is used so the nonlinear arm begins with the same
local representation as Linear. Any improvement must be learned from the FSM
objective, while failure to improve can return to the identity neighborhood.
This is more interpretable than comparing two unrelated random feature maps and
also provides a direct numerical nesting test for the implementation.

An unrestricted residual map of the form

$$
\phi(\boldsymbol x)
=
\boldsymbol x+R_{\eta}(\boldsymbol x)
$$

is more general, but it may change signs, create nonzero output from a zero
score coordinate, and mix directions. It is not the primary nonlinear method
considered here.

### 2.5 What the local gate changes

Pooling is generally many to one. Once local values have been replaced by a
mean or a small vector of means, information about their distribution cannot
be recovered by the subsequent readout. The Nonlinear gate arm acts before this
irreversible operation. Its purpose is therefore not merely to add another
nonlinear layer. Its purpose is to alter which local information survives the
fixed pooling bottleneck.

The readout alone cannot repair this loss. If two datasets have the same
Linear pooled summary, even an arbitrarily expressive readout must assign them
the same output. A local nonlinear map can separate those datasets before they
collide under pooling.

Identity initialization makes the architectural comparison nested. Within an
experiment, both arms use the same simulation bank, validation bank, minibatch
stream, shared readout architecture, optimizer settings, and checkpoint rule.
Exact reference quantities remain hidden until both arms are frozen.

### 2.6 Pilot based posterior inference

After Stage 1, the score network and all normalization constants are frozen.
A pilot estimate is computed from the observed data using a tractable
composite likelihood or composite score. The pilot does not use the generating
parameter. The frozen score is then evaluated at this pilot.

A pilot is required because a score is local: its value is interpretable only
together with the parameter location where it was evaluated. The pilot supplies
a coarse data based location, while the learned score supplies the local
direction and magnitude of evidence at that location. Providing both quantities
also removes an ambiguity of score only inference, since the same numerical
score can occur at different parameter locations.

Using the same notation in every experiment, the Stage 2 context is

$$
\boldsymbol c(Y)
=
\left(
\widehat{\boldsymbol\beta}_{\mathrm{pilot}}(Y),
\widehat{\boldsymbol S}_{\mathrm{frozen}}
\{Y,\widehat{\boldsymbol\beta}_{\mathrm{pilot}}(Y)\}
\right).
$$

A neural posterior estimator is trained to approximate the posterior of the
original parameter $\boldsymbol\theta$ conditional on this context. The true
generating parameter is used as the training target, not as an input. Training
and deployment therefore use the same context construction.

Each Stage 2 runner also includes a pilot only reference arm with context

$$
\boldsymbol c_{\mathrm{pilot}}(Y)
=
\widehat{\boldsymbol\beta}_{\mathrm{pilot}}(Y).
$$

This arm measures how much posterior information is already present in the
coarse estimator. It is not part of the primary architectural contrast. The
primary comparison is between Linear and the Nonlinear gate, for which the
pilot, simulated datasets, posterior architecture, optimizer, training seed,
and posterior sampling seeds are shared. The two score based arms have context
dimension $2d$; the pilot only arm has dimension $d$.

Neural posterior estimation is used instead of converting the score into a
Gaussian approximation because a finite sample posterior may be skewed,
bounded, or non Gaussian. The mixture density network maps the compact context
to a normalized density and therefore retains uncertainty rather than producing
only a point estimate. Eight mixture components provide moderate flexibility
without an excessive optimization burden at the available simulation budgets.

For all three experiments the posterior estimator is an eight component
mixture density network. Model 1 and Model 2 use the shared posterior helpers
in the selected package and 50,000 Stage 2 simulations. The max stable
experiment uses two hidden layers of width 64 and 10,000 Stage 2 simulations.

The larger one dimensional budget makes posterior comparisons in the mixture
experiments stable across many test datasets. The three dimensional spatial
simulation and pair score construction are substantially more expensive, so a
10,000 simulation budget is used there. Every method within an experiment uses
the same budget; the numerical budgets are not intended to equate computation
across different simulators.

Model 1 and Model 2 use a uniform prior for $\pi$ on $[0.05,0.70]$. The
max stable experiment uses a uniform prior for $\boldsymbol\theta$ on
$[0.15,0.85]^3$. These priors match the Stage 1 reference regions, so Stage 2
does not systematically query a score field outside its training support.
Uniform priors also give balanced coverage of the controlled design regions
without adding informative prior structure that could mask differences between
score representations.

## 3. Supplementary direct posterior baseline

The primary baseline is the Linear amortized FSM field defined above. It uses
the same local scores, pooling rule, readout, training objective, and posterior
estimator as the Nonlinear gate, so their contrast isolates the local map. A
second, supplementary baseline is retained for Model 1 and Model 2 only. It
bypasses Stage 1 and maps the complete raw dataset directly to a posterior.

For Model 1, the canonical dataset is flattened from $20\times20$ to 400
coordinates. For Model 2, it is flattened from $40\times40$ to 1,600
coordinates. The resulting vector is supplied directly to the same family of
neural posterior estimators. This baseline does not use a pilot, an FSM score,
an analytic local score, a constructed summary, or the true parameter as an
input.

The direct posterior baseline asks whether the compact pilot and score context
is useful at the available Stage 2 simulation budget. It is deliberately
generic and does not encode block or coordinate permutation symmetry. Its
result therefore compares two complete inference pipelines; it is not a causal
estimate of the local gate effect. A purpose built raw data DeepSets or
Transformer could be stronger, but it would introduce a second structured
architecture and a different computational budget.

The final max stable run surface does not contain a raw data FSM or a direct
raw data posterior baseline. Its primary comparison is restricted to Linear,
the Nonlinear gate, and the shared pilot only reference arm. Historical raw FSM
experiments retained elsewhere in the Model 1 and Model 2 package are
diagnostic provenance and are not part of the methodology reported here.

## 4. Experiment 1: Gaussian mean shift mixture

### 4.1 Data generating process

The complete dataset contains $K=20$ independent blocks, each with $m=20$
coordinates. For block $k$ and coordinate $j$,

$$
B_k\sim\operatorname{Bernoulli}(\pi),
\qquad
\varepsilon_{kj}\sim\mathcal N(0,1),
$$

$$
Y_{kj}=\varepsilon_{kj}+B_k\tau,
\qquad
\tau=0.5.
$$

The scientific parameter is $\theta=\pi$. Stage 1 uses

$$
\beta=\operatorname{logit}(\pi),
\qquad
\beta_0\text{ as the score reference in }\beta\text{ space}.
$$

This model is chosen as the simplest setting in which complete block evidence
is additive but each supplied local score is a bounded nonlinear function of
that evidence. It isolates information loss caused by pooling without adding
pairwise dependence. Twenty blocks provide repeated independent contributions,
while twenty coordinates per block make local pooling nontrivial. The shift
$\tau=0.5$ gives moderate local evidence: one coordinate is not decisive, but
aggregation across a block is informative.

Reference points follow a continuous stratified design on
$\pi\in[0.05,0.70]$. Continuous coverage is needed because Stage 2 pilots
are not restricted to a small reference grid. Stratification avoids large gaps
in the learned field while retaining randomization within each stratum.

### 4.2 Local and block likelihood ratios

The log likelihood ratio between the active and inactive distributions for one
coordinate is

$$
\ell_{kj}
=
\tau Y_{kj}-\frac{1}{2}\tau^2.
$$

Conditional independence within a block gives

$$
L_k
=
\sum_{j=1}^{m}\ell_{kj}.
$$

This additive identity makes the information target transparent. If local
scores could be converted back to $\ell_{kj}$ before pooling, summation would
recover $L_k$. The experiment can therefore separate failure of local
calibration from failure of the block representation.

For a generic log likelihood ratio $\ell$, define the common mixture score
link

$$
h_{\beta}(\ell)
=
\operatorname{sigmoid}(\beta+\ell)
-
\operatorname{sigmoid}(\beta).
$$

To see why this link appears, write a local mixture density as

$$
p(y\mid \beta)
=
(1-\pi)f_0(y)+\pi f_1(y),
\qquad
\ell(y)=\log\frac{f_1(y)}{f_0(y)}.
$$

The conditional probability of the active component is

$$
\Pr(B=1\mid y,\beta)
=
\operatorname{sigmoid}\{\beta+\ell(y)\}.
$$

Differentiating the mixture log likelihood with respect to $\beta$ gives

$$
\frac{\partial}{\partial \beta}\log p(y\mid \beta)
=
\Pr(B=1\mid y,\beta)-\pi
=
h_{\beta}\{\ell(y)\}.
$$

Thus $h_{\beta}$ is not an artificial neural feature. It is the exact local
likelihood score in the unconstrained parameter coordinate.

The exact likelihood score with respect to $\beta$ is

$$
S_{\mathrm{exact}}(Y,\beta)
=
\sum_{k=1}^{K}h_{\beta}(L_k).
$$

The Stage 1 network is not given $L_k$ or $S_{\mathrm{exact}}$. It receives
the local marginal scores

$$
s_{kj}(\beta_0)
=
h_{\beta_0}(\ell_{kj}).
$$

Local scores are used instead of raw log likelihood ratios to match the intended
composite score workflow and the spatial application, where local scores are
available but a useful analytic inverse is not generally known. Withholding
$L_k$ also prevents the toy model from bypassing the pooling question by
inserting its exact sufficient statistic.

### 4.3 Linear and Nonlinear gate implementations

Let

$$
z_{kj}(\beta_0)
=
\frac{s_{kj}(\beta_0)-\mu_s}{\sigma_s}
$$

denote the standardized local score. The Linear arm averages $z_{kj}$ within
each block. The Nonlinear gate arm, stored under the internal key `radial`, applies

$$
z'_{kj}
=
z_{kj}
m_{\eta}(z_{kj},\overline{\beta}_0)
$$

before averaging. The multiplier MLP has two hidden layers of width 16. The
shared readout has two SiLU hidden layers of width 64. Both functions are
shared across coordinates and blocks.

After the within block mean is formed, it receives a second affine
standardization whose constants are fitted on the Linear training features and
then frozen. The standardized block feature and
$\overline{\beta}_0$ enter the shared readout. The Nonlinear gate reuses the
same frozen block constants, which keeps the downstream coordinates and
readout architecture matched between arms.

Standardization keeps the gate input, pooled block feature, and reference
coordinate on comparable numerical scales. Model 1 applies the gate in this
standardized coordinate because this is the locked implementation used for all
reported seeds. Sharing the gate and readout expresses the exchangeability in
the simulator and prevents coordinate position from becoming an unintended
source of information.

The nonlinear motivation follows from the inverse of the mixture score link.
Since $h_{\beta_0}$ is strictly increasing and
$h_{\beta_0}(0)=0$, a raw local score and
its log likelihood ratio have the same sign. Moreover,

$$
h_{\beta_0}^{-1}(s)
=
\operatorname{logit}
\{\operatorname{sigmoid}(\beta_0)+s\}
-\beta_0,
$$

and therefore

$$
h_{\beta_0}^{-1}(s)
=
s\,m^{*}(s,\beta_0),
\qquad
m^{*}(s,\beta_0)
=
\begin{cases}
\displaystyle
\frac{h_{\beta_0}^{-1}(s)}{s}, & s\ne0,\\[6pt]
\displaystyle
\frac{1}{\pi_0(1-\pi_0)}, & s=0,
\end{cases}
\qquad
\pi_0=\operatorname{sigmoid}(\beta_0).
$$

The value at zero is the continuous limit. Strict monotonicity of
$h_{\beta_0}$ and $h_{\beta_0}(0)=0$ imply
$m^{*}(s,\beta_0)>0$ throughout the admissible local score interval.

This identity explains why a positive multiplier is a meaningful local
calibration. The Model 1 implementation applies its gate to the standardized
coordinate $z$, not directly to $s$. The inverse formula is therefore a
motivation for the function class rather than a hard coded recovery map. The
gate is learned only from the FSM target.

The inverse is not inserted even though it is available in this toy model.
Hard coding it would use special knowledge that does not transfer to the
spatial experiment and would change the question from learning a useful local
calibration to evaluating a known formula.

### 4.4 Pilot

The Model 1 pilot uses the marginal composite score

$$
S_{\mathrm{pilot}}(Y,\beta)
=
\sum_{k=1}^{K}
\frac{1}{m}
\sum_{j=1}^{m}h_{\beta}(\ell_{kj}).
$$

The score is integrated over the constrained parameter grid, and the global
maximizer of the resulting composite potential defines
$\widehat{\beta}_{\mathrm{pilot}}(Y)$. Multiplication by the positive constant
$m$ would not alter the roots or the maximizer.

Only marginal components are used because they are cheap, tractable, and
available under the same restrictions as the Stage 1 representation. The
global potential maximizer is preferred to the first numerical root because a
finite sample composite score may contain several roots or have a boundary
optimum. The pilot is intentionally coarse; the frozen learned score is
supplied to Stage 2 to refine the posterior information.

## 5. Experiment 2: Gaussian common factor mixture

### 5.1 Data generating process

The complete dataset contains $K=40$ independent blocks, each with $m=40$
coordinates. For block $k$,

$$
B_k\sim\operatorname{Bernoulli}(\pi),
\qquad
Z_k\sim\mathcal N(0,1),
\qquad
\varepsilon_{kj}\sim\mathcal N(0,1),
$$

$$
Y_{kj}=\varepsilon_{kj}+B_k\tau Z_k,
\qquad
\tau=1.
$$

When $B_k=0$, the block covariance is $\boldsymbol I_m$. When $B_k=1$,
the covariance is

$$
\boldsymbol I_m
+
\tau^2\boldsymbol 1_m\boldsymbol 1_m^{\mathsf T}.
$$

Again, $\theta=\pi$, $\beta=\operatorname{logit}(\pi)$, and
$\beta_0$ denotes a score reference in $\beta$ space.

This experiment is chosen to separate mean information from dependence
information. The active component has zero mean, so evidence for activity is
carried by a shared covariance fluctuation rather than a directional shift.
Univariate marginals detect variance inflation, but they cannot by themselves
recover all cross products generated by the common factor. Pairwise components
are therefore genuinely necessary.

Forty independent blocks provide repeated score contributions. A block size of
forty creates 780 unordered pairs and makes compression within a block much more
demanding than in Model 1. The choice $\tau=1$ gives visible dependence
without making every block state nearly deterministic. Reference points use the same
continuous stratified range $\pi\in[0.05,0.70]$, allowing the two mixture
experiments to share the same interpretation of $\beta$ and $\beta_0$.

### 5.2 Marginal, pairwise, and block likelihood ratios

For a subset of size $r$, define

$$
c_r
=
-\frac{1}{2}\log(1+r\tau^2),
\qquad
d_r
=
\frac{\tau^2}{2(1+r\tau^2)}.
$$

The univariate marginal log likelihood ratio is

$$
\ell_{ki}^{(1)}
=
c_1+d_1Y_{ki}^2.
$$

For an unordered coordinate pair,

$$
\ell_{k,ij}^{(2)}
=
c_2+d_2(Y_{ki}+Y_{kj})^2.
$$

These two component types are used because they are the lowest order tractable
terms that expose the required structure. Marginal terms provide squared
coordinates. Pairwise terms add the cross products needed to reconstruct the
squared block sum. Higher order components would increase computational cost
without being necessary for this controlled mechanism.

Define the block sum

$$
T_k
=
\sum_{i=1}^{m}Y_{ki}.
$$

The complete block log likelihood ratio is

$$
L_k
=
c_m+d_mT_k^2,
$$

and the exact complete likelihood score is

$$
S_{\mathrm{exact}}(Y,\beta)
=
\sum_{k=1}^{K}h_{\beta}(L_k).
$$

This exact score is used only after training for evaluation.

### 5.3 Information carried by the two local channels

Let

$$
P=\binom{m}{2},
\qquad
A_{1k}=\sum_i\ell_{ki}^{(1)},
\qquad
A_{2k}=\sum_{i<j}\ell_{k,ij}^{(2)},
$$

and let

$$
Q_k=\sum_iY_{ki}^2.
$$

Then

$$
A_{1k}=mc_1+d_1Q_k,
$$

$$
A_{2k}
=
Pc_2+d_2\{(m-2)Q_k+T_k^2\}.
$$

It follows that

$$
Q_k
=
\frac{A_{1k}-mc_1}{d_1},
$$

$$
T_k^2
=
\frac{A_{2k}-Pc_2}{d_2}
-(m-2)
\frac{A_{1k}-mc_1}{d_1}.
$$

Thus the marginal and pairwise log likelihood ratios jointly contain the
statistic required to construct $L_k$. The network is deliberately not given
$A_{1k}$, $A_{2k}$, or this analytic recovery formula. Its local tokens are
the bounded mixture scores

$$
s_{ki}^{(1)}(\beta_0)
=
h_{\beta_0}\{\ell_{ki}^{(1)}\},
\qquad
s_{k,ij}^{(2)}(\beta_0)
=
h_{\beta_0}\{\ell_{k,ij}^{(2)}\}.
$$

Averaging these bounded nonlinear transforms need not preserve the sums of the
underlying log likelihood ratios. This is the information bottleneck addressed
by the local gate.

The recovery formula is shown to establish that the local likelihood ratios
contain enough information in principle. It is not implemented because doing
so would hard code the answer in the tractable benchmark. The experiment instead
tests whether a learned local calibration can retain the relevant information
using the same type of input available in less tractable models.

### 5.4 Linear and Nonlinear gate implementations

Each channel has its own frozen mean and standard deviation. Write

$$
z_c
=
\frac{s_c-\mu_c}{\sigma_c},
\qquad
c\in\{1,2\}.
$$

The Linear arm averages $z_1$ and $z_2$ separately within each block. The
two means remain ordered features and are passed with the reference coordinate
to the shared readout.

The ordered two dimensional block vector receives a second channelwise affine
standardization fitted on the Linear training representation and then frozen.
Both arms use these same constants before the shared readout. This second
normalization changes numerical conditioning but does not merge or exchange the
two channels.

Separate standardization prevents the 780 pair scores from dominating the 40
marginal scores through count or scale alone. Keeping the pooled channels
ordered also allows the readout to use their different statistical roles.

The Nonlinear gate arm, stored under the internal key `shared_radial`,
reconstructs the raw local score $s_c$ and applies one
common scalar multiplier to both channels,

$$
s'_c
=
s_c\,m_{\eta}(s_c,\overline{\beta}_0),
$$

and returns to the standardized coordinate through

$$
z'_c
=
z_c
+
\frac{s_c\{m_{\eta}(s_c,\overline{\beta}_0)-1\}}{\sigma_c}.
$$

This correction form is exactly equal to
$(s'_c-\mu_c)/\sigma_c$, while preserving exact identity nesting when the
multiplier equals one. The gate is shared because marginal and pairwise scores
pass through the same link $h_{\beta_0}$. The two channels retain separate
normalization and separate pooled features, so sharing the gate does not merge
the channels.

The raw score coordinate is used because both channels are produced by the
same nonlinear link $h_{\beta_0}$. A given raw score therefore has the same
inverse link meaning in either channel, whereas a standardized value has a
different raw meaning under each channel normalization. One shared gate
encodes this common structure and uses fewer parameters than two unrelated
gates.

Unlike Model 1, this raw score implementation can represent the analytic
inverse link directly on compact score regions. The analytic inverse is still
not inserted into the network. It is learned, if useful, through the FSM loss.

### 5.5 Pilot

The Model 2 pilot uses equal weighting of the two channels:

$$
S_{\mathrm{pilot}}(Y,\beta)
=
\sum_{k=1}^{K}
\left[
\frac{1}{m}\sum_i h_{\beta}\{\ell_{ki}^{(1)}\}
+
\frac{1}{P}\sum_{i<j}h_{\beta}\{\ell_{k,ij}^{(2)}\}
\right].
$$

Separate means prevent the pairwise channel from dominating only because it
contains more tokens. The constrained global maximizer of the integrated
composite score potential defines $\widehat{\beta}_{\mathrm{pilot}}(Y)$.

The pilot does not use the analytic recovery of $L_k$. This keeps it a genuine
composite estimator and avoids giving Stage 2 an exact likelihood calculation
that would be unavailable in the motivating applications. Equal channel
weighting is a simple fixed rule that prevents the much larger pair count from
deciding the pilot scale by itself.

## 6. Experiment 3: Smith max stable rainfall model

### 6.1 Data and scientific parameter

One complete dataset contains 47 independent annual maxima fields observed at
79 Swiss rainfall sites. Dependence is described by a Smith covariance matrix

$$
\boldsymbol\Sigma
=
\begin{pmatrix}
\Sigma_{11} & \Sigma_{12}\\
\Sigma_{12} & \Sigma_{22}
\end{pmatrix}.
$$

Years are treated as blocks because the model regards annual maxima fields as
independent replicates. Sites are not treated as independent blocks because
their extremal dependence is the scientific object being estimated. The
dimensions 47 and 79 reproduce the rainfall data design rather than being
selected to favor either neural architecture.

Positive definiteness is enforced through two positive scales and one
correlation parameter:

$$
\Sigma_{11}=\sigma_x^2,
\qquad
\Sigma_{12}=\varrho\sigma_x\sigma_y,
\qquad
\Sigma_{22}=\sigma_y^2.
$$

This parameterization is used because arbitrary values of the three covariance
entries need not define a valid covariance matrix. Positive scales and a
correlation in $(-1,1)$ guarantee positive definiteness by construction. They
also separate overall dependence ranges in two spatial directions from
orientation through the correlation term.

Inference is indexed by a bounded normalized parameter

$$
\boldsymbol\theta
=(\theta_1,\theta_2,\theta_3)
\in[0.15,0.85]^3.
$$

Let the reference covariance be

$$
(\Sigma_{11,0},\Sigma_{12,0},\Sigma_{22,0})
=(332.1527,70.3982,184.6266),
$$

with reference scales $\sigma_{x,0}$, $\sigma_{y,0}$, and reference
correlation $\varrho_0$. The normalized parameter maps to the covariance
through

$$
\sigma_x(\boldsymbol\theta)
=
\sigma_{x,0}
\exp\{0.70(\theta_1-0.5)\},
$$

$$
\sigma_y(\boldsymbol\theta)
=
\sigma_{y,0}
\exp\{0.70(\theta_2-0.5)\},
$$

$$
\varrho(\boldsymbol\theta)
=
\tanh
\left[
\operatorname{arctanh}(\varrho_0)
+
1.40(\theta_3-0.5)
\right].
$$

The center $(0.5,0.5,0.5)$ maps exactly to the reference covariance. Seven
additional marginal simulator parameters are held fixed.

The normalized cube provides a finite scientific design region around a
rainfall based reference rather than allowing implausibly small or large
covariances. Centering the reference at 0.5 makes lower and higher simulation
truths interpretable as symmetric departures from that reference. Exponential
maps keep both scales positive, while the hyperbolic tangent keeps the
correlation valid. Holding the marginal parameters fixed isolates the question
of spatial extremal dependence; varying them simultaneously would confound
dependence recovery with marginal tail estimation.

The coefficients 0.70 and 1.40 set a moderate, symmetric range around the
reference on log scale and transformed correlation scale. They are large enough
to produce distinguishable dependence regimes while avoiding covariance values
that are numerically or scientifically extreme over the normalized cube.

The Smith model is used because the bivariate density is available while the
complete 79 site density is not practically available. It therefore provides a
scientifically meaningful setting for testing whether nonlinear processing of
composite scores helps beyond analytically soluble mixture models.

### 6.2 Unconstrained coordinate and FSM proposal

Each bounded coordinate is mapped to

$$
\beta_r
=
\operatorname{logit}
\left(
\frac{\theta_r-0.15}{0.70}
\right),
\qquad
\theta_r
=
0.15+0.70\operatorname{sigmoid}(\beta_r).
$$

The same general notation now applies: $\boldsymbol\beta$ is the
unconstrained coordinate, $\boldsymbol\beta_0$ is the score reference, and
$\widetilde{\boldsymbol\beta}$ is the proposal draw. The proposal is

$$
\widetilde{\boldsymbol\beta}\mid\boldsymbol\beta_0
\sim
\mathcal N
(\boldsymbol\beta_0,0.20^2\boldsymbol I_3).
$$

The shifted and rescaled logit maps the interior of the normalized cube to all
of $\mathbb R^3$. This avoids truncated Gaussian proposals and allows Newton
updates in an unconstrained coordinate, while the inverse map always returns a
valid normalized parameter. An isotropic proposal treats the three transformed
directions symmetrically and avoids inserting a preferred covariance direction
into FSM. The common standard deviation 0.20 is large enough to produce useful
target variation but remains local relative to the reference design.

Reference points are spread through the three dimensional normalized cube by a Latin
hypercube design before transformation. This gives balanced marginal coverage
without requiring a prohibitively large Cartesian grid, and it reduces empty
regions that would otherwise force the amortized field to extrapolate.

### 6.3 Bivariate score tokens

For year $t$ and unordered station pair $(i,j)$, the local token is the
exact bivariate Smith score

$$
\boldsymbol s_{t,ij}(\boldsymbol\beta_0)
=
\left.
\nabla_{\boldsymbol\beta}
\log f_{ij}(Y_{ti},Y_{tj}\mid\boldsymbol\beta)
\right|_{\boldsymbol\beta=\boldsymbol\beta_0}
\in\mathbb R^3.
$$

Each annual field supplies all 3,081 unordered station pair scores. The
complete 79 site likelihood and its score are unavailable and are never used.

Bivariate scores are chosen because they are the richest likelihood components
that are available exactly for every station pair. Using all pairs in Stage 1
retains the available spatial information and avoids making the gate comparison
depend on a random feature subset. Treating each pair as a local token also
preserves the location where nonlinear calibration must occur: before spatial
aggregation.

### 6.4 Linear and Nonlinear gate implementations

Pair scores are first represented in a frozen coordinate specific to their
distance bin. If pair $(i,j)$ belongs to bin $b$, write

$$
\boldsymbol z_{t,ij}
=
\frac{
\boldsymbol s_{t,ij}-\boldsymbol\mu_b
}{
\boldsymbol\sigma_b
}.
$$

The Linear arm averages these standardized pair scores without an additional
local transformation. For the shared gate input, define one global training
standardization of the raw score,

$$
\overline{\boldsymbol s}_{t,ij}
=
\frac{
\boldsymbol s_{t,ij}-\boldsymbol\mu_{\mathrm{gate}}
}{
\boldsymbol\sigma_{\mathrm{gate}}
}.
$$

The Nonlinear gate arm, stored under the internal key `positive_anchor`,
reconstructs the raw score and applies

$$
\boldsymbol s'_{t,ij}
=
\boldsymbol s_{t,ij}
\odot
\boldsymbol m_{\eta}
(\overline{\boldsymbol s}_{t,ij},\overline{\boldsymbol\beta}_0).
$$

It then returns to the bin coordinate through the exact correction

$$
\boldsymbol z'_{t,ij}
=
\boldsymbol z_{t,ij}
+
\frac{
\boldsymbol s_{t,ij}
\odot
\{\boldsymbol m_{\eta}-\boldsymbol 1\}
}{
\boldsymbol\sigma_b
}.
$$

This equals
$(\boldsymbol s'_{t,ij}-\boldsymbol\mu_b)/\boldsymbol\sigma_b$ but reduces
exactly to the Linear input when the multiplier is one. Applying the multiplier
in raw score units gives one common numerical meaning to the gate across bins;
applying it directly to bin standardized values would make an identical gate
input correspond to different raw evidence in different bins.

The gate receives the complete globally standardized three dimensional score
and the standardized three dimensional reference. Its input dimension is
therefore six. It has two SiLU hidden layers of width 16 and three softplus
outputs. Every multiplier may depend jointly on all three score coordinates and
all three reference coordinates, although the final multiplication is
coordinatewise. Pair distance, pair direction, and other geometry coordinates
are not gate inputs.

Distance is nevertheless retained in the representation through fixed pooling
membership. It determines which bin receives a pair after the local map. This
separation gives the gate one common score calibration across the spatial
design, while the ordered bin summary preserves coarse information about
spatial scale for the annual readout. The selected design uses one angle group,
so no directional binning is applied during Stage 1.

The 3,081 pairs are assigned to 20 fixed distance bins. Within every year and
bin, transformed pair scores are averaged. This produces 60 values per year.
The pooled values receive a second frozen affine standardization fitted on the
training bank. This prevents high variance bins or score coordinates from
dominating the annual readout through scale alone. Because both affine
standardizations are frozen and shared by Linear and the Nonlinear gate, they alter
optimization coordinates but not the information available to one arm.
The 60 values and the three standardized reference coordinates enter a shared
annual MLP:

```text
63 inputs → 64 → 64 → three annual score coordinates
```

The 47 annual score contributions are summed. Linear and the Nonlinear gate use the same
pair scores, fixed distance bin assignments, normalization, annual readout,
initialization of common layers, simulation bank, and optimization schedule.

Twenty distance bins provide a compromise between spatial resolution and a
stable fixed dimensional summary. Equal counts avoid bins with very different
Monte Carlo variability. Averaging within a bin prevents densely represented
distance ranges from dominating through pair count alone. A shared annual MLP
is appropriate because years are modeled as identically distributed replicates,
and summing annual contributions matches additivity of scores across independent
years.

### 6.5 Rough composite pilot and Stage 2 context

The common pilot is a weighted pairwise maximum composite likelihood estimate.
It uses a fixed stratified subset of 25 pairs from each distance bin, giving
500 pairs in total. A deterministic Newton method uses 15 starting points,
backtracking, projection to the bounded parameter region, and a global
objective comparison. These safeguards reduce failures due to poor starting
values or invalid Newton steps.

The reduced 500 pair design is used because a pilot only needs a reliable coarse
location, whereas evaluating and differentiating all 3,081 pairs repeatedly for
every Stage 2 simulation is expensive. Stratified selection preserves coverage
of the full distance range. Multiple starts address nonconcavity, backtracking
rejects steps that reduce the objective, projection preserves the allowed
parameter region, and final objective comparison prevents the result from
depending on the ordering of starting values.

The resulting context is

$$
\left(
\widehat{\boldsymbol\beta}_{\mathrm{pilot}}(Y),
\widehat{\boldsymbol S}_{\mathrm{frozen}}
\{Y,\widehat{\boldsymbol\beta}_{\mathrm{pilot}}(Y)\}
\right)
\in\mathbb R^6.
$$

The current Stage 2 runner trains three posterior estimators on the shared
simulation bank. The pilot only arm receives the three dimensional context
$\widehat{\boldsymbol\beta}_{\mathrm{pilot}}(Y)$. The Linear and Nonlinear
gate arms each receive the six dimensional pilot and score context above. The
two score based arms form the primary matched comparison; the pilot only arm
quantifies the information added by the frozen score field.

The bounded normalized parameter $\boldsymbol\theta$ is the NPE target. A
maximum composite likelihood estimate using all 3,081 pairs is retained only
as an evaluation comparator.

Keeping the complete pair estimate out of the Stage 2 context prevents the
baseline pilot from absorbing the computational work that the learned score is
intended to replace. It remains useful after training for measuring the cost of
the rough pilot approximation.

## 7. Training and comparison protocol

### 7.1 Stage 1 settings

Model 1 uses 40,000 training datasets, 8,000 validation datasets, proposal
standard deviation 0.20, batch size 512, and 20,000 updates. Model 2 uses
40,000 training datasets, 8,000 validation datasets, proposal standard
deviation 0.15, batch size 512, and 20,000 updates. The max stable experiment
uses 10,000 training datasets, 8,000 validation datasets, proposal standard
deviation 0.20 in each coordinate, batch size 128, and 10,000 updates.

The two mixture experiments use the same Stage 1 simulation count so their
controlled results are not driven by different data budgets. The max stable
budget is smaller because constructing 3,081 bivariate score vectors for every
annual field is substantially more expensive. Its smaller batch size reflects
the same pairwise memory cost. These budgets are matched between methods within
each experiment, which is the relevant condition for Linear versus Nonlinear gate;
they are not intended to make raw computational cost identical across the three
different simulators.

The proposal widths are chosen to avoid two opposite failures. A width that is
too small produces an extremely noisy target because target variance grows as
proposal variance shrinks. A width that is too large learns an excessively
smoothed field. Model 2 uses 0.15 because its larger number of blocks provides
more stable dataset evidence and the controlled diagnostics showed a clear rise
in smoothing error at wider proposals. The reported widths are fixed before
exact test evaluation.

All three experiments use learning rate $10^{-4}$, weight decay
$10^{-3}$, gradient clipping at norm 5, gate width 16, readout width 64, and
exponential moving average decay 0.995. Model 1 uses a cosine tail learning
rate schedule. Model 2 uses a constant schedule.

The gate width 16 keeps the local calibration small relative to the shared
readout and reduces the chance that a result is driven only by a large increase
in capacity. Width 64 gives the readout enough flexibility to map pooled
evidence to a block score in every arm. Weight decay limits unstable parameter
growth, gradient clipping protects the noisy FSM objective from occasional
large updates, and exponential moving averaging reduces checkpoint noise.
Learning rate schedules follow the locked protocol for each experiment and are
identical between Linear and the Nonlinear gate within that experiment.

For Model 1 and Model 2, the primary checkpoint for each arm minimizes FSM
validation loss. The exact likelihood score is not consulted. The max stable
experiment uses the exponential moving average checkpoint from the
prespecified final update as its primary checkpoint.

Validation selection is used for Model 1 and Model 2 because their trajectories
do not attain their best FSM validation loss at exactly the same update. Each
arm is allowed to select its own checkpoint using the same independent FSM
criterion. The exact score remains hidden, so this does not select on the
reported test outcome. The max stable protocol instead fixes the final
exponential moving average checkpoint in advance. This avoids an additional
selection layer in the current single seed confirmatory run. The two checkpoint
rules are reported separately and are not used for direct numerical comparison
between experiments.

### 7.2 Repetitions and paired comparisons

Model 1 and Model 2 use five independent Stage 1 training seeds. Within each
seed, Linear and the Nonlinear gate share training data, validation data, minibatch
indices, and evaluation datasets. The same NPE training seed is reused across
the five Stage 1 repetitions to isolate variation caused by Stage 1. Current
uncertainty intervals therefore quantify variation across Stage 1 seeds but do
not include independent NPE training variation.

The Stage 1 seed is the inferential repetition because it changes the simulated
training bank, parameter initialization, and optimization path. Pairing methods
within a seed removes unnecessary Monte Carlo variation and targets the
difference caused by the local map. Reusing one NPE seed further isolates the
Stage 1 effect, but it also means that the reported uncertainty is not a full
end to end uncertainty analysis.

The selected max stable experiment currently uses one Stage 1 training seed.
Its Linear and Nonlinear gate evaluations use paired test datasets. This result is a
conditional demonstration rather than a claim of stability across training
seeds.

One seed is used at this stage because the complete pair score cache and three
dimensional FSM training are much more expensive than the mixture experiments.
The paired dataset design still supports a conditional comparison for the
frozen models, but additional training seeds are required before making a
general replication claim.

## 8. Evaluation

### 8.1 Score accuracy in Model 1 and Model 2

After checkpoint selection, the frozen score field is evaluated against the
exact complete likelihood score on independent test datasets. Raw score MSE is
reported at each fixed generating value of $\pi$. Standardized MSE is

$$
\operatorname{stdMSE}(\pi)
=
\frac{
\mathbb E
\left[
\{\widehat S(Y,\beta_0)-S_{\mathrm{exact}}(Y,\beta_0)\}^2
\right]
}{
\operatorname{Var}
\{S_{\mathrm{exact}}(Y,\beta_0)\}
}.
$$

Within a fixed value of $\pi$, all methods use the same denominator. Raw MSE
therefore gives the direct paired comparison at that parameter value.
Standardized MSE is useful when averaging across parameter values whose exact
scores have different scales.

Raw MSE answers the direct approximation question in the natural score units
at one fixed truth. It should not be averaged naively across truths because the
variance of the exact score changes with $\pi$. Dividing by that variance
produces a dimensionless error and prevents parameter regions with intrinsically
larger scores from dominating the aggregate. Both forms are retained because
standardization improves comparability but hides the original error scale.

### 8.2 Posterior accuracy in Model 1 and Model 2

Exact likelihood grids are available for both mixture experiments. Evaluation
therefore includes posterior mean squared error relative to the generating
parameter, posterior mean error relative to the exact posterior mean, the
Wasserstein distance of order one relative to the exact posterior, absolute
error in posterior standard deviation, and empirical coverage of nominal 90
percent credible intervals.

These metrics answer different questions. Error relative to the generating
parameter measures point estimation performance but includes irreducible sample
variation. Error relative to the exact posterior mean more directly measures
posterior approximation. Wasserstein distance evaluates the complete marginal
posterior rather than only its mean. Standard deviation error measures posterior
width, and interval coverage checks calibration. No single metric is sufficient
because a method can estimate the mean well while producing a posterior that is
too narrow or too wide.

The exact posterior is used only for evaluation. It is not available to Stage
1, the pilot, or NPE training.

### 8.3 Evaluation in the max stable experiment

An exact 79 site posterior is unavailable. Synthetic evaluation instead uses
known generating parameters and a scientifically interpretable dependence
function. At each of three fixed normalized truths, 100 paired datasets are
generated. Posterior mean MSE is reported separately for
$\Sigma_{11}$, $\Sigma_{12}$, and $\Sigma_{22}$ on the original
covariance scale.

The three truths $(0.25,0.25,0.25)$, $(0.50,0.50,0.50)$, and
$(0.75,0.75,0.75)$ probe the lower region, the reference center, and the
upper region of the design. Reporting each covariance entry separately avoids
hiding a difficult parameter behind a pooled three dimensional error. The
original covariance scale is used because it has direct scientific meaning,
whereas normalized coordinates are primarily a computational device.

The lower and upper truths remain inside the prior rather than on its boundary,
so poor performance cannot be attributed only to truncation at the support
edge. One hundred datasets at each truth provide paired Monte Carlo comparisons
while keeping the expensive spatial simulation and posterior sampling workload
feasible.

For displacement $\boldsymbol h$, the Smith extremal coefficient is

$$
\delta_{\boldsymbol\Sigma}(\boldsymbol h)
=
2\Phi
\left[
\frac{1}{2}
\sqrt{
\boldsymbol h^{\mathsf T}
\boldsymbol\Sigma^{-1}
\boldsymbol h
}
\right].
$$

Its integrated squared error is evaluated on a fixed grid containing 12 radii
and 8 directions. Paired bootstrap intervals use the difference between
Linear and the Nonlinear gate on each dataset. These quantities measure recovery of the
generating covariance and its implied extremal dependence. They are not
distances to an unavailable exact posterior.

The extremal coefficient is included because covariance entry error does not
directly show how parameter error changes joint tail dependence. Evaluating
several radii and directions checks both spatial range and anisotropy. The grid
is fixed before comparison to prevent favorable locations from being selected
after results are observed. Paired bootstrap intervals preserve the common test
datasets and therefore estimate the method difference with less simulation
noise than two unpaired intervals.

## 9. Interpretation

The primary estimand is the effect of inserting a positive local calibration
before a fixed pooling operation. The identity initialization, common data,
common readout, and common checkpoint rule make Linear and the Nonlinear gate a nested
architectural comparison.

For Model 1 and Model 2, the monotone mixture score link provides an analytic
reason why local nonlinear calibration can be useful. Model 2 provides the
closest match to the inverse link argument because its shared gate acts in raw
score units. Model 1 implements the same positive multiplier principle in
standardized score coordinates.

The max stable model has no corresponding analytic inverse link theorem. In
that experiment the positive multiplier is a prespecified structural bias. Its
value must therefore be established empirically. A favorable result shows that
this restricted local calibration improves the chosen composite score
representation under the fixed protocol. It does not prove that the gate is
universally optimal or that an exact posterior has been recovered.
