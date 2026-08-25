# Max-stable rainfall-79: selected 10k experiment

This is the canonical Max-stable experiment used with Model 1 and Model 2.
It contains only the selected **Linear versus Nonlinear gate** comparison.
The old Jiang, residual, affine, bounded, GRU, geometry-conditioned, 40k and
full-ten-parameter experiments are not part of this package.

For a guided run, open
[`../RUN_THREE_MODELS.ipynb`](../RUN_THREE_MODELS.ipynb). The shared method and
result report is
[`../model1_model2_clean_20260818/METHOD_AND_RESULTS.md`](../model1_model2_clean_20260818/METHOD_AND_RESULTS.md).

## Statistical problem

One complete observation contains 47 iid annual-maxima fields at 79 Swiss
rainfall sites. The experiment estimates the three Smith dependence
coordinates of

$$
\Sigma=
\begin{pmatrix}
\Sigma_{11} & \Sigma_{12}\\
\Sigma_{12} & \Sigma_{22}
\end{pmatrix},
$$

while the seven marginal/GEV simulator coordinates remain fixed. Computation
uses a normalized parameter $u\in[0.15,0.85]^3$ and the unconstrained FSM
coordinate

$$
w_r=\operatorname{logit}\!\left(\frac{u_r-0.15}{0.70}\right).
$$

Every annual field supplies all $\binom{79}{2}=3{,}081$ exact bivariate
likelihood-score vectors in the three-dimensional $w$ coordinate. The full
79-site likelihood and exact posterior are unavailable.

## Selected Linear/Nonlinear-gate comparison

The displayed method name is **Nonlinear gate**. The code and existing
checkpoints retain the internal key `positive_anchor` for compatibility.

For one pairwise score vector $s$ evaluated at anchor $w_a$:

$$
\text{Linear: }\phi_L(s,w_a)=s,
\qquad
\text{Nonlinear gate: }\phi_N(s,w_a)=s\odot m_\eta(s,w_a),\quad m_\eta>0.
$$

The Nonlinear gate receives the complete three-coordinate score and
three-coordinate standardized anchor. A width-16, two-hidden-layer SiLU MLP
outputs three positive `softplus` multipliers and is initialized exactly at
$m_\eta\equiv1$.

The local map is applied before pooling. Pairs are assigned to 20 fixed
equal-count distance bins; each bin averages its three-dimensional transformed
scores. The ordered $20\times3=60$ bin summary and anchor enter a shared
width-64 annual MLP, and the 47 annual contributions are summed. Pair geometry
is used only for fixed bin membership; it is not an input to the gate.

## Stage 1 and Stage 2

Stage 1 uses

$$
w\mid w_a\sim N(w_a,0.20^2I_3),
\qquad T=(w-w_a)/0.20^2.
$$

Linear and the Nonlinear gate share 10,000 training datasets, 8,000 validation datasets,
anchors, targets, minibatches, optimizer settings and the fixed-final EMA
checkpoint rule.

Stage 2 first computes the same data-only rough MPLE pilot from 500 fixed
stratified pairs (25 per distance bin). Both arms use the six-dimensional
context

```text
(rough_mple_pilot_w[3], frozen_stage1_score_at_pilot[3])
```

and the same 10,000 simulations and 8-component MDN-NPE. The generating
parameter is an NPE target and evaluation truth only. Full all-pair MPLE is an
evaluation comparator, not an input to either NPE.

The selected runner fixes the Stage-2 density estimator to MDN. Flow-only
settings such as transform count and spline-bin count are intentionally not
part of its command-line interface. Stage 1 likewise requires the float32
all-pair cache produced by `prepare_pair_memmap.py` and checks that contract
unconditionally.

## Directory layout

```text
config.json                         locked selected protocol
code/src/maxstable_rainfall79/
  core.py                           parameter maps, scores, networks, pilot
  support.py                        frozen rainfall covariance/simulator adapter
code/upstream/maxstable/            frozen Smith simulator and pair likelihood
code/runners/
  prepare_stage1.py                 simulate anchored FSM train/validation banks
  prepare_pair_memmap.py            compute all 3,081 pair scores and statistics
  train_stage1.py                   matched Linear/nonlinear-gate training
  run_stage2.py                     rough pilot, contexts, MDN-NPE and evaluation
scripts/
  run_all.sh                        complete four-step reproduction
  check_package.py                  selected-artifact integrity check
  summarize_results.py              recompute reported summaries
tests/                              selected protocol and architecture tests
results/                            retained per-dataset results and NPE states
```

There is no Jiang adapter, geometry gate, residual arm, raw-data baseline or
real-data diagnostic in the canonical run surface.

## Installation

Python 3.11 or newer is recommended. Simulation requires R, `SpatialExtremes`
and `rpy2` in addition to the Python dependencies.

```bash
cd maxstable_rainfall79_10k_clean
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[npe,test]'
```

Verify the retained package before a costly rerun:

```bash
shasum -a 256 -c MANIFEST.sha256
python scripts/check_package.py
python scripts/summarize_results.py
pytest -q
```

## Full reproduction

Run the complete pipeline with one command:

```bash
DEVICE=cuda scripts/run_all.sh runs/rainfall79_10k
```

The launcher executes, in order:

1. `prepare_stage1.py`: reproduce the historical 40k source RNG stream and
   the 8k validation bank;
2. `prepare_pair_memmap.py`: retain the first 10k training rows and compute the
   all-pair cache;
3. `train_stage1.py`: train Linear and the Nonlinear gate for 10k optimizer updates;
4. `run_stage2.py`: build rough-pilot score contexts, train the two matched
   NPEs and evaluate 100 prior-wide test datasets.

Replaying the 40k source stream is intentional: only its first 10k rows enter
training, but generating the full source stream preserves the archived
validation RNG state. The all-pair float32 cache is roughly 31 GB for 10k
training plus 8k validation rows. Use fast local storage and sufficient page
cache; this is not a laptop-scale full rerun.

The clean snapshot includes the selected Stage-2 NPE checkpoints and result
tables, but not the historical Stage-1 checkpoint itself. `run_all.sh`
recreates that checkpoint.

`run_stage2.py --reuse-contexts` reuses `contexts.npz` from its output
directory. New caches embed their simulation, pilot, feature-evaluation and
originating Stage-1 checkpoint provenance. On reuse, those saved values—not
the currently supplied context-generation options—are reported in
`protocol.json`, and a known Stage-1 checkpoint digest must match. Compatible
legacy shared caches without embedded metadata remain readable, but their
generation settings and checkpoint match are reported explicitly as unknown.

## Retained results

Prior-wide posterior-mean error from one Stage-1 training seed:

| Method | RMSE $u_1$ | RMSE $u_2$ | RMSE $u_3$ | Joint L2 RMSE |
|---|---:|---:|---:|---:|
| Linear | 0.04657 | 0.04864 | **0.03473** | 0.07577 |
| Nonlinear gate | **0.04403** | **0.04405** | 0.03514 | **0.07151** |

At the low, center and high fixed truths, the Nonlinear gate reduces summed
original-covariance-scale MSE by 31.15%, 9.97% and 18.61%. It reduces the
integrated extremal-coefficient MSE by 16.31%, 13.98% and 18.20%.

Run `python scripts/summarize_results.py` to recompute every retained result
from the packaged per-dataset rows. Detailed tables and limitations are in
[`results/RESULTS.md`](results/RESULTS.md).

The evidence is currently one Stage-1 training seed with 100 paired datasets
per reported fixed-truth cell. It supports the selected experiment but is not
a multi-seed general claim.
