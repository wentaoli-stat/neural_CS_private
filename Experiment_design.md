Example:
* Model 1:
	- param dimension
		-- 1-dim
		-- 3-dim
	- param values

Tuning params: 
* Score matching: 
	- bandwith: (Tuning methods given in 'bandwidth_tuning.md')
		-- narrow 
		-- wide 
		-- a grid for explaining the tuning effect
* NLSA: 
	- learning rate of inner map: 
		-- try two different choices 
	- stacked map:
		-- dimension of MLP

* Score-only summary:
	- evaluation point
		-- truth
		-- MLE
		-- cheap pilot
		-- one-step correction
	- Training bandwidth: 
		-- narrow which is score optimized 
		-- wide which is posterior optimized

* Pilot only summary:
	- pilot choice
		-- MLE
		-- cheap pilot
		-- one-step correction
* Pair:
	- pilot/evaluation point
		-- MLE
		-- cheap pilot
		-- one-step correction
	- Training bandwidth:
		-- narrow one optimizing the Fisher score approximation
		-- wide one 


Fisher score approximation:
F1. raw input + generic MLP
F2. all sub-scores + generic MLP
F3. Jiang et al 2026 AISTATS
F4. ILSA (narrow/wide bandwith)
F5. NLSA + gate map (narrow/wide bandwith + two lr)
F6. NLSA + stacked map (narrow/wide bandwith + two lr + 3 different MLP dimensions)


Comparisons: 1 shows the effect of score structure; 2 shows the effect of permutation invariance structure; 3 is SOTA; 4 shows the benefit of inner map; 5 shows the difference with stacked map; 6 shows the tuning of the new method

Posterior inference: (For score-based: score only/pair)
P0. pilot only
P1. raw input
P2. F1 score (raw input + generic MLP)
P3. all sub-scores
P4. F2 score (all sub-scores + generic MLP)
P5. Multiple composite scores-based posterior (Li et al 2026)
P6. F4 score  (narrow/wide bandwith + best score)
P7. F5 score (narrow/wide bandwith  + best score)
P8. F6 score (narrow/wide bandwith  + best score)

P0 is naive method, P1-P4 are standard benchmarks. P5 is surrogate score-based posterior without compression. P6 is asymptotically equivalent to P5. P7 and P8 are proposed method.  



