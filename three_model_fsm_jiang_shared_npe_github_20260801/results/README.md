# Results

These are the small, reportable outputs retained from the formal experiments. They are sufficient to audit the tables in `docs/EXPERIMENT.md` without downloading model weights.

- `headline_score_stdmse.csv`: report-level pre-NPE score comparison.
- `headline_posterior_mean_rmse.csv`: report-level post-NPE comparison against exact posterior means.
- `model1/` and `model2/`: five-seed proposed-method score diagnostics, one frozen Jiang R1 diagnostic, and shared-NPE summaries.
- `model3/`: three proposed-method Stage-1 seeds, three shared-NPE summaries, and Jiang diagnostics.

The formal held-out comparisons use exact likelihood/score calculations only as evaluation references. Those quantities are not Stage-1 inputs.
