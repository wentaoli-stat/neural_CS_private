# Checkpoint layout

Checkpoint binaries are deliberately excluded from Git. The runners expect this layout:

```text
artifacts/checkpoints/
├── ours/
│   ├── model1/seed20260709/ ... seed20260713/
│   ├── model2/seed20260709/ ... seed20260713/
│   └── model3/seed20260721/ ... seed20260723/
└── jiang/
    ├── model1/models/jiang_round1.pt
    ├── model2/models/jiang_round1.pt
    └── model3/                     # if rerunning the Model 3 Jiang arm
```

Each Model 1/2 `ours` seed directory must contain:

```text
config.json
feature_stats.npz
model_linear.pt
model_radial.pt
```

Each Model 3 `ours` seed directory uses `models.pt` plus its corresponding retained JSON configuration. Set `FSM_CHECKPOINT_ROOT` to override `artifacts/checkpoints/ours`, and pass `--jiang-dir` to override the Jiang directory.
