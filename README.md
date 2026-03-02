  # Compositional Algorithmic Generalisation

  Please find the code for our L65 mini-project report, in which we study whether a GNN can learn to generalise compositionally across algorithmic variants by conditioning on execution traces, using the [CLRS benchmark](https://github.com/google-deepmind/clrs) as a foundation.                                  

  ## Structure

  | Directory | Description |
  |-----------|-------------|
  | `phase0/` | Variant definition, parameterised relaxation, curation |
  | `phase1/` | Data pipeline & sampling (conditioned trajectories) |
  | `phase2/` | Model, training loop, and experiment runner |

  ## Setup

  To get set up, please first install the necessary requirements: 

  ```bash
  pip install -r requirements.txt (on GPU please also run pip install "jax[cuda12]")
  ```

  Then, to reproduce the experiments in the paper:

  ```bash
  python -m phase2.run_experiments
  ```

  This runs all 12 experiments sequentially:
  - EXP 1–6: Group coverage (6 non-overlapping held-out groups, 10K steps each)
  - EXP 7–9: Leave-one-out on the 3 worst-performing variants
  - EXP 10: OOD graph-size generalisation (train n∈[8,20], eval up to n=64)
  - EXP 11–12: Diagnostics (k-sweep, wrong-conditioning ablation)
