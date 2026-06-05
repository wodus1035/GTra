# Bootstrap-N sweep on simulation data (F1 / GED convergence)

**Question:** how does GTra's recovered cell-state trajectory accuracy depend on
the bootstrap count `N` in `find_gclusters(N)`?

**Setup:** one FIXED simulated **bifurcation** dataset
(`simulation/sim.py::simulate_timeseries_topology`, seed 0, 6 timepoints,
~2800 cells, 430 genes), **labeled** GTra run, **answer-unconstrained**
(`answer_path_dir=""`). Ground-truth answer graph: `Prog->FateA`, `Prog->FateB`
(+ self-loops). For each N we run 5 independent trials (bootstrap is stochastic)
and score the predicted cell-type transition graph vs the answer with
`ged_utils` (transition/overall edge-F1, GED, normalized GED).

Run: `python sim_nsweep.py`  (quick: `--smoke`). Outputs: `nsweep_results.csv`
(per run), `nsweep_summary.csv` (per N), `nsweep_curves.pdf`.

## Result (mean over 5 trials)

| N  | transition F1 | overall F1 | GED | norm-GED |
|----|---------------|-----------|-----|----------|
| 1  | 0.00 ±0.00 | 0.00 | 5.0 | 0.45 |
| 2  | 0.00 ±0.00 | 0.00 | 5.0 | 0.45 |
| 3  | 0.13 ±0.30 | 0.39 | 4.2 | 0.32 |
| 5  | 0.28 ±0.26 | 0.50 | 4.0 | 0.28 |
| 10 | 0.65 ±0.15 | 0.78 | 2.2 | 0.14 |
| 20 | 0.69 ±0.15 | 0.86 | 1.6 | 0.09 |
| 30 | 0.72 ±0.07 | 0.86 | 1.6 | 0.09 |
| 50 | 0.67 ±0.00 | 0.83 | 2.0 | 0.11 |

random transition-F1 baseline = 0.32

## Takeaways
1. **Sharp rise at N≈10, plateau at N≈20–30.** N=50 does not beat N=20–30, so
   the manuscript's N (~20) is adequate/generous — independently consistent with
   the edge-bootstrap convergence result (network converges by ~20–30 iters).
2. **N≤2 fails entirely** (no significant transitions -> 0 edges -> GED=5):
   the per-edge Mann-Whitney/FDR test needs a distribution to call significance.
3. **Variance shrinks with N** (trans-F1 std 0.30 -> 0.07 -> 0.00): higher N =
   more reproducible recovery — quantitative support for the stability claim.
4. **Plateau ~0.70 (not 1.0)** is a real, interpretable ceiling: a persistent
   false `FateA->FateB` edge because the two fates share the bifurcation
   transition gene program — not a bootstrap artifact.

Addresses reviewer R4.4 (add a simulation) with a ground-truth-validated
F1/GED + bootstrap-convergence analysis.
