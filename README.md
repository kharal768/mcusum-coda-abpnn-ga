# Reproducible pipeline: diagnosing MCUSUM-CoDa signals with ABPNN-GA

This code produces every number, table and figure in the simulation study (Section 5) of
"A Genetic Algorithm-Optimized Neural Network for Diagnosing Out-of-Control Signals in the
Multivariate CUSUM Control Chart for Compositional Data".

## Files
- `simulate.py` – in-control parameters, MCUSUM (Crosier, 1988) and T² charts, control-limit calibration, signal simulation and network inputs
- `models.py` – BP, ABPNN (momentum + bold-driver learning rate), BP-GA and ABPNN-GA networks, with all settings
- `run.py` – full experiment: calibration, 5 replications, 8 classifiers, input ablation, T² comparison
- `analyze.py` – summary statistics (mean and SD over replications) -> `results/summary.json`
- `make_tables.py` – LaTeX table rows and Figure (`results/accuracy_vs_delta.pdf`)
- `verify_crosier.py` – checks the simulator against Crosier (1988): h = 5.5 gives ARL0 of about 200 in two dimensions
- `results/` – outputs of the run reported in the paper

## Run
```
pip install -r requirements.txt
python run.py --runs 5      # about 10 minutes on one CPU core; resumable, skips completed replications
python analyze.py
python make_tables.py
```
`python run.py --runs 1 --quick` runs a small smoke test.
Delete `results/` first to rerun from scratch. All random numbers come from fixed seeds, so the
accuracies are reproduced exactly; timings will differ between machines.

## Design summary
- ilr coordinates with the pivot basis; three-part (bronze-like) and four-part (lithium-ion-like) compositions. The parameters are simulation settings, not plant data.
- MCUSUM with k = 0.5, m = 1; h calibrated by simulation to ARL0 of about 370 (`results/h_calibration.json`).
- Each run: 20 in-control observations (false alarms discarded), then a sustained shift of delta standard deviations in each coordinate of the pattern, until the chart signals.
- All non-empty shift patterns; delta in {0.25, 0.5, 1, 1.5, 2, 2.5, 3}.
- Per replication and per (pattern, delta) cell: 500 training, 150 validation and 1000 test signals, simulated independently.
- Hidden nodes and input set chosen once on the validation set of replication 1.
