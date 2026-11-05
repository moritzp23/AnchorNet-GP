# Low-rank low-precision training of Gaussian processes

Code and results for the paper *Low-rank low precision Training of Gaussian Processes*
by Moritz Poguntke, Eda Oktay, Yuanzhe Xi, Erin Carson and Martin Stoll.

ANGP fits a Gaussian process with an ARD squared-exponential kernel by maximizing the
FITC marginal likelihood of a low-rank model whose landmarks are chosen by AnchorNet.
The objective and its analytic gradients use only low-rank factors, diagonals and small
dense matrices, and the whole fit runs in single precision on one GPU.

## Installation

### The package

To use ANGP in your own code, install the package from GitHub. It needs PyTorch (install
the wheel that matches your CUDA version first); KeOps and scikit-learn are optional:

```bash
pip install git+https://github.com/moritzp23/AnchorNet-GP
pip install "angp[keops,sklearn] @ git+https://github.com/moritzp23/AnchorNet-GP"   # with the optional dependencies
```

This installs only the `angp` package, not the experiment scripts.

### For the paper experiments

Clone the repository and install the versions used for the paper. The experiments were
run with Python 3.11, PyTorch 2.5.1 (CUDA 12.4) and KeOps 2.3 on an NVIDIA A40.

```bash
git clone https://github.com/moritzp23/AnchorNet-GP && cd AnchorNet-GP
conda create -n angp python=3.11
conda activate angp
# a torch wheel that matches `nvcc --version`, for example CUDA 12.4:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

KeOps compiles its kernels at first use with the local CUDA toolkit; the compiled
kernels are cached in `.keops_cache/`.

The data are the nine UCI datasets of Maddox et al. (2022), loaded through
`bayesian_benchmarks`. Download them once into its data directory:

```bash
wget https://tuc.cloud/index.php/s/ZSqcBdqWWJYxTJd/download/uci.tar.gz
tar -xzf uci.tar.gz && rm uci.tar.gz
mv uci "$(python -c 'from bayesian_benchmarks.paths import DATA_PATH; print(DATA_PATH)')"
```

## Usage

### In Python

`ANGPRegressor` follows the scikit-learn interface (`fit`, `predict`, `score`) and
works with its tools, e.g. `cross_val_score` or `GridSearchCV`:

```python
from angp import ANGPRegressor

model = ANGPRegressor(rank=1000)          # number of AnchorNet landmarks
model.fit(X_train, y_train)               # NumPy arrays, X of shape (n, d)
y_pred = model.predict(X_test)
y_pred, y_std = model.predict(X_test, return_std=True)

model.lengthscale_, model.signal_std_, model.noise_std_   # fitted hyperparameters
```

* `rank` sets the accuracy and the cost; the fit cost grows roughly with `rank**2`.
  The rank needed varies between datasets (Table 2 of the paper), so it is worth trying
  a few values.
* `mean='lowrank'` (default) returns the predictive mean of the low-rank model, which
  is cheap. `mean='exact'` returns the full-kernel predictive mean, with a dense solve
  up to 60,000 training points and preconditioned KeOps CG beyond that. In fp32 the CG
  solve does not reach full accuracy on ill-conditioned problems (Section 6).
* `return_std=True` also returns the predictive standard deviation of a new
  observation (`noise=False` gives that of the latent function). For `mean='lowrank'`
  it is the FITC predictive variance; for `mean='exact'` it needs a dense factorization
  and is available up to 60,000 training points. Everything is computed in fp32; on a
  synthetic test the low-rank standard deviation differed from a float64 computation by
  about 1%.
* Inputs and targets are standardized internally with the mean and standard deviation
  of the training data (`normalize=True`). `bayesian_benchmarks` standardizes with the
  statistics of the whole dataset, test points included, before splitting. To
  reproduce the paper, pass its already standardized data with `normalize=False`; this
  gives exactly the results of `experiments/run.py`:

  ```python
  from bayesian_benchmarks.data import get_regression_data
  d = get_regression_data('wilson_bike', split=1)
  model = ANGPRegressor(rank=100, mean='exact', normalize=False).fit(d.X_train, d.Y_train)
  ```
* A CUDA GPU is needed for `mean='exact'` on large data and is much faster in general.

### Paper experiments

One fit, followed by both predictive means, with the settings of the paper:

```bash
python experiments/run.py --dataset bike --rank 100 --split 1 --csv results/new/runs.csv
```

This prints the optimization, the fitted hyperparameters and the test RMSE of the
full-kernel and the low-rank predictive mean, and appends one row to the CSV. The
datasets are `poletele`, `elevator`, `bike`, `kin40k`, `protein`, `3droad`, `song`,
`buzz` and `houseelectric`. `python experiments/run.py -h` lists the options.

`python tests/test_gradients.py` checks on random data that the autograd, unblocked and
blocked implementations of the objective agree and that the gradient matches finite
differences; `python tests/test_variance.py` checks the predictive variances.

## Repository layout

```
angp/
  estimator.py        ANGPRegressor, the scikit-learn style interface
  train.py            the hyperparameter fit (initialization, priors, Adam)
  anchor_net.py       AnchorNet landmark selection
  fitc.py             FITC objective with autograd gradients (reference), safe_cholesky
  fitc_manual.py      FITC objective with analytic gradients, unblocked
  fitc_blocked.py     FITC objective with analytic gradients, blocked over the data
  fitc_mp.py          blocked objective with selected GEMMs in fp16 (Section 7)
  predict.py          full-kernel mean (dense or KeOps CG) and low-rank FITC mean
experiments/
  run.py              one fit and its evaluation
  pm_solve.py         fp32/fp64 CG predictive solve with a per-iteration trace
  mp_eval.py          reduced-precision objective evaluations at fitted hyperparameters
  commands.py         prints the command of every run in the paper
  make_tables.py      Tables 1-6 from results/
  make_figures.py     Figures 2-4 from results/
  mp_summary.py       values of Tables 7-9 and of the modified Cholesky runs from results/mp/
tests/
  test_gradients.py   agreement of the three objective implementations
  test_variance.py    predictive variances against the textbook formulas
results/
  runs.csv            every fit used in Tables 1-6
  fp64_traces.csv     fp64 CG traces (3DRoad, Buzz, HouseElectric; 5 splits each)
  theta/              fitted hyperparameters of the 3DRoad, Buzz and HouseElectric fits
  figures/            fp32 CG traces for Figures 2 and 3
  mp/                 mixed-precision fits on 3DRoad, rank 3000, and evaluations (Section 7)
```

## Reproducing the paper

The tables and figures can be rebuilt from the recorded results:

```bash
python experiments/make_tables.py --out tables/      # Tables 1-6 as LaTeX
python experiments/make_figures.py --out figures/    # Figures 2-4 as PNG
python experiments/mp_summary.py                     # values of Tables 7-9
```

To rerun the experiments, `experiments/commands.py` prints one command per run. Each
command writes to `results/new/`; `make_tables.py --runs ... --fp64 ...` and
`make_figures.py --results ...` read the new files instead of the recorded ones.

| Paper | Commands | Output | Approx. cost (A40) |
|---|---|---|---|
| Tables 1-6, low-rank and fp32 columns | `commands.py fits` | `runs.csv` | 100 GPU-hours |
| Tables 1, 2, 4-6, fp64 columns; Figure 4 | `commands.py fp64` | `fp64_traces.csv` | 1,200 GPU-hours |
| Figures 2 and 3 | `commands.py figs` | `buzz250_trace.csv`, `he_pr_sweep.csv` | 8 GPU-hours |
| Tables 7-9 and modified Cholesky (Section 7) | `commands.py mp` | `mp_fits.csv`, `mp_trajectory_*.csv`, `mp_scan.csv`, `mp_repair_fits.csv`, `mp_repair_scan.csv` | 2-3 GPU-hours |

All configurations are evaluated on the splits 1, 2, 3, 4 and 73 of
`bayesian_benchmarks`. The fp64 solves start from the hyperparameters in
`results/theta/`, so they can be run without repeating the fits. Figure 1 is an
illustration and is not produced here.

### How the recorded results were obtained

* **Fit times** are the median of repeated runs with `--prealloc_frac 0.9` on a GPU that
  no other job shared. `run.py` times the unblocked and the blocked gradient once and
  keeps the unblocked one only if it is at least 20% faster. The unblocked code became
  faster during the experiments, so `commands.py fits` fixes each cell to the gradient
  path its timing runs used. The fit time includes the AnchorNet selection, the
  initialization, the choice between the blocked and unblocked gradients and the 100
  Adam iterations, but not the prediction. `runs.csv` contains the accuracy runs (one per
  split) and these timing runs; the `use_*` columns state which rows enter which table
  column.
* **fp64 times** are the mean over the splits of the time until the test RMSE stays
  within 1e-4 of its final value. A few solves were stopped and resumed from a
  checkpoint; their clock in `fp64_traces.csv` continues across the restart.
* **Mixed precision.** `LRGP_MP_MODE=gemm2` is the two-GEMM variant of the paper and
  `gemm4` the four-GEMM variant. If the fp16 matrix M does not admit a Cholesky
  factorization, that call forms M again in fp32; `run.py` records these calls in the
  column `mp_fallback_calls`. With `LRGP_MP_REPAIR=shift` or `clamp`, an indefinite (but
  finite) M is instead repaired by a modified Cholesky factorization: M + tau I with tau
  increased by factors of 10 from 1e-3, or M with its eigenvalues raised to at least 1
  (the exact M has lambda_min >= 1). These calls are recorded in `mp_repair_calls`.

Timings depend on the GPU and on other jobs on the machine. In our checks, repeating a
fit reproduced the fitted hyperparameters and the RMSE values to about four digits.

## Reference

Maddox, Potapczynski and Wilson, *Low-precision arithmetic for fast Gaussian processes*,
UAI 2022. The numbers in the column `Maddox` of the tables are from their Table 1.

## License

MIT, see `LICENSE`.
