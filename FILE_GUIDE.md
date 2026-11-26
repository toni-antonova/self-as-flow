# File and data guide

`reproduce.py` reads the processed observations in `data/` and writes the estimates to `results/`.

The summary calculations use NumPy. Repeating the tracking simulation uses Numba, and training or replaying the networks uses PyTorch.

All three experiments use simulated quantities in arbitrary units. A seed is an integer identifier for a random stream. In result columns, `low` and `high`, or `lower_95` and `upper_95`, denote the lower and upper limits of a pointwise 95% percentile interval. Limits have the units of their estimate.

## Tracking a target

`code/tracking_simulation.py` generates the gain, offset, targets, and observation noise. It selects settings on 200 discovery seeds, then runs five controllers on 400 confirmation seeds per regime. `code/tracking_results.py` uses a common set of 5,000 resamples of the paired confirmation seeds to produce Table 1, Table S2.1, and the quantities in S2.4.

| File | Fields and units |
| --- | --- |
| `data/tracking_discovery.csv` | One row per candidate setting and regime. `regime` names the drift schedule. `controller` names the filter or bank. `fixed_ratio` and `fixed_level` give the self-to-world prior precision ratio and overall precision level of a fixed filter, or the selected reference settings for a bank. `forgetting_factor` is the fraction of the previous log score retained by a bank. Inapplicable settings are blank. `runs` is the number of discovery runs. `mean_cost` is their mean squared position error. `guarded_steps` counts scored steps on which the denominator was floored or the command was clipped. |
| `data/tracking_runs.csv` | One row per `seed`, `regime`, and `controller`. `cost` is mean squared position error over steps 200–3,999. `gain_fast_ratio` and `offset_fast_ratio` give the ratio-only bank’s mean pre-command precision ratio in the corresponding phases of alternating drift. Each step uses the weighted geometric mean of candidate ratios. The phase summaries omit burn-in and the first 50 steps in each block. Phase fields are blank for the other conditions. |
| `results/tracking_performance.csv` | `regime` and `controller` identify the condition. `cost_times_1000` is 1,000 times the mean squared position error, with limits `cost_lower_95` and `cost_upper_95`. `gap_closed_percent` is the percentage of the fixed-to-ideal gap recovered, with limits `gap_lower_95` and `gap_upper_95`. The two reference values have empty interval fields. `winning_bank_reduction_percent_of_fixed_cost` gives the percentage reduction in the entire fixed-controller cost for the winning bank in each regime. |
| `results/tracking_precision_ratios.csv` | `faster_drift` names the faster-changing state. `mean_precision_ratio`, `lower_95`, and `upper_95` give the dimensionless ratio and its interval. |

## Cross-object forecasts

`code/train_portability.py` generates episodes, trains the prediction network, and probes a frozen network on 512 fixed histories. `code/portability.py` calculates the Gaussian reference values and resamples the 30 training seeds within each condition. For a contrast it takes the paired difference before resampling.

| File | Fields and units |
| --- | --- |
| `data/portability_by_seed.csv` | `condition` identifies the precision pair and `seed` identifies the fit. `transfer_to_b` is the change in B’s forecast at command 1. `object_specific_response` is the change on A minus that on B at command 0. `total_response_on_a` is the change on A at command 1. `b_at_zero_command` is B’s response at command 0, reported for larger motor changes. Each response is a forecast change per unit of signed error, averaged over evaluation histories. |
| `data/portability_weights.npz` | `conditions` and `seeds` label the first two axes of every parameter array. `recurrent.weight_ih_l0` and `recurrent.weight_hh_l0` map the input and previous hidden state to the GRU gates. `recurrent.bias_ih_l0` and `recurrent.bias_hh_l0` are the corresponding biases. `readout.0.weight` and `readout.0.bias` are the hidden prediction layer. `readout.2.weight` and `readout.2.bias` are the scalar output layer. Within a fit, weight axes are output × input. |
| `results/portability.csv` | `condition` is the first condition. A populated `comparison` field means the second condition was subtracted from the first. `measure` names the response. `mean`, `lower_95`, and `upper_95` give the estimate and interval per unit error. `exact_observer` gives the corresponding analytical value where reported. |

## Allocation of sensing

`code/sensing_model.py` defines the three environments in Table S3.2 and trains the network. `code/sensing_measurements.py` evaluates the original weights on the held-out histories. `code/sensing_results.py` resamples the 30 confirmation fits per environment to calculate the results in S3.2.

The labels `novice_self` and `novice_world` identify which state starts with greater uncertainty. The other state has the larger per-step drift variance throughout the episode. `balanced` gives both states the same initial uncertainty and drift. Early is steps 1–3 and late is steps 10–12.

| File | Fields and units |
| --- | --- |
| `data/sensing_measurements.csv` | One row per `environment` and `seed`. `early_fraction` and `late_fraction` are the freely chosen fractions of adjustable precision assigned to self-sensing. `free_error` and `even_error` are mean squared prediction error with free or even allocation over all 12 steps. `error_reduction` is the paired reduction from even to free allocation. `early_cost` and `late_cost` are the additional mean squared errors from imposing 85% instead of 50% self-sensing in the named window. `late_minus_early_cost` is the paired difference in these costs. `shared_shift` is the change in self-minus-world correction per unit shared error after world-emphasizing versus self-emphasizing readings. `exact_shared_shift` is the Gaussian observer’s value. |
| `data/sensing_weights.npz` | Each array starts with environment (initially self-uncertain, initially world-uncertain, balanced), then seed (0–29). `cell.weight_ih` and `cell.weight_hh` map the input and previous hidden state to the GRU gates, with biases `cell.bias_ih` and `cell.bias_hh`. `gate.weight` and `gate.bias` control the self-sensing fraction. `readout.0.weight` and `readout.0.bias` form the hidden prediction layer. `readout.2.weight` and `readout.2.bias` produce the self and world forecasts. Weight axes within each model are output × input, and biases have one entry per output. |
| `results/sensing_allocation.csv` | One row per `environment`. `early_fraction`, `late_fraction`, `free_error`, `even_error`, and `error_reduction` are means of the per-fit quantities above. `reduction_percent` expresses the reduction relative to the mean even-allocation error. Fraction and paired reduction estimates have `_low` and `_high` limits. |
| `results/sensing_forced.csv` | `environment`, `early_cost`, `late_cost`, and `late_minus_early_cost` give the three paired estimates in squared state units. Each estimate has `_low` and `_high` limits. |
| `results/sensing_shared_error.csv` | `environment`, `shared_shift`, its `_low` and `_high` limits, and `exact_shared_shift` give the learned and Gaussian responses per unit of signed error. |

## Rebuild the observations

Run the commands below from the repository root. They write to a separate directory. The simulations regenerate the training episodes and held-out histories from seeds. The two network experiments can also start from their included confirmation weights.

```sh
mkdir -p rerun/data
python code/tracking_simulation.py discovery rerun/data/tracking_discovery.csv
python code/tracking_simulation.py confirmation rerun/data/tracking_runs.csv --discovery rerun/data/tracking_discovery.csv
python code/train_portability.py --weights data/portability_weights.npz --output rerun/data/portability_by_seed.csv
python code/sensing_measurements.py data/sensing_weights.npz rerun/data/sensing_measurements.csv
python reproduce.py --data rerun/data --output rerun/results
```

To retrain both network cohorts from their initial seeds, first run the two tracking commands above. Replace the two checkpoint evaluation commands with the three commands below, then run the same `reproduce.py` command. `--workers` sets the number of independent fits run at once.

```sh
python code/train_portability.py --workers 3 --output rerun/data/portability_by_seed.csv
python code/sensing_model.py rerun/data/sensing_weights.npz --workers 3
python code/sensing_measurements.py rerun/data/sensing_weights.npz rerun/data/sensing_measurements.csv
```
