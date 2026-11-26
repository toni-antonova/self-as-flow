# Self as flow

Code and data for *Self as flow: reification as a control parameter in self-modeling systems*, by Antonia Antonova.

| Paper location | Calculation | Results |
| --- | --- | --- |
| §5(a), Table 1 and Supplement S2 | Tracking accuracy when the actuator and environment drift, and the prior precision ratio selected in each phase | [Tracking performance](results/tracking_performance.csv) · [Precision ratios](results/tracking_precision_ratios.csv) |
| §5(b) and Supplement S3.1 | How much a forecast changes on the observed object and on a second object after the same surprise | [Object forecasts and contrasts](results/portability.csv) |
| §5(c) and Supplement S3.2 | Where the network allocates sensing, the cost of imposing extra self-sensing, and the effect of earlier sensing on a later error | [Allocation](results/sensing_allocation.csv) · [Forced sensing](results/sensing_forced.csv) · [Shared error](results/sensing_shared_error.csv) |

## Reproduce the results

From the repository root, use Python 3.13.13 on a CPU.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

This writes six CSVs to `results/`. To save the results elsewhere, specify an output directory.

```sh
python reproduce.py --output rerun/results
```

The [file and data guide](FILE_GUIDE.md) describes each script, dataset, result field, and unit. It also gives commands for rebuilding the data from simulator seeds or trained weights. The [requirements](requirements.txt) give the package versions. The [MIT license](LICENSE) covers the original code.
