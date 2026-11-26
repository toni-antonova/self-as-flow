import csv
from pathlib import Path

import numpy as np


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def reproduce(data_dir, output_dir):
    with (Path(data_dir) / "sensing_measurements.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    allocation, forced, shared = [], [], []
    resamples = np.random.default_rng(20261124).integers(0, 30, (5000, 30))
    for environment in ("novice_self", "novice_world", "balanced"):
        group = sorted((r for r in rows if r["environment"] == environment), key=lambda r: int(r["seed"]))
        if [int(row["seed"]) for row in group] != list(range(30)):
            raise ValueError(f"Expected one measurement for every seed from 0 to 29 in {environment}.")

        def column(name):
            return np.array([float(row[name]) for row in group])

        def estimate(name, decimals):
            values = column(name)
            low, high = np.quantile(values[resamples].mean(axis=1), [0.025, 0.975])
            return {key: f"{value:.{decimals}f}" for key, value in (
                (name, values.mean()), (name + "_low", low), (name + "_high", high),
            )}

        free, even = column("free_error").mean(), column("even_error").mean()
        allocation.append(dict(
            environment=environment, **estimate("early_fraction", 3), **estimate("late_fraction", 3),
            free_error=f"{free:.5f}", even_error=f"{even:.5f}",
            reduction_percent=f"{100 * (even - free) / even:.2f}",
            **estimate("error_reduction", 6 if environment == "balanced" else 5),
        ))
        forced.append(dict(
            environment=environment, **estimate("early_cost", 5), **estimate("late_cost", 5),
            **estimate("late_minus_early_cost", 5),
        ))
        shared.append(dict(
            environment=environment, **estimate("shared_shift", 3),
            exact_shared_shift=f'{column("exact_shared_shift").mean():.3f}',
        ))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "sensing_allocation.csv", allocation)
    write_rows(output_dir / "sensing_forced.csv", forced)
    write_rows(output_dir / "sensing_shared_error.csv", shared)
