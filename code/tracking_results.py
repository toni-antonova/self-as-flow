import csv
from pathlib import Path

import numpy as np

REGIMES = ("Alternating", "Lockstep")
CONTROLLERS = ("Best fixed", "Reification ratio", "Learning rate", "Both", "Ideal observer")


def mean_and_draws(values, indices):
    return np.concatenate(([values.mean()], values[indices].mean(axis=1)))


def rounded_interval(values, decimals):
    lower, upper = np.quantile(values[1:], [0.025, 0.975])
    return [f"{number:.{decimals}f}" for number in (values[0], lower, upper)]


def write_results(data_dir, output_dir):
    with (data_dir / "tracking_runs.csv").open(newline="") as stream:
        runs = list(csv.DictReader(stream))
    by_condition = {}
    for row in runs:
        key = (row["regime"], row["controller"])
        by_condition.setdefault(key, []).append(row)
    if set(by_condition) != {(r, c) for r in REGIMES for c in CONTROLLERS}:
        raise ValueError("The tracking data must contain five controllers in each of the two drift regimes.")
    for condition, rows in by_condition.items():
        rows.sort(key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in rows] != list(range(400)):
            raise ValueError(f"Expected 400 paired seeds for {condition}.")

    # Each row resamples the same seeds for all controllers, costs, and ratios.
    rng = np.random.default_rng(20261120)
    indices = rng.integers(0, 400, size=(5_000, 400), dtype=np.int64)
    with (output_dir / "tracking_performance.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["regime", "controller", "cost_times_1000", "cost_lower_95", "cost_upper_95",
                         "gap_closed_percent", "gap_lower_95", "gap_upper_95",
                         "winning_bank_reduction_percent_of_fixed_cost"])
        for regime in REGIMES:
            cost = {}
            for controller in CONTROLLERS:
                values = np.array([float(row["cost"]) for row in by_condition[regime, controller]])
                cost[controller] = mean_and_draws(values, indices)
            fixed, ideal = cost["Best fixed"], cost["Ideal observer"]
            winner = min(("Reification ratio", "Learning rate", "Both"), key=lambda bank: cost[bank][0])
            for controller in CONTROLLERS:
                savings = fixed - cost[controller]
                gap_closed = 100 * savings / (fixed - ideal)
                gap = rounded_interval(gap_closed, 1)
                if controller in ("Best fixed", "Ideal observer"):
                    gap = [f"{gap_closed[0]:.0f}", "", ""]
                reduction = f"{100 * savings[0] / fixed[0]:.2f}" if controller == winner else ""
                writer.writerow([regime, controller, *rounded_interval(1000 * cost[controller], 3), *gap, reduction])

    with (output_dir / "tracking_precision_ratios.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["faster_drift", "mean_precision_ratio", "lower_95", "upper_95"])
        rows = by_condition["Alternating", "Reification ratio"]
        for phase, field in (("Actuator gain", "gain_fast_ratio"), ("Environmental offset", "offset_fast_ratio")):
            values = np.array([float(row[field]) for row in rows])
            writer.writerow([phase, *rounded_interval(mean_and_draws(values, indices), 3)])


if __name__ == "__main__":
    import argparse
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Calculate tracking costs and precision ratios for Tables 1 and S2.1.")
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_results(args.data, args.output)
