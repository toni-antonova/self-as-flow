import csv
from pathlib import Path

import numpy as np


CONDITIONS = {
    "Larger motor changes": (1., 4.),
    "Larger object changes": (4., 1.),
    "Equal uncertainty": (2., 2.),
    "Both priors loosened": (2., 0.5),
}
MEASURES = {
    "Transfer to B": "transfer_to_b",
    "Object-specific response": "object_specific_response",
    "Total response on A": "total_response_on_a",
    "Response on B at zero command": "b_at_zero_command",
}


def exact_observer(condition):
    motor_precision, object_precision = CONDITIONS[condition]
    motor_variance = 1 / motor_precision
    object_variance = 1 / object_precision
    total_variance = 1 + motor_variance + object_variance
    return {
        "Transfer to B": motor_variance / total_variance,
        "Object-specific response": object_variance / total_variance,
        "Total response on A": (motor_variance + object_variance) / total_variance,
    }


def summarize(condition, comparison, measure, values, reference=None):
    generator = np.random.default_rng(20261124)
    draws = generator.integers(0, 30, size=(5000, 30))
    lower, upper = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
    digits = 5 if (condition, comparison, measure) == ("Larger motor changes", "", "Object-specific response") else 3
    return {
        "condition": condition,
        "comparison": comparison,
        "measure": measure,
        "mean": f"{values.mean():.{digits}f}",
        "lower_95": f"{lower:.{digits}f}",
        "upper_95": f"{upper:.{digits}f}",
        "exact_observer": "" if reference is None else f"{reference:.3f}",
    }


def run(data_dir, output_dir):
    with (Path(data_dir) / "portability_by_seed.csv").open(newline="") as source:
        measurements = list(csv.DictReader(source))
    expected = {(condition, seed) for condition in CONDITIONS for seed in range(30)}
    keys = [(row["condition"], int(row["seed"])) for row in measurements]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Expected exactly one measurement for each of 30 seeds in four conditions.")

    seeds = dict(zip(keys, measurements))

    def scores(condition, measure):
        column = MEASURES[measure]
        return np.array([float(seeds[condition, seed][column]) for seed in range(30)])

    rows = []
    for condition in CONDITIONS:
        for measure, reference in exact_observer(condition).items():
            rows.append(summarize(condition, "", measure, scores(condition, measure), reference))

    motor = "Larger motor changes"
    objects = "Larger object changes"
    loosened = "Both priors loosened"
    zero_command = "Response on B at zero command"
    rows.append(summarize(motor, "", zero_command, scores(motor, zero_command)))

    for first, second, measure in [
        (motor, objects, "Transfer to B"),
        (objects, motor, "Object-specific response"),
        (loosened, objects, "Transfer to B"),
        (loosened, objects, "Object-specific response"),
    ]:
        difference = scores(first, measure) - scores(second, measure)
        reference = None
        if first == loosened and measure == "Transfer to B":
            reference = exact_observer(first)[measure] - exact_observer(second)[measure]
        rows.append(summarize(first, second, measure, difference, reference))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "portability.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
