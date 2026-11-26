import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from sensing_model import (
    ENVIRONMENTS, HIDDEN, SEEDS, STEPS, SensingNetwork, episodes, experiment_seed,
    observe_step, precisions,
)


def direct_forecasts(network, paths, imposed=None, window=range(STEPS)):
    count = len(paths["state"])
    memory = torch.zeros(count, HIDDEN)
    direct = torch.ones(count, dtype=torch.bool)
    fractions, predictions = [], []
    for step in range(STEPS):
        fraction = network.self_fraction(memory)
        if imposed is not None and step in window:
            fraction = torch.full_like(fraction, imposed)
        memory, prediction = observe_step(network, memory, fraction, paths, step, direct)
        fractions.append(fraction)
        predictions.append(prediction)
    errors = (torch.stack(predictions, dim=1) - paths["state"][:, 1:]).square().mean(-1)
    return torch.stack(fractions, dim=1), errors


def shared_response(network, environment, fraction, noise):
    count, length, _ = noise.shape
    variance, drift = (torch.tensor(pair) for pair in ENVIRONMENTS[environment])
    precision = precisions(torch.tensor(fraction))
    scales, weights = [], []
    for _ in range(length):
        reading_variance = variance + 1 / precision
        gain = variance / reading_variance
        scales.append(reading_variance.sqrt())
        weights.append(gain * reading_variance.sqrt())
        variance = (1 - gain) * variance + drift

    # Match the ideal observer's predicted shared reading at zero after either history.
    weights = torch.stack(weights)
    noise = noise - (noise * weights).sum((1, 2), keepdim=True) * weights / weights.square().sum()
    memory, mean = torch.zeros(count, HIDDEN), torch.zeros(count, 2)
    zero = torch.zeros(count)
    direct = torch.ones(count, dtype=torch.bool)
    allocation = torch.full((count,), fraction)
    for step in range(length):
        reading = mean + noise[:, step] * scales[step]
        memory, _ = network.observe(memory, zero, reading[:, 0], reading[:, 1], zero, direct, allocation)
        mean = mean + noise[:, step] * weights[step]
    if mean.sum(-1).abs().max() > 1e-5:
        raise ValueError("Shared-probe histories have unequal predicted readings.")

    predictions = []
    for reading in (-0.5, 0.5):
        _, estimate = network.observe(memory, torch.ones(count), zero, zero,
                                      torch.full((count,), reading), ~direct, zero)
        predictions.append(estimate)
    return (predictions[1] - predictions[0]).numpy(), variance / (1 + variance.sum())


@torch.no_grad()
def measure_fit(network, environment, paths, noise):
    fractions, free = direct_forecasts(network, paths)
    _, even = direct_forecasts(network, paths, imposed=0.5)
    values = dict(
        early_fraction=fractions[:, :3].mean().item(), late_fraction=fractions[:, -3:].mean().item(),
        free_error=free.mean().item(), even_error=even.mean().item(),
        error_reduction=float((even.numpy() - free.numpy()).mean(axis=1).mean()),
    )

    changes = {}
    for name, window in (("early", range(3)), ("late", range(9, 12))):
        _, high = direct_forecasts(network, paths, imposed=0.85, window=window)
        _, half = direct_forecasts(network, paths, imposed=0.5, window=window)
        block = slice(window.start, window.stop)
        high, half = high[:, block], half[:, block]
        values[name + "_cost"] = (high.mean(1) - half.mean(1)).mean().item()
        changes[name] = (high.numpy() - half.numpy()).mean(axis=1)
    values["late_minus_early_cost"] = float((changes["late"] - changes["early"]).mean())

    high_world, world_reference = shared_response(network, environment, 0.15, noise)
    high_self, self_reference = shared_response(network, environment, 0.85, noise)
    values["shared_shift"] = float(((high_world - high_self) @ np.array([1, -1])).mean())
    values["exact_shared_shift"] = (
        (world_reference[0] - world_reference[1]) - (self_reference[0] - self_reference[1])
    ).item()
    return values


def measure(weights_path, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with np.load(weights_path, allow_pickle=False) as packed:
        weights = {name: torch.from_numpy(packed[name]) for name in packed.files}
    rows = []
    for index, environment in enumerate(ENVIRONMENTS):
        stream = torch.Generator().manual_seed(experiment_seed("eval", "confirmation"))
        paths = episodes(environment, 384, stream)
        noise = torch.randn(384, 6, 2, generator=stream)
        network = SensingNetwork().eval()
        for seed in SEEDS:
            network.load_state_dict({name: tensor[index, seed] for name, tensor in weights.items()})
            rows.append(dict(environment=environment, seed=seed, **measure_fit(network, environment, paths, noise)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the 90 trained sensing networks.")
    parser.add_argument("weights", type=Path, help="Network weights (.npz)")
    parser.add_argument("output", type=Path, help="Destination CSV of one row per fit")
    arguments = parser.parse_args()
    measure(arguments.weights, arguments.output)
