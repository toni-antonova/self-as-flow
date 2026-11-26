import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn


# Each pair lists self first, then world.
ENVIRONMENTS = {
    "novice_self": ((2.0, 0.125), (0.005, 0.24)),
    "novice_world": ((0.125, 2.0), (0.24, 0.005)),
    "balanced": ((0.5, 0.5), (0.06, 0.06)),
}
STEPS = 12
HIDDEN = 64
SEEDS = range(30)


def experiment_seed(*parts):
    text = ":".join(map(str, ("roster20261124", "attention", *parts)))
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def precisions(fraction):
    self_precision = 0.05 + 3.9 * fraction
    return torch.stack((self_precision, 4.0 - self_precision), dim=-1)


class SensingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.GRUCell(8, HIDDEN)
        self.readout = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 2))
        self.gate = nn.Linear(HIDDEN, 1)

    def self_fraction(self, memory):
        return self.gate(memory).sigmoid().squeeze(-1)

    def observe(self, memory, command, self_reading, world_reading, shared, direct, fraction):
        available = direct.to(memory.dtype)
        shared_available = 1 - available
        inputs = torch.stack((
            command, self_reading * available, world_reading * available,
            shared * shared_available, available, available, shared_available, fraction * available,
        ), dim=-1)
        memory = self.cell(inputs, memory)
        return memory, self.readout(memory)


def episodes(environment, count, generator):
    initial, drift = (torch.tensor(pair) for pair in ENVIRONMENTS[environment])
    start = torch.randn(count, 1, 2, generator=generator) * initial.sqrt()
    increments = torch.randn(count, STEPS, 2, generator=generator) * drift.sqrt()
    state = torch.cat((start, start + increments.cumsum(1)), dim=1)
    commands = torch.rand(count, STEPS, generator=generator) * 3 - 1.5
    direct = torch.rand(count, STEPS, generator=generator) < 0.8
    sensor_noise = torch.randn(count, STEPS, 2, generator=generator)
    shared_noise = torch.randn(count, STEPS, generator=generator)
    targets = state[:, 1:] + 0.1 * torch.randn(count, STEPS, 2, generator=generator)
    return dict(state=state, commands=commands, direct=direct, sensor_noise=sensor_noise,
                shared_noise=shared_noise, targets=targets)


def observe_step(network, memory, fraction, paths, step, direct):
    state = paths["state"][:, step]
    command = paths["commands"][:, step]
    readings = state + paths["sensor_noise"][:, step] / precisions(fraction).sqrt()
    shared = state[:, 0] * command + state[:, 1] + paths["shared_noise"][:, step]
    return network.observe(memory, command, readings[:, 0], readings[:, 1], shared, direct, fraction)


def prediction_loss(network, paths, generator):
    count = len(paths["state"])
    memory = torch.zeros(count, HIDDEN)
    prefix_length = torch.randint(1, STEPS + 1, (count,), generator=generator)
    prefix_length *= (torch.rand(count, generator=generator) < 0.4).long()
    imposed = torch.tensor([0.15, 0.5, 0.85])[torch.randint(3, (count,), generator=generator)]
    predictions = []
    for step in range(STEPS):
        fraction = torch.where(step < prefix_length, imposed, network.self_fraction(memory))
        memory, estimate = observe_step(network, memory, fraction, paths, step, paths["direct"][:, step])
        predictions.append(estimate)
    return (torch.stack(predictions, dim=1) - paths["targets"]).square().mean()


def train_fit(job):
    environment, seed = job
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(experiment_seed("initial", seed))
    stream = torch.Generator().manual_seed(experiment_seed("train", environment, seed))
    network = SensingNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.002)
    for _ in range(3000):
        optimizer.zero_grad(set_to_none=True)
        loss = prediction_loss(network, episodes(environment, 48, stream), stream)
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    return {name: value.numpy() for name, value in network.state_dict().items()}


def save_weights(output, fits):
    names = tuple(fits[0])
    arrays = {}
    for name in names:
        stacked = np.stack([fit[name] for fit in fits])
        arrays[name] = stacked.reshape(len(ENVIRONMENTS), len(SEEDS), *stacked.shape[1:])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description="Train the 90 sensing networks from their original seeds.")
    parser.add_argument("output", type=Path, help="Destination .npz file for the fitted weights")
    parser.add_argument("--workers", type=int, default=1, help="Number of single-threaded CPU jobs")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    for setting in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[setting] = "1"
    jobs = [(environment, seed) for environment in ENVIRONMENTS for seed in SEEDS]
    with ProcessPoolExecutor(args.workers) as pool:
        save_weights(args.output, list(pool.map(train_fit, jobs)))


if __name__ == "__main__":
    main()
