import argparse
import csv
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn

from portability import CONDITIONS


# These labels are part of the original random seeds for the training episodes.
STREAM_NAMES = {
    "Larger motor changes": "self_volatile",
    "Larger object changes": "world_volatile",
    "Equal uncertainty": "balanced",
    "Both priors loosened": "global_relaxation",
}
QUERY_COMMANDS = torch.tensor([-2., -1., 0., 1., 2.])
SIGNED_ERRORS = torch.tensor([-1., -0.5, 0., 0.5, 1.])


def seed_number(*parts):
    label = ":".join(map(str, ("roster20261124", "portability", *parts)))
    digest = hashlib.sha256(label.encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


class ObjectPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = nn.GRU(4, 64, batch_first=True)
        self.readout = nn.Sequential(nn.Linear(67, 64), nn.Tanh(), nn.Linear(64, 1))

    def forecast(self, hidden, commands, objects):
        history = hidden[:, None, :].expand(-1, commands.shape[1], -1)
        identity = nn.functional.one_hot(objects, 2).to(hidden)
        query = torch.cat((history, commands[:, :, None], identity), dim=-1)
        return self.readout(query).squeeze(-1)


def object_sequence(size, generator):
    first = torch.randint(2, (size, 1), generator=generator)
    calibration = (first + torch.arange(6)) % 2
    after_change = torch.randint(2, (size, 4), generator=generator)
    return torch.cat((calibration, after_change), dim=1)


def observations(commands, readings, objects):
    identity = nn.functional.one_hot(objects, 2).to(commands)
    return torch.cat((commands[:, :, None], readings[:, :, None], identity), dim=-1)


def training_batch(condition, generator, size=64):
    motor_precision, object_precision = CONDITIONS[condition]
    objects = object_sequence(size, generator)
    gain_change = torch.randn(size, 1, generator=generator) / np.sqrt(motor_precision)
    offsets = torch.randn(size, 2, generator=generator) / np.sqrt(object_precision)
    commands = torch.rand(size, 10, generator=generator) * 4 - 2
    readings = commands + torch.randn(size, 10, generator=generator)
    readings[:, 6:] += commands[:, 6:] * gain_change + offsets.gather(1, objects[:, 6:])

    future_commands = torch.rand(size, 4, generator=generator) * 4 - 2
    future_objects = torch.randint(2, (size, 4), generator=generator)
    future_signal = (1 + gain_change) * future_commands + offsets.gather(1, future_objects)
    targets = future_signal + torch.randn(size, 4, generator=generator)
    return observations(commands, readings, objects), future_commands, future_objects, targets


def prediction_loss(model, batch):
    inputs, commands, objects, targets = batch
    history, _ = model.recurrent(inputs)
    after_one = model.forecast(history[:, 6], commands, objects)
    after_four = model.forecast(history[:, 9], commands, objects)
    return ((after_one - targets).square().mean() + (after_four - targets).square().mean()) / 2


def train(condition, seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed_number("initial", seed))
    generator = torch.Generator().manual_seed(seed_number("train", STREAM_NAMES[condition], seed))
    model = ObjectPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for _ in range(5000):
        optimizer.zero_grad(set_to_none=True)
        loss = prediction_loss(model, training_batch(condition, generator))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    return model.eval()


@torch.no_grad()
def probe(model):
    size = 512
    generator = torch.Generator().manual_seed(seed_number("eval", "confirmation"))
    objects = object_sequence(size, generator)[:, :7]
    objects[:, -1] = torch.arange(size) % 2
    commands = torch.rand(size, 7, generator=generator) * 4 - 2
    commands[:, -1] = 1
    readings = commands + torch.randn(size, 7, generator=generator)
    query_commands = QUERY_COMMANDS.expand(size, -1)
    object_a = objects[:, -1]
    forecasts = []
    for error in SIGNED_ERRORS:
        readings[:, -1] = 1 + error
        _, hidden = model.recurrent(observations(commands, readings, objects))
        sequence = []
        for query_object in (object_a, 1 - object_a):
            query_ids = query_object[:, None].expand_as(query_commands)
            sequence.append(model.forecast(hidden[-1], query_commands, query_ids))
        forecasts.append(torch.stack(sequence, dim=1))
    return torch.stack(forecasts, dim=1)


def seed_measurements(condition, seed, predictions):
    # Slopes of forecasts on signed error, keeping each calibration history fixed.
    weighted = predictions * SIGNED_ERRORS[None, :, None, None]
    slopes = weighted.sum(dim=1) / SIGNED_ERRORS.square().sum()
    other_at_zero = slopes[:, 1, 2]
    return {
        "condition": condition,
        "seed": seed,
        "transfer_to_b": float(slopes[:, 1, 3].mean()),
        "object_specific_response": float((slopes[:, 0, 2] - other_at_zero).mean()),
        "total_response_on_a": float(slopes[:, 0, 3].mean()),
        "b_at_zero_command": float(other_at_zero.mean()) if condition == "Larger motor changes" else "",
    }


def fit(job):
    condition, seed = job
    return seed_measurements(condition, seed, probe(train(condition, seed)))


def checkpoint_measurements(weights):
    torch.set_num_threads(1)
    model = ObjectPredictor().eval()
    with np.load(weights, allow_pickle=False) as archive:
        names = tuple(model.state_dict())
        tensors = {name: archive[name] for name in names}
        for index, condition in enumerate(archive["conditions"].tolist()):
            for seed in archive["seeds"].tolist():
                model.load_state_dict({name: torch.from_numpy(tensors[name][index, seed]) for name in names})
                yield seed_measurements(condition, seed, probe(model))


def write_measurements(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / "measurements.csv"
        with staged.open("w", newline="") as destination:
            first = next(rows)
            writer = csv.DictWriter(destination, fieldnames=list(first))
            writer.writeheader()
            writer.writerow(first)
            writer.writerows(rows)
        staged.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or replay the 120 two-object networks.")
    parser.add_argument("--output", type=Path, required=True, help="Path for portability_by_seed.csv")
    parser.add_argument("--weights", type=Path, help="Replay the fitted networks in portability_weights.npz")
    parser.add_argument("--workers", type=int, default=1, help="Number of independent training workers")
    args = parser.parse_args()
    if args.weights:
        write_measurements(checkpoint_measurements(args.weights), args.output)
    else:
        jobs = [(condition, seed) for condition in CONDITIONS for seed in range(30)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            write_measurements(pool.map(fit, jobs), args.output)
