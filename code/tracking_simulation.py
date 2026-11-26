import argparse
import csv
import math
from itertools import product
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numba import njit

REGIMES = ("Alternating", "Lockstep")
BANKS = ("Reification ratio", "Learning rate", "Both")
RATIOS = 2.0 ** np.arange(-6, 7)
LEVELS = 12.5 * 2.0 ** np.arange(13)
MEMORIES = (0.90, 0.95, 0.98, 0.99)
STEPS = 4_000
BURN_IN = 200
BLOCK = 200
PERSISTENCE = 0.98
NOISE_VARIANCE = 0.0225


class World(NamedTuple):
    target: np.ndarray
    noise: np.ndarray
    gain: np.ndarray
    offset: np.ndarray
    gain_drift: np.ndarray
    offset_drift: np.ndarray


def make_world(seed, regime):
    gain_rng, offset_rng, magnitude_rng, sign_rng, noise_rng = (
        np.random.default_rng([stream, seed]) for stream in range(1, 6)
    )
    even = (np.arange(STEPS) // BLOCK) % 2 == 0
    fast, slow = 4 * 0.0002, 0.0002 / 4
    gain_drift = np.where(even, fast, slow)
    offset_drift = np.where(even, slow, fast) if regime == "Alternating" else gain_drift
    gain_noise = np.sqrt(gain_drift) * gain_rng.standard_normal(STEPS)
    offset_noise = np.sqrt(offset_drift) * offset_rng.standard_normal(STEPS)
    gain = np.ones(STEPS)
    offset = np.zeros(STEPS)
    for step in range(STEPS - 1):
        gain[step + 1] = 1 + PERSISTENCE * (gain[step] - 1) + gain_noise[step]
        offset[step + 1] = PERSISTENCE * offset[step] + offset_noise[step]
    target = magnitude_rng.uniform(0.5, 1.5, STEPS) * sign_rng.choice([-1., 1.], size=STEPS)
    noise = math.sqrt(NOISE_VARIANCE) * noise_rng.standard_normal(STEPS)
    return World(target, noise, gain, offset, gain_drift, offset_drift)


@njit
def take_action(world, step, gain, offset):
    raw = (world.target[step] - offset) / max(gain, 0.2)
    command = min(max(raw, -4.), 4.)
    outcome = command * world.gain[step] + world.offset[step] + world.noise[step]
    return command, outcome, gain < 0.2 or command != raw


@njit
def filter_bank(world, ratios, levels, memory, record_ratios):
    gain_variance = 1 / (levels * np.sqrt(ratios))
    offset_variance = 1 / (levels / np.sqrt(ratios))
    log_ratios = np.log(ratios)
    size = len(ratios)
    gain, offset = np.ones(size), np.zeros(size)
    scores, weights = np.zeros(size), np.full(size, 1. / size)
    loss, guard_count = 0., 0
    phase_sum, phase_count = np.zeros(2), np.zeros(2)

    for step in range(STEPS):
        mean_gain, mean_offset = 0., 0.
        for j in range(size):
            mean_gain += weights[j] * gain[j]
            mean_offset += weights[j] * offset[j]
        command, outcome, guarded = take_action(world, step, mean_gain, mean_offset)
        if step >= BURN_IN:
            loss += (outcome - world.target[step]) ** 2
            guard_count += guarded
            if record_ratios and step % BLOCK >= 50:
                phase = (step // BLOCK) % 2
                weighted_log_ratio = 0.
                for j in range(size):
                    weighted_log_ratio += weights[j] * log_ratios[j]
                phase_sum[phase] += math.exp(weighted_log_ratio)
                phase_count[phase] += 1

        for j in range(size):
            error = outcome - (command * gain[j] + offset[j])
            variance = command * command * gain_variance[j] + offset_variance[j] + NOISE_VARIANCE
            gain[j] += (command * gain_variance[j] / variance) * error
            offset[j] += (offset_variance[j] / variance) * error
            gain[j] = 1 + PERSISTENCE * (gain[j] - 1)
            offset[j] *= PERSISTENCE
            if size > 1:
                scores[j] = memory * scores[j] - 0.5 * (math.log(variance) + error * error / variance)
        if size > 1:
            largest = scores.max()
            total = 0.
            for j in range(size):
                weights[j] = math.exp(scores[j] - largest)
                total += weights[j]
            weights /= total

    if record_ratios:
        phase_sum /= phase_count
    return loss / (STEPS - BURN_IN), guard_count, phase_sum


@njit
def ideal_observer(world):
    gain, offset = 1., 0.
    gain_variance, covariance, offset_variance = 0., 0., 0.
    loss, guard_count = 0., 0
    for step in range(STEPS):
        command, outcome, guarded = take_action(world, step, gain, offset)
        gain_weight = gain_variance * command + covariance
        offset_weight = covariance * command + offset_variance
        variance = command * gain_weight + offset_weight + NOISE_VARIANCE
        error = outcome - (command * gain + offset)
        gain += (gain_weight / variance) * error
        offset += (offset_weight / variance) * error
        gain = 1 + PERSISTENCE * (gain - 1)
        offset *= PERSISTENCE
        gain_variance = PERSISTENCE**2 * (gain_variance - gain_weight**2 / variance) + world.gain_drift[step]
        covariance = PERSISTENCE**2 * (covariance - gain_weight * offset_weight / variance)
        offset_variance = PERSISTENCE**2 * (offset_variance - offset_weight**2 / variance) + world.offset_drift[step]
        if step >= BURN_IN:
            loss += (outcome - world.target[step]) ** 2
            guard_count += guarded
    return loss / (STEPS - BURN_IN), guard_count


def grid(controller, ratio, level):
    ratios = RATIOS if controller in ("Reification ratio", "Both") else [ratio]
    levels = LEVELS if controller in ("Learning rate", "Both") else [level]
    return np.array(list(product(ratios, levels)), dtype=float).T.copy()


def evaluate(world, controller, ratio, level, memory, record_ratios=False):
    if controller == "Ideal observer":
        cost, guards = ideal_observer(world)
        return cost, guards, ()
    ratios, levels = grid(controller, ratio, level)
    return filter_bank(world, ratios, levels, float(memory or 0), record_ratios)


def discovery_row(worlds, regime, controller, ratio="", level="", memory=""):
    measurements = [evaluate(world, controller, ratio, level, memory) for world in worlds]
    return dict(regime=regime, controller=controller, fixed_ratio=ratio, fixed_level=level,
                forgetting_factor=memory, runs=len(worlds),
                mean_cost=math.fsum(value[0] for value in measurements) / len(worlds),
                guarded_steps=sum(value[1] for value in measurements))


def best_fixed(rows):
    return min((row for row in rows if row["controller"] == "Fixed"),
               key=lambda row: (float(row["mean_cost"]), float(row["fixed_ratio"]), float(row["fixed_level"])))


def discover():
    rows = []
    for regime in REGIMES:
        worlds = [make_world(seed, regime) for seed in range(10_000, 10_200)]
        fixed = [discovery_row(worlds, regime, "Fixed", ratio, level)
                 for ratio, level in product(RATIOS, LEVELS)]
        winner = best_fixed(fixed)
        rows.extend(fixed)
        rows.append(discovery_row(worlds, regime, "Ideal observer"))
        for controller, memory in product(BANKS, MEMORIES):
            rows.append(discovery_row(worlds, regime, controller,
                                      winner["fixed_ratio"], winner["fixed_level"], memory))
    return rows


def selected_settings(rows):
    settings = {}
    for regime in REGIMES:
        group = [row for row in rows if row["regime"] == regime]
        winner = best_fixed(group)
        for row in group:
            if row is winner or row["controller"] != "Fixed":
                fraction_guarded = int(row["guarded_steps"]) / (int(row["runs"]) * (STEPS - BURN_IN))
                if fraction_guarded > 0.001:
                    raise ValueError(f"Guard rate exceeded 0.1% for {regime}, {row['controller']}")
        memories = {}
        for controller in BANKS:
            candidates = [row for row in group if row["controller"] == controller]
            best = min(candidates, key=lambda row: (float(row["mean_cost"]), float(row["forgetting_factor"])))
            memories[controller] = float(best["forgetting_factor"])
        settings[regime] = (float(winner["fixed_ratio"]), float(winner["fixed_level"]), memories)
    return settings


def confirm(discovery, seeds=range(400)):
    settings = selected_settings(discovery)
    for seed, regime in product(seeds, REGIMES):
        world = make_world(seed, regime)
        ratio, level, memories = settings[regime]
        for controller in ("Best fixed", *BANKS, "Ideal observer"):
            record_ratios = regime == "Alternating" and controller == "Reification ratio"
            cost, _, phases = evaluate(world, controller, ratio, level, memories.get(controller), record_ratios)
            yield dict(seed=seed, regime=regime, controller=controller, cost=cost,
                       gain_fast_ratio=phases[0] if record_ratios else "",
                       offset_fast_ratio=phases[1] if record_ratios else "")


def write_csv(path, rows):
    rows = iter(rows)
    first = next(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=first, lineterminator="\n")
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repeat tracking parameter selection or confirmation from simulator seeds.")
    parser.add_argument("stage", choices=("discovery", "confirmation"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--discovery", type=Path, default=Path(__file__).resolve().parents[1] / "data/tracking_discovery.csv")
    args = parser.parse_args()
    if args.stage == "discovery":
        write_csv(args.output, discover())
    else:
        with args.discovery.open(newline="") as stream:
            write_csv(args.output, confirm(list(csv.DictReader(stream))))
