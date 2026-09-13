#!/usr/bin/env python3
"""Does co-activating an internal-state circuit change a sensorimotor curve?

    python run_sweep.py --drive sugar --rates 25,50,75,100,150 --readout MN9 --co "NPFP1+NPFL1-I:100"
    python run_sweep.py --drive sugar --readout MN9 --co re:^PAM:50 --co re:^PPL1:50

Runs the dose-response of --drive on --readout, alone and with each --co group
firing at a fixed rate throughout, and reports the shift. Writes
results/sweeps/<name>.csv and <name>.png.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from flybrain import Anatomy, Connectome, LIFNetwork, Names  # noqa: E402
from flybrain.logging_setup import configure  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK2, GRID, PAPER = "#0b0b0b", "#52514e", "#dddcd8", "#fcfcfb"
LABEL_SEPARATION_HZ = 6.0     # direct-label a series only if its endpoint is this clear of the others


@dataclass
class Condition:
    label: str
    co_drive: dict[tuple[int, ...], float]     # neuron indices -> constant rate (Hz)


def parse_conditions(co_specs: list[str], names: Names) -> list[Condition]:
    """
    Build the condition list: drive alone, then one per --co spec.

    :param co_specs: "NAME:HZ" strings.
    :param names: Resolver for neuron group names.
    :return: Conditions.
    """
    conditions = [Condition("alone", {})]
    for spec in co_specs:
        name, hz_text = spec.rsplit(":", 1)
        hz = float(hz_text)
        indices = tuple(names.resolve(name).tolist())
        conditions.append(Condition(f"+ {name} {hz:.0f} Hz", {indices: hz}))
    return conditions


def run_conditions(connectome: Connectome, drive: tuple[int, ...], readout: int, rates: list[float],
                   conditions: list[Condition], t_run: float, n_trials: int, seed: int) -> pd.DataFrame:
    """
    Run every condition at every drive rate.

    :param connectome: Loaded connectome.
    :param drive: Model indices of the driven group.
    :param readout: Model index of the readout neuron.
    :param rates: Drive rates in Hz.
    :param conditions: Conditions to run.
    :param t_run: Trial duration in seconds.
    :param n_trials: Repeats per point.
    :param seed: Base random seed.
    :return: One row per (condition, rate).
    """
    rows = []
    for condition_number, condition in enumerate(conditions):
        for rate in rates:
            network = LIFNetwork(connectome, seed=seed + condition_number * 100 + int(rate))
            result = network.run({drive: rate, **condition.co_drive}, t_run=t_run, n_trials=n_trials)
            readout_per_trial = result.counts[:, readout] / t_run
            n_active = int((result.rates > 0).sum())
            rows.append({"condition": condition.label, "drive_hz": rate,
                         "readout_hz": readout_per_trial.mean(), "sd": readout_per_trial.std(),
                         "n_active": n_active})
            logger.info("{:<28s} {:5.0f} Hz -> {:6.1f} ± {:4.1f} Hz   ({} neurons active)",
                        condition.label, rate, readout_per_trial.mean(), readout_per_trial.std(), n_active)
    return pd.DataFrame(rows)


def log_shifts(table: pd.DataFrame, conditions: list[Condition]) -> None:
    """
    Log each condition's mean shift relative to the drive alone.

    :param table: Output of ``run_conditions``.
    :param conditions: Conditions run.
    """
    baseline = table[table.condition == "alone"].set_index("drive_hz").readout_hz
    for condition in conditions[1:]:
        curve = table[table.condition == condition.label].set_index("drive_hz").readout_hz
        shift = curve - baseline
        logger.info("shift vs alone  {:<28s} {:+6.1f} Hz mean   (per rate: {})", condition.label,
                    shift.mean(), ", ".join(f"{value:+.0f}" for value in shift.values))


def plot_curves(table: pd.DataFrame, conditions: list[Condition], drive_name: str,
                readout_name: str, out_path: Path) -> None:
    """
    Plot readout rate against drive rate, one series per condition.

    :param table: Output of ``run_conditions``.
    :param conditions: Conditions run.
    :param drive_name: Label for the x axis.
    :param readout_name: Label for the y axis.
    :param out_path: PNG path.
    """
    figure, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)

    endpoints: dict[str, pd.Series] = {}
    for condition_number, condition in enumerate(conditions):
        series = table[table.condition == condition.label]
        ax.errorbar(series.drive_hz, series.readout_hz, yerr=series.sd,
                    color=SERIES_COLOURS[condition_number % len(SERIES_COLOURS)],
                    lw=2, marker="o", ms=5, capsize=3, label=condition.label)
        endpoints[condition.label] = series.iloc[-1]
    # direct labels only where they will not collide (the legend always carries identity)
    for label, endpoint in endpoints.items():
        clear = all(abs(endpoint.readout_hz - other.readout_hz) > LABEL_SEPARATION_HZ
                    for other_label, other in endpoints.items() if other_label != label)
        if clear:
            ax.annotate(label, (endpoint.drive_hz, endpoint.readout_hz), xytext=(5, 0),
                        textcoords="offset points", color=INK2, fontsize=8.5, va="center")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    ax.set_xlabel(f"{drive_name} drive (Hz)", color=INK2)
    ax.set_ylabel(f"{readout_name} rate (Hz)", color=INK2)
    ax.set_title(f"Does co-activation shift the {drive_name} → {readout_name} curve?",
                 color=INK, fontsize=11, loc="left")
    figure.tight_layout()
    figure.savefig(out_path, dpi=160, facecolor=PAPER)
    plt.close(figure)


def default_name(drive: str, readout: str, co_specs: list[str]) -> str:
    """
    Results file stem derived from the drive, readout and co-drives.

    :param drive: Drive spec.
    :param readout: Readout spec.
    :param co_specs: "NAME:HZ" strings.
    :return: Filesystem-safe stem.
    """
    suffix = "".join("_" + spec.split(":")[0].replace("re:^", "").replace("+", "-") for spec in co_specs)
    return f"{drive}_to_{readout}{suffix}"


def build_parser() -> argparse.ArgumentParser:
    """
    Command-line interface.

    :return: Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drive", required=True, metavar="NAME")
    parser.add_argument("--rates", default="25,50,75,100,150", help="comma-separated drive rates (Hz)")
    parser.add_argument("--readout", required=True, metavar="NAME")
    parser.add_argument("--co", action="append", default=[], metavar="NAME:HZ", help="co-activated group, repeatable")
    parser.add_argument("--t-run", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--name", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main() -> None:
    """
    Run a sweep from the command line.
    """
    args = build_parser().parse_args()
    configure(verbose=args.verbose, log_file=PROJECT_ROOT / "results" / "flybrain.log")

    connectome = Connectome()
    anatomy = Anatomy(connectome)
    names = Names(connectome, anatomy)
    drive = tuple(names.resolve(args.drive).tolist())
    readout = int(names.resolve(args.readout)[0])
    rates = [float(text) for text in args.rates.split(",")]
    conditions = parse_conditions(args.co, names)
    logger.info("drive {} ({} neurons), readout {}, {} conditions x {} rates x {} trials",
                args.drive, len(drive), args.readout, len(conditions), len(rates), args.trials)

    table = run_conditions(connectome, drive, readout, rates, conditions, args.t_run, args.trials, args.seed)
    log_shifts(table, conditions)

    out_dir = PROJECT_ROOT / "results" / "sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or default_name(args.drive, args.readout, args.co)
    table.to_csv(out_dir / f"{name}.csv", index=False)
    plot_curves(table, conditions, args.drive, args.readout, out_dir / f"{name}.png")
    logger.success("wrote {} and {}", out_dir / f"{name}.csv", out_dir / f"{name}.png")


if __name__ == "__main__":
    main()
