#!/usr/bin/env python3
"""Reproduce the core taste -> proboscis-extension results on the FlyWire connectome.

Runs four experiments and writes CSVs + a figure into results/:
  1. dose-response: sugar GRN rate -> MN9 rate
  2. channel comparison: sugar / bitter / water / Ir94e -> MN9
  3. bitter suppression: sugar 100 Hz + bitter at increasing rate -> MN9
  4. the most active neurons under sugar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from flybrain import Connectome, LIFNetwork, circuits  # noqa: E402
from flybrain.logging_setup import configure  # noqa: E402

# dataviz categorical slots 1-4 (light mode), validated for CVD separation
CHANNEL_COLOURS = {"sugar": "#2a78d6", "bitter": "#eb6834", "water": "#1baf7a", "ir94e": "#eda100"}
INK, INK2, GRID, PAPER = "#0b0b0b", "#52514e", "#dddcd8", "#fcfcfb"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
NeuronGroup = tuple[int, ...]


def style_axes(ax: Axes) -> None:
    """
    Light-surface styling for a line-plot axes.

    :param ax: Axes to style.
    """
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def sugar_dose_response(network: LIFNetwork, sugar: NeuronGroup, mn9: int, rates: list[int],
                        t_run: float, n_trials: int) -> pd.DataFrame:
    """
    MN9 rate at each sugar drive rate.

    :param network: Network to run.
    :param sugar: Model indices of the sugar GRNs.
    :param mn9: Model index of MN9.
    :param rates: Drive rates in Hz.
    :param t_run: Trial duration in seconds.
    :param n_trials: Repeats per point.
    :return: Columns grn_hz, mn9_hz, mn9_std.
    """
    rows = []
    for rate in rates:
        result = network.run({sugar: float(rate)}, t_run=t_run, n_trials=n_trials)
        rows.append({"grn_hz": rate, "mn9_hz": result.rate(mn9), "mn9_std": result.rates_std[mn9]})
        logger.info("sugar {:3d} Hz -> MN9 {:5.1f} Hz", rate, result.rate(mn9))
    return pd.DataFrame(rows)


def channel_comparison(network: LIFNetwork, channels: dict[str, NeuronGroup], mn9: int,
                       rates: list[int], t_run: float, n_trials: int) -> pd.DataFrame:
    """
    MN9 rate for each taste channel at each drive rate.

    :param network: Network to run.
    :param channels: Channel name -> model indices.
    :param mn9: Model index of MN9.
    :param rates: Drive rates in Hz.
    :param t_run: Trial duration in seconds.
    :param n_trials: Repeats per point.
    :return: Columns channel, grn_hz, mn9_hz.
    """
    rows = []
    for channel, indices in channels.items():
        for rate in rates:
            result = network.run({indices: float(rate)}, t_run=t_run, n_trials=n_trials)
            rows.append({"channel": channel, "grn_hz": rate, "mn9_hz": result.rate(mn9)})
            logger.info("{:<7s} {:3d} Hz -> MN9 {:5.1f} Hz", channel, rate, result.rate(mn9))
    return pd.DataFrame(rows)


def bitter_suppression(network: LIFNetwork, sugar: NeuronGroup, bitter: NeuronGroup, mn9: int,
                       bitter_rates: list[int], t_run: float, n_trials: int) -> pd.DataFrame:
    """
    MN9 rate with sugar at 100 Hz and bitter at each rate.

    :param network: Network to run.
    :param sugar: Model indices of the sugar GRNs.
    :param bitter: Model indices of the bitter GRNs.
    :param mn9: Model index of MN9.
    :param bitter_rates: Bitter drive rates in Hz.
    :param t_run: Trial duration in seconds.
    :param n_trials: Repeats per point.
    :return: Columns bitter_hz, mn9_hz.
    """
    rows = []
    for rate in bitter_rates:
        result = network.run({sugar: 100.0, bitter: float(rate)}, t_run=t_run, n_trials=n_trials)
        rows.append({"bitter_hz": rate, "mn9_hz": result.rate(mn9)})
        logger.info("sugar 100 + bitter {:3d} Hz -> MN9 {:5.1f} Hz", rate, result.rate(mn9))
    return pd.DataFrame(rows)


def plot_summary(dose: pd.DataFrame, channels: pd.DataFrame, suppression: pd.DataFrame, out_path: Path) -> None:
    """
    Three-panel figure: dose-response, channel comparison, bitter suppression.

    :param dose: Output of ``sugar_dose_response``.
    :param channels: Output of ``channel_comparison``.
    :param suppression: Output of ``bitter_suppression``.
    :param out_path: PNG path.
    """
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.1), facecolor=PAPER)

    ax = axes[0]
    style_axes(ax)
    ax.plot(dose.grn_hz, dose.mn9_hz, color=CHANNEL_COLOURS["sugar"], lw=2, marker="o", ms=5, solid_capstyle="round")
    ax.set_xlabel("sugar GRN drive (Hz)", color=INK2)
    ax.set_ylabel("MN9 firing rate (Hz)", color=INK2)
    ax.set_title("Sweeter input, stronger PER drive", color=INK, fontsize=11, loc="left")

    ax = axes[1]
    style_axes(ax)
    for channel, series in channels.groupby("channel"):
        ax.plot(series.grn_hz, series.mn9_hz, color=CHANNEL_COLOURS[channel], lw=2, marker="o", ms=5,
                label=channel, solid_capstyle="round")
        endpoint = series.iloc[-1]
        if endpoint.mn9_hz > 1:
            ax.annotate(channel, (endpoint.grn_hz, endpoint.mn9_hz), color=INK2, fontsize=9,
                        xytext=(4, 0), textcoords="offset points", va="center")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    ax.set_xlabel("GRN drive (Hz)", color=INK2)
    ax.set_ylabel("MN9 firing rate (Hz)", color=INK2)
    ax.set_title("Only sugar and (strong) water reach MN9", color=INK, fontsize=11, loc="left")

    ax = axes[2]
    style_axes(ax)
    ax.plot(suppression.bitter_hz, suppression.mn9_hz, color=CHANNEL_COLOURS["bitter"], lw=2, marker="o", ms=5,
            solid_capstyle="round")
    ax.set_xlabel("bitter GRN drive (Hz), sugar held at 100 Hz", color=INK2)
    ax.set_ylabel("MN9 firing rate (Hz)", color=INK2)
    ax.set_title("Bitter shuts the reflex down", color=INK, fontsize=11, loc="left")

    figure.suptitle("A virtual fly tastes something: FlyWire connectome, LIF dynamics",
                    color=INK, fontsize=13, x=0.007, ha="left", y=0.99)
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    figure.savefig(out_path, dpi=160, facecolor=PAPER)
    plt.close(figure)


def main(t_run: float, n_trials: int, quick: bool) -> None:
    """
    Run the four experiments and write CSVs and the figure.

    :param t_run: Trial duration in seconds.
    :param n_trials: Repeats per point.
    :param quick: Use fewer drive levels.
    """
    connectome = Connectome()
    network = LIFNetwork(connectome)
    mn9 = connectome.index(circuits.MN9)
    channels: dict[str, NeuronGroup] = {name: tuple(connectome.index(ids).tolist()) for name, ids in circuits.SETS.items()}
    RESULTS_DIR.mkdir(exist_ok=True)

    dose = sugar_dose_response(network, channels["sugar"], mn9,
                               [25, 50, 75, 100, 150, 200] if quick else list(range(10, 201, 20)), t_run, n_trials)
    dose.to_csv(RESULTS_DIR / "sugar_dose_response.csv", index=False)

    comparison = channel_comparison(network, channels, mn9, [50, 100, 200] if quick else [50, 100, 150, 200, 250],
                                    t_run, n_trials)
    comparison.to_csv(RESULTS_DIR / "channel_comparison.csv", index=False)

    suppression = bitter_suppression(network, channels["sugar"], channels["bitter"], mn9,
                                     [0, 50, 100, 200] if quick else [0, 20, 40, 60, 80, 100, 150, 200], t_run, n_trials)
    suppression.to_csv(RESULTS_DIR / "bitter_suppression.csv", index=False)

    result = network.run({channels["sugar"]: 100.0}, t_run=t_run, n_trials=n_trials)
    active = result.table(min_rate=1.0, names=circuits.names(connectome))
    active.to_csv(RESULTS_DIR / "sugar100_active_neurons.csv")
    logger.info("{} neurons active above 1 Hz under 100 Hz sugar; top 10:\n{}", len(active), active.head(10).to_string())

    plot_summary(dose, comparison, suppression, RESULTS_DIR / "taste_to_per.png")
    logger.success("wrote {}", RESULTS_DIR / "taste_to_per.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t-run", type=float, default=1.0, help="trial duration (s)")
    parser.add_argument("--trials", type=int, default=1, help="repeats per condition")
    parser.add_argument("--quick", action="store_true", help="fewer drive levels")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    configure(verbose=args.verbose)
    main(args.t_run, args.trials, args.quick)
