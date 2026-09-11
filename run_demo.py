#!/usr/bin/env python3
"""Reproduce the core taste -> proboscis-extension results on the FlyWire connectome.

Runs four experiments and writes CSVs + a figure into results/:
  1. dose-response: sugar GRN rate  -> MN9 rate
  2. channel comparison at 100 Hz:  sugar / bitter / water / Ir94e -> MN9
  3. bitter suppression: sugar 100 Hz + bitter at increasing rate -> MN9
  4. what the brain does under sugar: the most active downstream neurons
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flybrain import Connectome, LIFNetwork, circuits

# dataviz categorical slots 1-4 (light mode), validated for CVD separation
C = {"sugar": "#2a78d6", "bitter": "#eb6834", "water": "#1baf7a", "ir94e": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"
RESULTS = Path(__file__).resolve().parent / "results"


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def main(t_run, n_trials, quick):
    c = Connectome()
    print(c, flush=True)
    net = LIFNetwork(c)
    mn9 = c.index(circuits.MN9)
    sets = {k: tuple(c.index(v).tolist()) for k, v in circuits.SETS.items()}
    RESULTS.mkdir(exist_ok=True)
    run = lambda stim: net.run(stim, t_run=t_run, n_trials=n_trials)

    # 1. sugar dose-response ------------------------------------------------
    rates = [25, 50, 75, 100, 150, 200] if quick else list(range(10, 201, 20))
    dose = []
    for f in rates:
        r = run({sets["sugar"]: float(f)})
        dose.append((f, r.rate(mn9), r.rates_std[mn9]))
        print(f"sugar {f:3d} Hz -> MN9 {dose[-1][1]:5.1f} Hz", flush=True)
    dose = pd.DataFrame(dose, columns=["grn_hz", "mn9_hz", "mn9_std"])
    dose.to_csv(RESULTS / "sugar_dose_response.csv", index=False)

    # 2. all four taste channels -------------------------------------------
    chan = []
    for tag, idx in sets.items():
        for f in ([50, 100, 200] if quick else [50, 100, 150, 200, 250]):
            r = run({idx: float(f)})
            chan.append((tag, f, r.rate(mn9)))
            print(f"{tag:7s} {f:3d} Hz -> MN9 {r.rate(mn9):5.1f} Hz", flush=True)
    chan = pd.DataFrame(chan, columns=["channel", "grn_hz", "mn9_hz"])
    chan.to_csv(RESULTS / "channel_comparison.csv", index=False)

    # 3. bitter suppresses sugar -------------------------------------------
    supp = []
    for b in ([0, 50, 100, 200] if quick else [0, 20, 40, 60, 80, 100, 150, 200]):
        r = run({sets["sugar"]: 100.0, sets["bitter"]: float(b)})
        supp.append((b, r.rate(mn9)))
        print(f"sugar 100 + bitter {b:3d} Hz -> MN9 {r.rate(mn9):5.1f} Hz", flush=True)
    supp = pd.DataFrame(supp, columns=["bitter_hz", "mn9_hz"])
    supp.to_csv(RESULTS / "bitter_suppression.csv", index=False)

    # 4. the population response under sugar --------------------------------
    r = run({sets["sugar"]: 100.0})
    tbl = r.table(min_rate=1.0, names=circuits.names(c))
    tbl.to_csv(RESULTS / "sugar100_active_neurons.csv")
    print(f"\n{len(tbl)} neurons active above 1 Hz under 100 Hz sugar")
    print(tbl.head(15).to_string())

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), facecolor="#fcfcfb")

    ax = axes[0]; style(ax)
    ax.plot(dose.grn_hz, dose.mn9_hz, color=C["sugar"], lw=2,
            marker="o", ms=5, solid_capstyle="round")
    ax.set_xlabel("sugar GRN drive (Hz)", color=INK2)
    ax.set_ylabel("MN9 firing rate (Hz)", color=INK2)
    ax.set_title("Sweeter input, stronger PER drive", color=INK, fontsize=11, loc="left")

    ax = axes[1]; style(ax)
    for tag, sub in chan.groupby("channel"):
        ax.plot(sub.grn_hz, sub.mn9_hz, color=C[tag], lw=2, marker="o", ms=5,
                label=tag, solid_capstyle="round")
        last = sub.iloc[-1]
        if last.mn9_hz > 1:
            ax.annotate(tag, (last.grn_hz, last.mn9_hz), color=INK2, fontsize=9,
                        xytext=(4, 0), textcoords="offset points", va="center")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    ax.set_xlabel("GRN drive (Hz)", color=INK2)
    ax.set_ylabel("MN9 firing rate (Hz)", color=INK2)
    ax.set_title("Only sugar and (strong) water reach MN9", color=INK, fontsize=11, loc="left")

    ax = axes[2]; style(ax)
    ax.plot(supp.bitter_hz, supp.mn9_hz, color=C["bitter"], lw=2,
            marker="o", ms=5, solid_capstyle="round")
    ax.set_xlabel("bitter GRN drive (Hz), sugar held at 100 Hz", color=INK2)
    ax.set_ylabel("MN9 firing rate (Hz)", color=INK2)
    ax.set_title("Bitter shuts the reflex down", color=INK, fontsize=11, loc="left")

    fig.suptitle("A virtual fly tastes something: FlyWire connectome, LIF dynamics",
                 color=INK, fontsize=13, x=0.007, ha="left", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = RESULTS / "taste_to_per.png"
    fig.savefig(out, dpi=160, facecolor="#fcfcfb")
    print("\nwrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-run", type=float, default=1.0, help="trial duration (s)")
    ap.add_argument("--trials", type=int, default=1, help="repeats per condition")
    ap.add_argument("--quick", action="store_true", help="fewer drive levels")
    a = ap.parse_args()
    main(a.t_run, a.trials, a.quick)
