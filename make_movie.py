#!/usr/bin/env python3
"""Simulate a taste episode and render it: brain, body, and motor output.

Timeline (2.4 s of simulated time):
    0.00-0.40  nothing
    0.40-1.10  sugar GRNs at 100 Hz          -> the fly should extend
    1.10-1.50  nothing
    1.50-2.20  sugar 100 Hz + bitter 150 Hz  -> the veto
    2.20-2.40  nothing
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import imageio.v2 as imageio

from flybrain import Connectome, LIFNetwork, circuits
from flybrain.anatomy import Anatomy
from flybrain import viz
from flybrain.viz import SURFACE, INK, INK2, ACCENT, SUGAR, BITTER, MUTED

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
BIN = 0.02
SEGMENTS = [(0.40, 1.10, "sugar"), (1.50, 2.20, "sugar+bitter")]


def simulate(t_run, seed):
    conn = Connectome()
    anat = Anatomy(conn)
    net = LIFNetwork(conn, seed=seed)
    S = lambda k: tuple(conn.index(circuits.SETS[k]).tolist())
    res = net.run({S("sugar"):  [(0.40, 1.10, 100.0), (1.50, 2.20, 100.0)],
                   S("bitter"): [(1.50, 2.20, 150.0)]},
                  t_run=t_run, bin_width=BIN)
    return conn, anat, net, res


def main(t_run, seed, fps, gif):
    conn, anat, net, res = simulate(t_run, seed)
    mn9 = conn.index(circuits.MN9)
    sugar_idx = conn.index(circuits.SUGAR_GRN)
    bitter_idx = conn.index(circuits.BITTER_GRN)

    binned = res.binned[0]                       # (n_bins, n_neurons)
    times = res.bin_times()
    active = np.flatnonzero(binned.sum(axis=0) > 0)
    print(f"{len(active)} neurons spiked at some point; {len(times)} frames")

    rates = binned[:, active].astype(np.float32) / BIN       # Hz
    vmax = float(np.percentile(rates[rates > 0], 90)) if (rates > 0).any() else 1.0
    mn9_trace = binned[:, mn9].astype(np.float32) / BIN
    # a real proboscis has inertia; low-pass the motor command
    ext = np.clip(mn9_trace / 90.0, 0, 1)
    alpha = 0.28
    smooth = np.zeros_like(ext)
    for i in range(1, len(ext)):
        smooth[i] = smooth[i - 1] + alpha * (ext[i] - smooth[i - 1])

    fx, fy = anat.project("frontal")
    dx, dz = anat.project("dorsal")
    front = viz.BrainRaster(fx, fy, width=520)
    dorsal = viz.BrainRaster(dx, dz, width=520, flip_y=False)

    fig = plt.figure(figsize=(13.2, 6.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0],
                          height_ratios=[1.55, 1.0],
                          hspace=0.22, wspace=0.10,
                          left=0.045, right=0.985, top=0.90, bottom=0.10)
    ax_f = fig.add_subplot(gs[0, 0]); ax_d = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[:, 2]);  ax_t = fig.add_subplot(gs[1, 0:2])
    for ax in (ax_f, ax_d, ax_b):
        ax.set_facecolor(SURFACE); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

    im_f = ax_f.imshow(front.frame(active, rates[0], vmax), interpolation="bilinear")
    im_d = ax_d.imshow(dorsal.frame(active, rates[0], vmax), interpolation="bilinear")
    ax_f.set_title("frontal view — the fly's face", color=INK2, fontsize=10, loc="left")
    ax_d.set_title("dorsal view — looking down", color=INK2, fontsize=10, loc="left")
    for ax, ras in ((ax_f, front), (ax_d, dorsal)):
        mx, my = ras.marker(mn9)
        ax.plot([mx], [my], marker="o", ms=11, mfc="none", mec=ACCENT, mew=1.6, zorder=5)
        ax.annotate("MN9", (mx, my), xytext=(13, -2), textcoords="offset points",
                    color=ACCENT, fontsize=9, zorder=5)
        sx = ras.px[sugar_idx].mean(); sy = ras.py[sugar_idx].mean()
        ax.annotate("sugar GRNs", (sx, sy), xytext=(-62, 26),
                    textcoords="offset points", color=SUGAR, fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=SUGAR, lw=0.9))

    # body panel
    ax_b.set_xlim(-1.5, 1.9); ax_b.set_ylim(2.45, -1.35); ax_b.set_aspect("equal")
    ax_b.set_title("proboscis, driven by MN9", color=INK2, fontsize=10, loc="left")
    body = {}
    P = viz.proboscis_polys(0.0)
    for key, fc, ec, z in (("head", "#3c3b36", "#6f6e68", 1),
                           ("eye", "#7a4436", "#a9614c", 2),
                           ("antenna", "#4a4942", "#6f6e68", 2),
                           ("rostrum", "#6b6a60", "#a2a094", 3),
                           ("haustellum", "#7d7b70", "#b3b1a3", 4),
                           ("lobeA", "#9a9788", "#d6d3c2", 5),
                           ("lobeB", "#9a9788", "#d6d3c2", 5)):
        p = Polygon(P[key], closed=True, fc=fc, ec=ec, lw=1.1, zorder=z)
        ax_b.add_patch(p); body[key] = p
    drop = ax_b.plot([0.55], [1.92], marker="o", ms=9, color=SUGAR, alpha=0.55,
                     zorder=6)[0]
    ax_b.plot([-1.4, 1.8], [2.03, 2.03], color="#4a4942", lw=2.5, zorder=0)
    ax_b.annotate("sugar droplet", (-0.05, 1.96), color=SUGAR, fontsize=9,
                  ha="right", va="center")

    # motor trace
    ax_t.set_facecolor(SURFACE)
    for s in ("top", "right"): ax_t.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax_t.spines[s].set_color("#3a3a37")
    ax_t.tick_params(colors=INK2, labelsize=9)
    ax_t.grid(True, color="#2b2b29", lw=0.6); ax_t.set_axisbelow(True)
    ax_t.set_xlim(0, times[-1] + BIN); ax_t.set_ylim(-6, max(120, mn9_trace.max() * 1.15))
    ax_t.set_xlabel("time (s)", color=INK2, fontsize=9)
    ax_t.set_ylabel("MN9 (Hz)", color=INK2, fontsize=9)
    for (t0, t1, lab) in SEGMENTS:
        col = SUGAR if lab == "sugar" else BITTER
        ax_t.axvspan(t0, t1, color=col, alpha=0.14, lw=0)
        ax_t.annotate(lab, ((t0 + t1) / 2, ax_t.get_ylim()[1] * 0.93), color=col,
                      fontsize=9, ha="center")
    line, = ax_t.plot([], [], color=ACCENT, lw=1.8, solid_capstyle="round")
    cursor = ax_t.axvline(0, color=INK2, lw=0.9, alpha=0.7)

    fig.suptitle("A virtual fly tastes sugar, then sugar with bitter",
                 color=INK, fontsize=14, x=0.015, ha="left", y=0.975)
    sub = fig.text(0.015, 0.925, "", color=INK2, fontsize=10, ha="left")

    frames = []
    for k, t in enumerate(times):
        im_f.set_data(front.frame(active, rates[k], vmax))
        im_d.set_data(dorsal.frame(active, rates[k], vmax))
        P = viz.proboscis_polys(smooth[k])
        for key, poly in body.items():
            poly.set_xy(P[key])
        touching = P["tip"][1] > 1.72
        drop.set_alpha(1.0 if touching else 0.5)
        drop.set_markersize(13 if touching else 9)
        line.set_data(times[:k + 1], mn9_trace[:k + 1])
        cursor.set_xdata([t, t])
        state = next((l for (a, b, l) in SEGMENTS if a <= t < b), "no stimulus")
        sub.set_text(f"t = {t:5.2f} s     {state}     "
                     f"MN9 {mn9_trace[k]:5.0f} Hz     "
                     f"{int((rates[k] > 0).sum()):4d} neurons firing")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        if k % 20 == 0:
            print(f"  frame {k}/{len(times)}", flush=True)

    OUT.mkdir(exist_ok=True)
    imageio.mimwrite(OUT / "taste_episode.mp4", frames, fps=fps,
                     codec="libx264", quality=8,
                     macro_block_size=None, ffmpeg_log_level="error")
    print("wrote", OUT / "taste_episode.mp4")
    if gif:
        imageio.mimwrite(OUT / "taste_episode.gif", frames[::2], fps=fps // 2, loop=0)
        print("wrote", OUT / "taste_episode.gif")

    # payload for the interactive page
    export(conn, anat, res, active, rates, times, mn9_trace, smooth, front, dorsal)


def export(conn, anat, res, active, rates, times, mn9_trace, smooth, front, dorsal):
    """Compact JSON the scrubber page can replay without re-running anything."""
    bg = np.flatnonzero(anat.known)
    rng = np.random.default_rng(0)
    bg = rng.choice(bg, size=min(12000, len(bg)), replace=False)
    q = np.clip(np.round(rates / max(rates.max(), 1e-9) * 255), 0, 255).astype(np.uint8)
    payload = {
        "bin": BIN,
        "times": [round(float(t), 3) for t in times],
        "segments": [[a, b, l] for a, b, l in SEGMENTS],
        "rate_max": float(rates.max()),
        "mn9": [round(float(v), 1) for v in mn9_trace],
        "extension": [round(float(v), 3) for v in smooth],
        "views": {},
        "active": {
            "label": [anat.describe(i) for i in active],
            "root_id": [str(conn.root_ids[i]) for i in active],
            "cls": [str(anat.super_class[i]) for i in active],
            "rates": [row.tolist() for row in q],
        },
        "mn9_slot": int(np.flatnonzero(active == conn.index(circuits.MN9))[0]),
    }
    for name, ras in (("frontal", front), ("dorsal", dorsal)):
        payload["views"][name] = {
            "w": ras.W, "h": ras.H,
            "bg": [[int(ras.px[i]), int(ras.py[i])] for i in bg],
            "xy": [[int(ras.px[i]), int(ras.py[i])] for i in active],
        }
    (OUT / "episode.json").write_text(json.dumps(payload, separators=(",", ":")))
    mb = (OUT / "episode.json").stat().st_size / 1e6
    print(f"wrote {OUT / 'episode.json'} ({mb:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-run", type=float, default=2.4)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--no-gif", dest="gif", action="store_false")
    a = ap.parse_args()
    main(a.t_run, a.seed, a.fps, a.gif)
