"""Render an episode: brain views, a readout trace, and either the proboscis rig
(when the readout is MN9) or a live panel of the loudest output neurons.

Also writes the JSON payload that the interactive page replays.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from . import circuits, viz  # noqa: E402
from .anatomy import Anatomy  # noqa: E402
from .connectome import Connectome  # noqa: E402
from .lif import Result  # noqa: E402
from .viz import ACCENT, INK, INK2, MUTED, SUGAR, SURFACE, BrainRaster  # noqa: E402

# stimulus band colours, in order of appearance (dataviz categorical slots)
BAND_COLOURS = ["#3987e5", "#d55181", "#199e70", "#c98500", "#9085e9", "#d95926"]
BODY_PART_STYLE = {  # part -> (face colour, edge colour, z-order)
    "head": ("#3c3b36", "#6f6e68", 1), "eye": ("#7a4436", "#a9614c", 2),
    "antenna": ("#4a4942", "#6f6e68", 2), "rostrum": ("#6b6a60", "#a2a094", 3),
    "haustellum": ("#7d7b70", "#b3b1a3", 4), "lobeA": ("#9a9788", "#d6d3c2", 5),
    "lobeB": ("#9a9788", "#d6d3c2", 5)}
DROPLET_POSITION = (0.55, 1.92)
TIP_TOUCH_Y = 1.72
MOTOR_SATURATION_HZ = 90.0
MUSCLE_SMOOTHING = 0.28
N_BACKGROUND_DOTS = 12000


@dataclass
class Drive:
    """One named stimulus: which neurons, and when they fire."""
    name: str
    indices: np.ndarray
    windows: list[tuple[float, float, float]]     # (t_start, t_stop, hz)

    def is_on(self, t: float) -> bool:
        """
        Whether this drive is active at a time.

        :param t: Time in seconds.
        :return: True if inside any window.
        """
        return any(t_start <= t < t_stop for t_start, t_stop, _ in self.windows)


@dataclass
class EpisodeData:
    """Everything the renderer and the page need, derived once from a Result."""
    times: np.ndarray                 # (n_bins,)
    active: np.ndarray                # model indices that ever spiked
    rates_hz: np.ndarray              # (n_bins, n_active)
    rate_display_max: float
    readout_trace_hz: np.ndarray      # (n_bins,)
    extension: np.ndarray             # low-passed readout, 0-1
    bin_width: float

    @classmethod
    def from_result(cls, result: Result, readout: int) -> "EpisodeData":
        """
        Derive the per-frame arrays a movie or page needs.

        :param result: A Result recorded with ``bin_width``.
        :param readout: Model index of the readout neuron.
        :return: EpisodeData.
        :raises ValueError: if the result has no binned record.
        """
        if result.binned is None or result.bin_width is None:
            raise ValueError("render needs a Result recorded with bin_width")
        binned = result.binned[0]
        active = np.flatnonzero(binned.sum(axis=0) > 0)
        rates_hz = binned[:, active].astype(np.float32) / result.bin_width
        display_max = float(np.percentile(rates_hz[rates_hz > 0], 90)) if (rates_hz > 0).any() else 1.0
        readout_trace = binned[:, readout].astype(np.float32) / result.bin_width
        return cls(result.bin_times(), active, rates_hz, display_max, readout_trace,
                   low_pass(np.clip(readout_trace / MOTOR_SATURATION_HZ, 0, 1), MUSCLE_SMOOTHING),
                   result.bin_width)

    def stimulus_label(self, drives: list[Drive], t: float) -> str:
        """
        Names of the drives active at a time.

        :param drives: The episode's drives.
        :param t: Time in seconds.
        :return: "a + b", or "no stimulus".
        """
        on = [drive.name for drive in drives if drive.is_on(t)]
        return " + ".join(on) if on else "no stimulus"


def low_pass(signal: np.ndarray, alpha: float) -> np.ndarray:
    """
    First-order low-pass filter (a real proboscis has inertia).

    :param signal: Input samples.
    :param alpha: Smoothing factor per sample, in (0, 1].
    :return: Filtered samples.
    """
    smoothed = np.zeros_like(signal)
    for step in range(1, len(signal)):
        smoothed[step] = smoothed[step - 1] + alpha * (signal[step] - smoothed[step - 1])
    return smoothed


def style_trace_axes(ax: Axes) -> None:
    """
    Dark-surface styling for a line-plot axes.

    :param ax: Axes to style.
    """
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#3a3a37")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color="#2b2b29", lw=0.6)
    ax.set_axisbelow(True)


def style_blank_axes(ax: Axes) -> None:
    """
    Dark-surface styling for an image axes with no ticks or spines.

    :param ax: Axes to style.
    """
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def annotate_brain_view(ax: Axes, raster: BrainRaster, anatomy: Anatomy, readout: int,
                        readout_name: str, drives: list[Drive]) -> None:
    """
    Mark the readout with a tick and leader, and label each drive's centroid.

    :param ax: Brain-view axes.
    :param raster: Raster the axes shows.
    :param anatomy: Loaded anatomy.
    :param readout: Model index of the readout neuron.
    :param readout_name: Label for the readout.
    :param drives: Drives to label (first three).
    """
    marker_x, marker_y = raster.marker(readout)
    # a tick and a leader, not a ring: a ring this size covers ~100 neurons
    ax.plot([marker_x], [marker_y], marker="+", ms=7, mec=ACCENT, mew=1.1, zorder=5)
    ax.annotate(readout_name, (marker_x, marker_y), xytext=(34, -22), textcoords="offset points",
                color=ACCENT, fontsize=9, zorder=5,
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8, shrinkA=0, shrinkB=5))
    for position, drive in enumerate(drives[:3]):
        colour = BAND_COLOURS[position % len(BAND_COLOURS)]
        known = anatomy.known[drive.indices]
        if not known.any():
            continue
        centroid_x = raster.pixel_x[drive.indices[known]].mean()
        centroid_y = raster.pixel_y[drive.indices[known]].mean()
        ax.annotate(drive.name, (centroid_x, centroid_y), xytext=(-62, 26 + 16 * position),
                    textcoords="offset points", color=colour, fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.9))


@dataclass
class BodyPanel:
    """The proboscis rig, or a bar chart of output neurons, on the right-hand axes."""
    ax: Axes
    show_proboscis: bool
    readout_name: str
    parts: dict[str, Polygon] = field(default_factory=dict)
    droplet: object = None
    bars: object = None
    bar_slots: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def build(self, data: EpisodeData, anatomy: Anatomy, readout: int) -> None:
        """
        Draw the static parts of the right-hand panel.

        :param data: Episode arrays.
        :param anatomy: Loaded anatomy.
        :param readout: Model index of the readout neuron.
        """
        if self.show_proboscis:
            self._build_proboscis()
        else:
            self._build_output_bars(data, anatomy, readout)

    def _build_proboscis(self) -> None:
        """
        Draw the head, the resting proboscis, the surface and the droplet.
        """
        ax = self.ax
        ax.set_xlim(-1.5, 1.9)
        ax.set_ylim(2.45, -1.35)
        ax.set_aspect("equal")
        ax.set_title(f"proboscis, driven by {self.readout_name}", color=INK2, fontsize=10, loc="left")
        for part_name, polygon in viz.proboscis_pose(0.0).polygons().items():
            face, edge, z_order = BODY_PART_STYLE[part_name]
            patch = Polygon(polygon, closed=True, fc=face, ec=edge, lw=1.1, zorder=z_order)
            ax.add_patch(patch)
            self.parts[part_name] = patch
        self.droplet = ax.plot([DROPLET_POSITION[0]], [DROPLET_POSITION[1]], marker="o", ms=9,
                               color=SUGAR, alpha=0.55, zorder=6)[0]
        ax.plot([-1.4, 1.8], [2.03, 2.03], color="#4a4942", lw=2.5, zorder=0)
        ax.annotate("sugar droplet", (-0.05, 1.96), color=SUGAR, fontsize=9, ha="right", va="center")

    def _build_output_bars(self, data: EpisodeData, anatomy: Anatomy, readout: int) -> None:
        """
        Draw empty bars for the eight loudest motor and descending neurons.

        :param data: Episode arrays.
        :param anatomy: Loaded anatomy.
        :param readout: Model index of the readout neuron (coloured with the accent).
        """
        ax = self.ax
        style_trace_axes(ax)
        ax.grid(False)
        ax.set_title("loudest motor & descending neurons", color=INK2, fontsize=10, loc="left")
        super_class = np.asarray(anatomy.super_class, dtype=object)[data.active]
        output_slots = np.flatnonzero(np.isin(super_class, ["motor", "descending"]))
        self.bar_slots = output_slots[np.argsort(-data.rates_hz[:, output_slots].mean(axis=0))][:8]
        labels = [f"{anatomy.type_label(data.active[slot])} ({str(anatomy.side[data.active[slot]])[:1] or '?'})"
                  for slot in self.bar_slots]
        y_positions = np.arange(len(self.bar_slots))[::-1]
        self.bars = ax.barh(y_positions, np.zeros(len(self.bar_slots)), color=MUTED, height=0.62)
        shown = list(data.active[self.bar_slots])
        if readout in shown:
            self.bars[shown.index(readout)].set_color(ACCENT)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=8.5, color=INK2, family="monospace")
        x_max = float(data.rates_hz[:, self.bar_slots].max()) * 1.05 if len(self.bar_slots) else 60.0
        ax.set_xlim(0, max(60.0, x_max))
        ax.set_xlabel("rate (Hz)", color=INK2, fontsize=9)

    def update(self, data: EpisodeData, frame: int) -> None:
        """
        Move the proboscis, or set the bar lengths, for one frame.

        :param data: Episode arrays.
        :param frame: Frame index.
        """
        if self.show_proboscis:
            pose = viz.proboscis_pose(data.extension[frame])
            for part_name, polygon in pose.polygons().items():
                self.parts[part_name].set_xy(polygon)
            touching = pose.tip[1] > TIP_TOUCH_Y
            self.droplet.set_alpha(1.0 if touching else 0.5)
            self.droplet.set_markersize(13 if touching else 9)
        elif self.bars is not None:
            for bar, slot in zip(self.bars, self.bar_slots):
                bar.set_width(data.rates_hz[frame, slot])


def build_trace_axes(ax: Axes, data: EpisodeData, drives: list[Drive], readout_name: str) -> tuple[Line2D, Line2D]:
    """
    Draw the static readout-trace axes: bands for each drive window, labels, limits.

    :param ax: Trace axes.
    :param data: Episode arrays.
    :param drives: The episode's drives.
    :param readout_name: Label for the y axis.
    :return: (trace line, time cursor) artists to animate.
    """
    style_trace_axes(ax)
    ax.set_xlim(0, data.times[-1] + data.bin_width)
    ax.set_ylim(-6, max(120, data.readout_trace_hz.max() * 1.15))
    ax.set_xlabel("time (s)", color=INK2, fontsize=9)
    ax.set_ylabel(f"{readout_name} (Hz)", color=INK2, fontsize=9)
    y_top = ax.get_ylim()[1]
    for position, drive in enumerate(drives):
        colour = BAND_COLOURS[position % len(BAND_COLOURS)]
        for t_start, t_stop, hz in drive.windows:
            ax.axvspan(t_start, t_stop, color=colour, alpha=0.14, lw=0)
            ax.annotate(f"{drive.name} {hz:.0f} Hz", ((t_start + t_stop) / 2, y_top * (0.93 - 0.09 * position)),
                        color=colour, fontsize=8.5, ha="center")
    line, = ax.plot([], [], color=ACCENT, lw=1.8, solid_capstyle="round")
    cursor = ax.axvline(0, color=INK2, lw=0.9, alpha=0.7)
    return line, cursor


def render_episode(connectome: Connectome, anatomy: Anatomy, result: Result, drives: list[Drive],
                   readout: int, out_prefix: str | Path, title: str, fps: int = 20, gif: bool = True,
                   body: bool | None = None) -> str:
    """
    Render an episode to <out_prefix>.mp4 (and .gif) and write <out_prefix>.json for the page.

    :param connectome: Loaded connectome.
    :param anatomy: Loaded anatomy.
    :param result: A Result recorded with ``bin_width``.
    :param drives: The episode's drives.
    :param readout: Model index of the readout neuron.
    :param out_prefix: Output path without extension.
    :param title: Figure title.
    :param fps: Movie frame rate.
    :param gif: Also write a half-rate GIF.
    :param body: Show the proboscis rig; None means only when the readout is MN9.
    :return: Path of the mp4.
    """
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    data = EpisodeData.from_result(result, readout)
    is_mn9 = readout == connectome.index(circuits.MN9)
    show_proboscis = is_mn9 if body is None else body
    readout_name = "MN9" if is_mn9 else anatomy.type_label(readout)
    logger.info("rendering {} frames, {} active neurons, readout {}", len(data.times), len(data.active), readout_name)

    frontal_x, frontal_y = anatomy.project("frontal")
    dorsal_x, dorsal_z = anatomy.project("dorsal")
    frontal = BrainRaster(frontal_x, frontal_y, width=520)
    dorsal = BrainRaster(dorsal_x, dorsal_z, width=520, flip_y=False)

    figure: Figure = plt.figure(figsize=(13.2, 6.4), facecolor=SURFACE)
    grid = figure.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], height_ratios=[1.55, 1.0],
                               hspace=0.22, wspace=0.10, left=0.045, right=0.985, top=0.90, bottom=0.10)
    ax_frontal = figure.add_subplot(grid[0, 0])
    ax_dorsal = figure.add_subplot(grid[0, 1])
    ax_body = figure.add_subplot(grid[:, 2])
    ax_trace = figure.add_subplot(grid[1, 0:2])
    for ax in (ax_frontal, ax_dorsal, ax_body):
        style_blank_axes(ax)

    highlight = (readout, float(data.readout_trace_hz[0]))
    image_frontal = ax_frontal.imshow(frontal.frame(data.active, data.rates_hz[0], data.rate_display_max,
                                                    highlight=highlight), interpolation="bilinear")
    image_dorsal = ax_dorsal.imshow(dorsal.frame(data.active, data.rates_hz[0], data.rate_display_max,
                                                 highlight=highlight), interpolation="bilinear")
    ax_frontal.set_title("frontal view — the fly's face", color=INK2, fontsize=10, loc="left")
    ax_dorsal.set_title("dorsal view — looking down", color=INK2, fontsize=10, loc="left")
    annotate_brain_view(ax_frontal, frontal, anatomy, readout, readout_name, drives)
    annotate_brain_view(ax_dorsal, dorsal, anatomy, readout, readout_name, drives)

    body_panel = BodyPanel(ax_body, show_proboscis, readout_name)
    body_panel.build(data, anatomy, readout)
    trace_line, trace_cursor = build_trace_axes(ax_trace, data, drives, readout_name)
    figure.suptitle(title, color=INK, fontsize=14, x=0.015, ha="left", y=0.975)
    subtitle = figure.text(0.015, 0.925, "", color=INK2, fontsize=10, ha="left")

    frames: list[np.ndarray] = []
    for frame, t in enumerate(data.times):
        highlight = (readout, float(data.readout_trace_hz[frame]))
        image_frontal.set_data(frontal.frame(data.active, data.rates_hz[frame], data.rate_display_max, highlight=highlight))
        image_dorsal.set_data(dorsal.frame(data.active, data.rates_hz[frame], data.rate_display_max, highlight=highlight))
        body_panel.update(data, frame)
        trace_line.set_data(data.times[:frame + 1], data.readout_trace_hz[:frame + 1])
        trace_cursor.set_xdata([t, t])
        subtitle.set_text(f"t = {t:5.2f} s     {data.stimulus_label(drives, t)}     "
                          f"{readout_name} {data.readout_trace_hz[frame]:5.0f} Hz     "
                          f"{int((data.rates_hz[frame] > 0).sum()):4d} neurons firing")
        figure.canvas.draw()
        frames.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
        if frame % 40 == 0:
            logger.debug("frame {}/{}", frame, len(data.times))
    plt.close(figure)

    mp4_path = f"{out_prefix}.mp4"
    imageio.mimwrite(mp4_path, frames, fps=fps, codec="libx264", quality=8,
                     macro_block_size=None, ffmpeg_log_level="error")
    logger.success("wrote {}", mp4_path)
    if gif:
        imageio.mimwrite(f"{out_prefix}.gif", frames[::2], fps=max(1, fps // 2), loop=0)
        logger.success("wrote {}.gif", out_prefix)
    write_page_payload(connectome, anatomy, data, frontal, dorsal, drives, readout, f"{out_prefix}.json")
    return mp4_path


def write_page_payload(connectome: Connectome, anatomy: Anatomy, data: EpisodeData, frontal: BrainRaster,
                       dorsal: BrainRaster, drives: list[Drive], readout: int, path: str | Path) -> None:
    """
    Write the compact JSON the interactive page replays.

    :param connectome: Loaded connectome.
    :param anatomy: Loaded anatomy.
    :param data: Episode arrays.
    :param frontal: Frontal-view raster.
    :param dorsal: Dorsal-view raster.
    :param drives: The episode's drives.
    :param readout: Model index of the readout neuron.
    :param path: Output .json path.
    """
    background = np.flatnonzero(anatomy.known)
    background = np.random.default_rng(0).choice(background, size=min(N_BACKGROUND_DOTS, len(background)), replace=False)
    quantised = np.clip(np.round(data.rates_hz / max(data.rates_hz.max(), 1e-9) * 255), 0, 255).astype(np.uint8)
    segments = [[t_start, t_stop, f"{drive.name} {hz:.0f} Hz"]
                for drive in drives for t_start, t_stop, hz in drive.windows]
    readout_slot = int(np.flatnonzero(data.active == readout)[0]) if readout in data.active else -1
    payload = {
        "bin": data.bin_width,
        "times": [round(float(t), 3) for t in data.times],
        "segments": segments,
        "rate_max": float(data.rates_hz.max()),
        "mn9": [round(float(v), 1) for v in data.readout_trace_hz],
        "extension": [round(float(v), 3) for v in data.extension],
        "views": {name: view_payload(raster, background, data.active)
                  for name, raster in (("frontal", frontal), ("dorsal", dorsal))},
        "active": {"label": [anatomy.describe(i) for i in data.active],
                   "root_id": [str(connectome.root_ids[i]) for i in data.active],
                   "cls": [str(anatomy.super_class[i]) for i in data.active],
                   "rates": [row.tolist() for row in quantised]},
        "mn9_slot": readout_slot,
    }
    Path(path).write_text(json.dumps(payload, separators=(",", ":")))
    logger.success("wrote {} ({:.1f} MB)", path, Path(path).stat().st_size / 1e6)


def view_payload(raster: BrainRaster, background: np.ndarray, active: np.ndarray) -> dict:
    """
    Pixel positions for one view.

    :param raster: The view's raster.
    :param background: Model indices of the faint background dots.
    :param active: Model indices of the neurons that ever fired.
    :return: Dict with width, height, background and active pixel lists.
    """
    return {"w": raster.width, "h": raster.height,
            "bg": [[int(raster.pixel_x[i]), int(raster.pixel_y[i])] for i in background],
            "xy": [[int(raster.pixel_x[i]), int(raster.pixel_y[i])] for i in active]}
