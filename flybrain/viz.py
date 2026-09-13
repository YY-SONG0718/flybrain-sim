"""Rendering primitives: the brain lighting up, and a proboscis that follows MN9.

The brain panels are rasterised directly with NumPy (one additive splat per
neuron, then a blur) rather than drawn as 138,000 matplotlib markers, which
keeps a 100-frame movie to a few seconds per panel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

SURFACE = "#1a1a19"
INK = "#ffffff"
INK2 = "#c3c2b7"
GRID = "#3a3a37"
ACCENT = "#d95926"      # the behavioural readout neuron  (categorical slot 2)
SUGAR = "#3987e5"       # stimulus: sugar                 (categorical slot 1)
BITTER = "#d95926"
MUTED = "#6f6e68"

# sequential ramp for firing rate: one hue, dark -> light, on a dark surface
ACTIVITY = LinearSegmentedColormap.from_list(
    "activity", ["#12243b", "#1f4f8f", "#3987e5", "#7fb6f2", "#e8f2ff"])
# the readout neuron, kept off the blue ramp so it cannot be confused with neighbours
HIGHLIGHT = LinearSegmentedColormap.from_list(
    "highlight", ["#3a1a0c", "#a83f16", "#f0793f", "#ffb98a", "#fff1e6"])

# brightness of a single blurred point source, used to renormalise the splat image
SINGLE_NEURON_PEAK = 0.055

Point = tuple[float, float]


class BrainRaster:
    """Rasterises neuron positions into an image; call `frame()` once per timestep."""

    def __init__(self, x_um: np.ndarray, y_um: np.ndarray, width: int = 560,
                 height: int | None = None, pad_fraction: float = 0.04, flip_y: bool = True) -> None:
        """
        Fit a pixel grid to the neuron positions and draw the static background.

        :param x_um: Horizontal coordinate per neuron (micrometres; NaN if unknown).
        :param y_um: Vertical coordinate per neuron.
        :param width: Image width in pixels.
        :param height: Image height in pixels; derived from the data's aspect ratio if None.
        :param pad_fraction: Margin around the data as a fraction of its extent.
        :param flip_y: Put small y at the bottom of the image.
        """
        x_um = np.asarray(x_um, float)
        y_um = np.asarray(y_um, float)
        self.known: np.ndarray = np.isfinite(x_um) & np.isfinite(y_um)

        x_min, x_max = np.nanmin(x_um[self.known]), np.nanmax(x_um[self.known])
        y_min, y_max = np.nanmin(y_um[self.known]), np.nanmax(y_um[self.known])
        x_pad, y_pad = pad_fraction * (x_max - x_min), pad_fraction * (y_max - y_min)
        x_min, x_max, y_min, y_max = x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad
        self.width: int = width
        self.height: int = height or max(1, int(round(width * (y_max - y_min) / (x_max - x_min))))
        self.extent_um: tuple[float, float, float, float] = (x_min, x_max, y_min, y_max)

        pixel_x = (x_um - x_min) / (x_max - x_min) * (self.width - 1)
        pixel_y = (y_um - y_min) / (y_max - y_min) * (self.height - 1)
        if flip_y:
            pixel_y = (self.height - 1) - pixel_y
        self.pixel_x: np.ndarray = np.where(self.known, pixel_x, 0).astype(np.int32)
        self.pixel_y: np.ndarray = np.where(self.known, pixel_y, 0).astype(np.int32)
        self._flat_index: np.ndarray = self.pixel_y.astype(np.int64) * self.width + self.pixel_x

        density = np.bincount(self._flat_index[self.known], minlength=self.width * self.height)
        density = ndimage.gaussian_filter(density.reshape(self.height, self.width).astype(np.float32), 1.1)
        density = np.log1p(density) / np.log1p(density.max())
        self.background_rgb: np.ndarray = (
            np.stack([density] * 3, -1) * np.array([0.19, 0.19, 0.18])) ** 0.9

    def frame(self, indices: np.ndarray, values: np.ndarray, value_max: float,
              sigma_px: float = 2.4, floor: float = 0.35,
              highlight: tuple[int, float] | None = None) -> np.ndarray:
        """
        Render one frame: the background plus a glow at each active neuron.

        :param indices: Model indices of the neurons to draw.
        :param values: Value (rate) per neuron; zero draws nothing.
        :param value_max: Value that maps to full brightness.
        :param sigma_px: Blur radius in pixels.
        :param floor: Brightness for any non-zero value, so a single spike stays visible.
        :param highlight: (model index, value) to draw in the accent colour, from its own value only.
        :return: RGB image array in [0, 1].
        """
        indices = np.asarray(indices)
        values = np.asarray(values, dtype=np.float32)
        if highlight is not None:
            keep = indices != highlight[0]
            indices, values = indices[keep], values[keep]

        image = self.background_rgb + self._splat(indices, values, value_max, sigma_px, floor, ACTIVITY)
        if highlight is not None and highlight[1] > 0:
            image = image + self._splat(np.array([highlight[0]]), np.array([highlight[1]], dtype=np.float32),
                                        value_max, sigma_px * 0.9, floor, HIGHLIGHT)
        return np.clip(image, 0, 1)

    def _splat(self, indices: np.ndarray, values: np.ndarray, value_max: float,
               sigma_px: float, floor: float, colormap: LinearSegmentedColormap) -> np.ndarray:
        """
        Additive glow image for one set of neurons.

        :param indices: Model indices.
        :param values: Value per neuron.
        :param value_max: Value that maps to full brightness.
        :param sigma_px: Blur radius in pixels.
        :param floor: Brightness for any non-zero value.
        :param colormap: Colour ramp to apply.
        :return: RGB contribution to add to the background.
        """
        intensity = np.zeros(self.width * self.height, dtype=np.float32)
        visible = self.known[indices] & (values > 0)
        if visible.any():
            brightness = floor + (1 - floor) * np.clip(values[visible] / max(value_max, 1e-9), 0, 1)
            np.add.at(intensity, self._flat_index[indices[visible]], brightness.astype(np.float32))
        blurred = ndimage.gaussian_filter(intensity.reshape(self.height, self.width), sigma_px)
        normalised = np.clip(blurred / SINGLE_NEURON_PEAK, 0, 1)
        alpha = np.clip(normalised * 1.35, 0, 1)[..., None] ** 0.5
        return colormap(normalised)[..., :3] * alpha

    def marker(self, index: int) -> Point:
        """
        Pixel coordinates of one neuron.

        :param index: Model index.
        :return: (x, y) in pixels.
        """
        return float(self.pixel_x[index]), float(self.pixel_y[index])


@dataclass
class ProboscisPose:
    """Polygons (in head-radius units) for a fly head in profile."""
    head: np.ndarray
    eye: np.ndarray
    antenna: np.ndarray
    rostrum: np.ndarray
    haustellum: np.ndarray
    lobe_a: np.ndarray
    lobe_b: np.ndarray
    tip: np.ndarray

    def polygons(self) -> dict[str, np.ndarray]:
        """
        Named polygons for drawing.

        :return: Part name -> (n, 2) polygon array.
        """
        return {"head": self.head, "eye": self.eye, "antenna": self.antenna, "rostrum": self.rostrum,
                "haustellum": self.haustellum, "lobeA": self.lobe_a, "lobeB": self.lobe_b}


def proboscis_pose(extension: float) -> ProboscisPose:
    """
    Fly head in profile with the proboscis extended by a fraction.

    A kinematic rig, not a physics model: the rostrum and haustellum joints move through the range a real fly uses during the proboscis extension response.

    :param extension: 0 (retracted) to 1 (fully extended).
    :return: ProboscisPose.
    """
    e = float(np.clip(extension, 0, 1))
    rostrum_angle = np.deg2rad(-118 + 46 * e)
    rostrum_length = 0.42 + 0.30 * e
    rostrum_base = np.array([0.02, 0.62])
    rostrum_tip = rostrum_base + rostrum_length * np.array([np.cos(rostrum_angle), -np.sin(rostrum_angle)])
    haustellum_angle = rostrum_angle + np.deg2rad(96 - 88 * e)
    haustellum_length = 0.26 + 0.34 * e
    labellum = rostrum_tip + haustellum_length * np.array([np.cos(haustellum_angle), -np.sin(haustellum_angle)])
    lobe_spread = 0.06 + 0.14 * e
    lobe_normal = np.array([-np.sin(haustellum_angle), -np.cos(haustellum_angle)])
    return ProboscisPose(
        head=ellipse_polygon(0.0, 0.0, 1.00, 0.92, n_points=64),
        eye=ellipse_polygon(0.42, 0.16, 0.46, 0.60, n_points=48),
        antenna=ellipse_polygon(0.16, -0.40, 0.15, 0.22, n_points=24),
        rostrum=capsule_polygon(rostrum_base, rostrum_tip, 0.145 - 0.02 * e),
        haustellum=capsule_polygon(rostrum_tip, labellum, 0.105 - 0.015 * e),
        lobe_a=ellipse_polygon(*(labellum + lobe_spread * lobe_normal), 0.15, 0.10, rotation=haustellum_angle, n_points=28),
        lobe_b=ellipse_polygon(*(labellum - lobe_spread * lobe_normal), 0.15, 0.10, rotation=haustellum_angle, n_points=28),
        tip=labellum,
    )


def ellipse_polygon(cx: float, cy: float, rx: float, ry: float, rotation: float = 0.0,
                    n_points: int = 48) -> np.ndarray:
    """
    Polygon approximating an ellipse.

    :param cx: Centre x.
    :param cy: Centre y.
    :param rx: Radius along the ellipse's own x axis.
    :param ry: Radius along its y axis.
    :param rotation: Rotation in radians.
    :param n_points: Number of vertices.
    :return: (n_points, 2) array.
    """
    angles = np.linspace(0, 2 * np.pi, n_points)
    x, y = rx * np.cos(angles), ry * np.sin(angles)
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    return np.stack([cx + x * cos_r - y * sin_r, cy + x * sin_r + y * cos_r], -1)


def capsule_polygon(start: np.ndarray, end: np.ndarray, radius: float, n_points: int = 16) -> np.ndarray:
    """
    Polygon for a line segment with rounded ends.

    :param start: Segment start (x, y).
    :param end: Segment end (x, y).
    :param radius: Half-thickness.
    :param n_points: Vertices per end cap.
    :return: (n, 2) array.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    axis_angle = np.arctan2(*(end - start)[::-1])
    normal = np.array([-np.sin(axis_angle), np.cos(axis_angle)]) * radius
    start_cap = arc_points(start, radius, axis_angle + np.pi / 2, axis_angle + 3 * np.pi / 2, n_points)
    end_cap = arc_points(end, radius, axis_angle - np.pi / 2, axis_angle + np.pi / 2, n_points)
    return np.concatenate([[start + normal], start_cap, [start - normal], [end - normal], end_cap, [end + normal]])


def arc_points(centre: np.ndarray, radius: float, angle_start: float, angle_end: float, n_points: int) -> np.ndarray:
    """
    Points along a circular arc.

    :param centre: Arc centre (x, y).
    :param radius: Arc radius.
    :param angle_start: Start angle in radians.
    :param angle_end: End angle in radians.
    :param n_points: Number of points.
    :return: (n_points, 2) array.
    """
    angles = np.linspace(angle_start, angle_end, n_points)
    return np.stack([centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles)], -1)
