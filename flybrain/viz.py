"""Rendering: the brain lighting up, and a proboscis that follows MN9.

The brain panels are rasterised directly with NumPy (one additive splat per
neuron, then a blur) rather than drawn as 138,000 matplotlib markers, which
keeps a 100-frame movie to a few seconds per panel.
"""
import numpy as np
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap

# --- palette (dark surface) -------------------------------------------------
SURFACE = "#1a1a19"
INK = "#ffffff"
INK2 = "#c3c2b7"
GRID = "#3a3a37"
ACCENT = "#d95926"      # MN9 / the behavioural readout  (categorical slot 2)
SUGAR = "#3987e5"       # stimulus: sugar               (categorical slot 1)
BITTER = "#d95926"
MUTED = "#6f6e68"

# sequential ramp for firing rate: one hue, dark -> light, on a dark surface
ACTIVITY = LinearSegmentedColormap.from_list(
    "activity", ["#12243b", "#1f4f8f", "#3987e5", "#7fb6f2", "#e8f2ff"])


class BrainRaster:
    """Rasterises neuron positions into an image; call `frame()` per timestep."""

    def __init__(self, x, y, width=560, height=None, pad=0.04, flip_y=True):
        ok = np.isfinite(x) & np.isfinite(y)
        self.ok = ok
        x, y = np.asarray(x, float), np.asarray(y, float)
        x0, x1 = np.nanmin(x[ok]), np.nanmax(x[ok])
        y0, y1 = np.nanmin(y[ok]), np.nanmax(y[ok])
        dx, dy = x1 - x0, y1 - y0
        x0 -= pad * dx; x1 += pad * dx
        y0 -= pad * dy; y1 += pad * dy
        aspect = (y1 - y0) / (x1 - x0)
        self.W = width
        self.H = height or max(1, int(round(width * aspect)))
        self.extent = (x0, x1, y0, y1)

        px = (x - x0) / (x1 - x0) * (self.W - 1)
        py = (y - y0) / (y1 - y0) * (self.H - 1)
        if flip_y:
            py = (self.H - 1) - py
        self.px = np.where(ok, px, 0).astype(np.int32)
        self.py = np.where(ok, py, 0).astype(np.int32)
        self.flat = self.py.astype(np.int64) * self.W + self.px

        # static background: every neuron, faint
        bg = np.bincount(self.flat[ok], minlength=self.W * self.H)
        bg = ndimage.gaussian_filter(bg.reshape(self.H, self.W).astype(np.float32), 1.1)
        bg = np.log1p(bg) / np.log1p(bg.max())
        self.bg_rgb = (np.stack([bg] * 3, -1) * np.array([0.19, 0.19, 0.18])) ** 0.9

    def frame(self, idx, values, vmax, sigma=2.4, floor=0.35):
        """RGB image with `values` splatted at the positions of `idx`.

        `floor` is the brightness a neuron gets for spiking at all, so a single
        spike in a single 20 ms bin is still visible against the background.
        """
        img = np.zeros(self.W * self.H, dtype=np.float32)
        m = self.ok[idx] & (values > 0)
        if m.any():
            v = floor + (1 - floor) * np.clip(values[m] / max(vmax, 1e-9), 0, 1)
            np.add.at(img, self.flat[idx[m]], v.astype(np.float32))
        img = ndimage.gaussian_filter(img.reshape(self.H, self.W), sigma)
        # the blur spreads a point source thin; renormalise so one neuron reads
        norm = np.clip(img / (0.055 + 1e-9), 0, 1)
        rgb = ACTIVITY(norm)[..., :3] * np.clip(norm * 1.35, 0, 1)[..., None] ** 0.5
        return np.clip(self.bg_rgb + rgb, 0, 1)

    def marker(self, i):
        """Pixel coordinates of one neuron, for annotating a panel."""
        return float(self.px[i]), float(self.py[i])


# --- the body ---------------------------------------------------------------
def proboscis_polys(extension):
    """Fly head in profile with the proboscis extended by `extension` in [0, 1].

    A kinematic rig, not a physics model: the two proboscis joints (rostrum and
    haustellum) rotate and extend through the range a real fly uses during the
    proboscis extension response, driven here by MN9's firing rate.
    """
    e = float(np.clip(extension, 0, 1))

    head = _ellipse(0.0, 0.0, 1.00, 0.92, n=64)
    eye = _ellipse(0.42, 0.16, 0.46, 0.60, n=48)
    antenna = _ellipse(0.16, -0.40, 0.15, 0.22, n=24)

    # rostrum: swings down and forward out of the head capsule
    a1 = np.deg2rad(-118 + 46 * e)
    L1 = 0.42 + 0.30 * e
    base = np.array([0.02, 0.62])
    j1 = base + L1 * np.array([np.cos(a1), -np.sin(a1)])
    # haustellum: folded back at rest, straightens as the proboscis extends
    a2 = a1 + np.deg2rad(96 - 88 * e)
    L2 = 0.26 + 0.34 * e
    j2 = j1 + L2 * np.array([np.cos(a2), -np.sin(a2)])

    rostrum = _capsule(base, j1, 0.145 - 0.02 * e)
    haustellum = _capsule(j1, j2, 0.105 - 0.015 * e)
    # labellum: the two lobes spread open when the fly is ready to drink
    spread = 0.06 + 0.14 * e
    n2 = np.array([-np.sin(a2), -np.cos(a2)])
    lobeA = _ellipse(*(j2 + spread * n2), 0.15, 0.10, rot=a2, n=28)
    lobeB = _ellipse(*(j2 - spread * n2), 0.15, 0.10, rot=a2, n=28)
    return dict(head=head, eye=eye, antenna=antenna, rostrum=rostrum,
                haustellum=haustellum, lobeA=lobeA, lobeB=lobeB, tip=j2)


def _ellipse(cx, cy, rx, ry, rot=0.0, n=48):
    t = np.linspace(0, 2 * np.pi, n)
    x, y = rx * np.cos(t), ry * np.sin(t)
    c, s = np.cos(rot), np.sin(rot)
    return np.stack([cx + x * c - y * s, cy + x * s + y * c], -1)


def _capsule(p0, p1, r, n=16):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    ang = np.arctan2(d[1], d[0])
    nrm = np.array([-np.sin(ang), np.cos(ang)]) * r
    arc0 = _arc(p0, r, ang + np.pi / 2, ang + 3 * np.pi / 2, n)
    arc1 = _arc(p1, r, ang - np.pi / 2, ang + np.pi / 2, n)
    return np.concatenate([[p0 + nrm], arc0, [p0 - nrm], [p1 - nrm], arc1, [p1 + nrm]])


def _arc(c, r, a0, a1, n):
    t = np.linspace(a0, a1, n)
    return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], -1)
