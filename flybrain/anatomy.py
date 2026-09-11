"""Where each neuron actually sits in the brain.

Soma / anchor coordinates and cell-type annotations for FlyWire v783, from the
whole-brain annotation release of Schlegel et al. (2024),
https://github.com/flyconnectome/flywire_annotations (CC-BY-NC).

Coordinates are FAFB v14.1 voxels: x and y at 4 nm, z at 40 nm.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VOXEL_NM = np.array([4.0, 4.0, 40.0])


class Anatomy:
    """Per-neuron position and annotation, aligned to the Connectome's index order.

    Attributes
    ----------
    xyz_nm   : (N, 3) float32   position in nanometres (soma where known,
                                otherwise the neuron's anchor point)
    has_soma : (N,) bool
    super_class, cell_class, cell_type, side, top_nt : (N,) object arrays
    """

    def __init__(self, connectome, data_dir=DATA_DIR,
                 annotations="flywire_783_annotations.csv.gz"):
        ann = pd.read_csv(Path(data_dir) / annotations, low_memory=False)
        ann = ann.drop_duplicates("root_id").set_index("root_id")
        ann = ann.reindex(connectome.root_ids)

        soma = ann[["soma_x", "soma_y", "soma_z"]].to_numpy(dtype=np.float64)
        anchor = ann[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)
        self.has_soma = np.isfinite(soma).all(axis=1)
        xyz = np.where(self.has_soma[:, None], soma, anchor)
        self.known = np.isfinite(xyz).all(axis=1)
        xyz[~self.known] = np.nan
        self.xyz_nm = (xyz * VOXEL_NM).astype(np.float32)

        for col in ("super_class", "cell_class", "cell_type",
                    "hemibrain_type", "side", "top_nt"):
            setattr(self, col, ann[col].fillna("").to_numpy(dtype=object))
        self.c = connectome

    # -- projections -----------------------------------------------------
    def project(self, view="frontal"):
        """2-D coordinates in micrometres for a standard anatomical view.

        frontal : looking at the fly's face  (x right, y down = ventral)
        dorsal  : looking down on the head   (x right, z down = posterior)
        """
        um = self.xyz_nm / 1000.0
        if view == "frontal":
            return um[:, 0], um[:, 1]
        if view == "dorsal":
            return um[:, 0], um[:, 2]
        if view == "sagittal":
            return um[:, 2], um[:, 1]
        raise ValueError(view)

    def describe(self, idx):
        """One-line description of a model neuron."""
        t = self.cell_type[idx] or self.hemibrain_type[idx] or self.cell_class[idx]
        return f"{t or '?'} ({self.super_class[idx] or '?'}, {self.side[idx] or '?'})"

    def summary(self, idx, rates):
        """Activity grouped by super_class, for a set of neurons."""
        df = pd.DataFrame({"super_class": self.super_class[idx],
                           "rate_hz": rates[idx]})
        g = df.groupby("super_class").agg(n=("rate_hz", "size"),
                                          mean_hz=("rate_hz", "mean"),
                                          max_hz=("rate_hz", "max"))
        return g.sort_values("mean_hz", ascending=False)
