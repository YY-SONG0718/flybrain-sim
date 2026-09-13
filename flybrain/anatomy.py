"""Where each neuron sits in the brain, and what kind of cell it is.

Soma / anchor coordinates and cell-type annotations for FlyWire v783, from the
whole-brain annotation release of Schlegel et al. (2024),
https://github.com/flyconnectome/flywire_annotations (CC-BY-NC).

Coordinates are FAFB v14.1 voxels: x and y at 4 nm, z at 40 nm.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

from .connectome import Connectome

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VOXEL_SIZE_NM = np.array([4.0, 4.0, 40.0])
ANNOTATION_COLUMNS = ("super_class", "cell_class", "cell_type", "hemibrain_type", "side", "top_nt")

View = Literal["frontal", "dorsal", "sagittal"]


class Anatomy:
    """Per-neuron position and annotation, aligned to the Connectome's index order.

    Attributes
    ----------
    position_nm : (N, 3) float32   soma where known, otherwise the neuron's anchor point
    has_soma, known : (N,) bool
    super_class, cell_class, cell_type, hemibrain_type, side, top_nt : (N,) object arrays
    """

    def __init__(self, connectome: Connectome, data_dir: str | Path = DATA_DIR,
                 annotation_file: str = "flywire_783_annotations.csv.gz") -> None:
        """
        Load positions and annotations aligned to the connectome's index order.

        :param connectome: Loaded connectome.
        :param data_dir: Folder holding the annotation file.
        :param annotation_file: Compressed CSV from the FlyWire annotation release.
        """
        self.connectome = connectome
        annotations = pd.read_csv(Path(data_dir) / annotation_file, low_memory=False)
        annotations = annotations.drop_duplicates("root_id").set_index("root_id")
        annotations = annotations.reindex(connectome.root_ids)

        soma_voxels = annotations[["soma_x", "soma_y", "soma_z"]].to_numpy(dtype=np.float64)
        anchor_voxels = annotations[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)
        self.has_soma: np.ndarray = np.isfinite(soma_voxels).all(axis=1)
        position_voxels = np.where(self.has_soma[:, None], soma_voxels, anchor_voxels)
        self.known: np.ndarray = np.isfinite(position_voxels).all(axis=1)
        position_voxels[~self.known] = np.nan
        self.position_nm: np.ndarray = (position_voxels * VOXEL_SIZE_NM).astype(np.float32)

        for column in ANNOTATION_COLUMNS:
            setattr(self, column, annotations[column].fillna("").to_numpy(dtype=object))
        logger.info("loaded anatomy: positions for {:.1%} of neurons ({:.1%} with a soma)",
                    self.known.mean(), self.has_soma.mean())

    def project(self, view: View = "frontal") -> tuple[np.ndarray, np.ndarray]:
        """
        Project every neuron into a standard 2-D anatomical view, in micrometres.

        :param view: "frontal" (x right, y down), "dorsal" (x right, z down) or "sagittal" (z right, y down).
        :return: (horizontal, vertical) coordinate arrays.
        :raises ValueError: for an unknown view name.
        """
        position_um = self.position_nm / 1000.0
        if view == "frontal":
            return position_um[:, 0], position_um[:, 1]
        if view == "dorsal":
            return position_um[:, 0], position_um[:, 2]
        if view == "sagittal":
            return position_um[:, 2], position_um[:, 1]
        raise ValueError(f"unknown view {view!r}")

    def type_label(self, index: int) -> str:
        """
        Best available cell-type name for one neuron.

        :param index: Model index.
        :return: cell_type, else hemibrain_type, else cell_class, else "?".
        """
        return self.cell_type[index] or self.hemibrain_type[index] or self.cell_class[index] or "?"

    def describe(self, index: int) -> str:
        """
        One-line description such as "LB3 (sensory, left)".

        :param index: Model index.
        :return: Description string.
        """
        return f"{self.type_label(index)} ({self.super_class[index] or '?'}, {self.side[index] or '?'})"

    def summary(self, indices: np.ndarray, rates: np.ndarray) -> pd.DataFrame:
        """
        Group activity by super_class.

        :param indices: Model indices to summarise.
        :param rates: Per-neuron rates in Hz (indexed by model index).
        :return: DataFrame of count, mean and max rate per super_class.
        """
        table = pd.DataFrame({"super_class": self.super_class[indices], "rate_hz": rates[indices]})
        grouped = table.groupby("super_class").agg(n=("rate_hz", "size"),
                                                   mean_hz=("rate_hz", "mean"),
                                                   max_hz=("rate_hz", "max"))
        return grouped.sort_values("mean_hz", ascending=False)
