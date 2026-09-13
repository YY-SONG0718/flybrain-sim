"""Load the FlyWire (v783) whole-brain connectome into a sparse signed weight matrix."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger
from scipy import sparse

from .minipq import ParquetFile

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EDGE_COLUMNS = ("Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity")


def read_edge_table(path: Path) -> dict[str, np.ndarray]:
    """
    Read the three edge columns of the connectivity file.

    Uses pyarrow when it is installed. Otherwise uses the bundled reader, which
    needs a system Brotli library.

    :param path: Path to the Parquet connectivity file.
    :return: Column name -> 1-D array.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.debug("pyarrow not installed; using the bundled Parquet reader")
        parquet = ParquetFile(path)
        return {column: parquet.read_column(column) for column in EDGE_COLUMNS}
    table = pq.read_table(path, columns=list(EDGE_COLUMNS))
    logger.debug("read {} with pyarrow", path.name)
    return {column: table.column(column).to_numpy() for column in EDGE_COLUMNS}


class Connectome:
    """The FlyWire adult-female-brain wiring diagram.

    Attributes
    ----------
    root_ids : (N,) int64      FlyWire root IDs, in model-index order
    completed : (N,) bool      FlyWire's "proofreading complete" flag
    n_neurons : int
    weights : csr_matrix       weights[pre, post] = signed synapse count from pre to post
                               (sign from the FlyWire neurotransmitter prediction)
    n_edges, n_synapses : int
    """

    def __init__(self, data_dir: str | Path = DATA_DIR,
                 completeness_file: str = "Completeness_783.csv",
                 connectivity_file: str = "Connectivity_783.parquet") -> None:
        """
        Load the neuron list and the connectivity table.

        :param data_dir: Folder holding the data files.
        :param completeness_file: CSV of root IDs (model index order) and completion flags.
        :param connectivity_file: Parquet table of signed synapse counts per edge.
        """
        data_dir = Path(data_dir)
        neuron_table = pd.read_csv(data_dir / completeness_file, index_col=0)
        self.root_ids: np.ndarray = neuron_table.index.values.astype(np.int64)
        self.completed: np.ndarray = neuron_table["Completed"].values.astype(bool)
        self.n_neurons: int = len(self.root_ids)
        self._root_id_to_index: dict[int, int] = {
            int(root_id): index for index, root_id in enumerate(self.root_ids)}

        edges = read_edge_table(data_dir / connectivity_file)
        presynaptic = edges["Presynaptic_Index"].astype(np.int64)
        postsynaptic = edges["Postsynaptic_Index"].astype(np.int64)
        signed_synapse_counts = edges["Excitatory x Connectivity"].astype(np.float32)

        self.weights: sparse.csr_matrix = sparse.csr_matrix(
            (signed_synapse_counts, (presynaptic, postsynaptic)),
            shape=(self.n_neurons, self.n_neurons))
        self.n_edges: int = len(presynaptic)
        self.n_synapses: int = int(np.abs(signed_synapse_counts).sum())
        logger.info("loaded connectome: {:,} neurons, {:,} edges, {:,} synapses",
                    self.n_neurons, self.n_edges, self.n_synapses)

    def index(self, root_ids: int | Iterable[int]) -> int | None | np.ndarray:
        """
        Map FlyWire root ID(s) to model index/indices.

        :param root_ids: One root ID, or an iterable of them.
        :return: One index (or None) for a scalar input; an int64 array for an iterable, with unknown IDs dropped.
        """
        if np.isscalar(root_ids):
            return self._root_id_to_index.get(int(root_ids))
        return np.array([self._root_id_to_index[int(root_id)] for root_id in root_ids
                         if int(root_id) in self._root_id_to_index], dtype=np.int64)

    def missing(self, root_ids: Iterable[int]) -> list[int]:
        """
        List the root IDs absent from this release.

        :param root_ids: Root IDs to check.
        :return: Those not in the connectome.
        """
        return [int(root_id) for root_id in root_ids
                if int(root_id) not in self._root_id_to_index]

    def out_degree(self, index: int) -> int:
        """
        Count a neuron's output connections.

        :param index: Model index.
        :return: Number of postsynaptic partners.
        """
        return int(self.weights.indptr[index + 1] - self.weights.indptr[index])

    def downstream(self, root_id: int, top: int = 20) -> pd.DataFrame:
        """
        Strongest direct postsynaptic partners of one neuron.

        :param root_id: FlyWire root ID of the presynaptic neuron.
        :param top: How many partners to return.
        :return: DataFrame with columns root_id and signed_synapses.
        """
        index = self.index(root_id)
        row = self.weights.getrow(index).tocoo()
        order = np.argsort(-np.abs(row.data))[:top]
        return pd.DataFrame({"root_id": self.root_ids[row.col[order]],
                             "signed_synapses": row.data[order]})

    def __repr__(self) -> str:
        return (f"<Connectome {self.n_neurons:,} neurons, {self.n_edges:,} edges, "
                f"{self.n_synapses:,} synapses>")
