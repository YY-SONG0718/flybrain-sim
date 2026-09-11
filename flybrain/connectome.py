"""Load the FlyWire (v783) whole-brain connectome into sparse matrices."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

try:
    from .minipq import ParquetFile
except ImportError:
    from minipq import ParquetFile

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class Connectome:
    """The FlyWire adult-female-brain wiring diagram as an N x N signed weight matrix.

    Attributes
    ----------
    root_ids : (N,) int64        FlyWire root IDs, in model-index order
    n        : int               number of neurons
    W        : csr_matrix        W[i, j] = signed synapse count from neuron i to neuron j
                                 (sign from the FlyWire neurotransmitter prediction)
    """

    def __init__(self, data_dir=DATA_DIR, completeness="Completeness_783.csv",
                 connectivity="Connectivity_783.parquet"):
        data_dir = Path(data_dir)
        comp = pd.read_csv(data_dir / completeness, index_col=0)
        self.root_ids = comp.index.values.astype(np.int64)
        self.completed = comp["Completed"].values.astype(bool)
        self.n = len(self.root_ids)
        self._id2idx = {int(r): i for i, r in enumerate(self.root_ids)}

        pf = ParquetFile(str(data_dir / connectivity))
        pre = pf.read_column("Presynaptic_Index").astype(np.int64)
        post = pf.read_column("Postsynaptic_Index").astype(np.int64)
        wsyn = pf.read_column("Excitatory x Connectivity").astype(np.float32)

        self.W = sparse.csr_matrix((wsyn, (pre, post)), shape=(self.n, self.n))
        self.n_edges = len(pre)
        self.n_synapses = int(np.abs(wsyn).sum())

    # -- lookup helpers -------------------------------------------------
    def index(self, root_ids):
        """FlyWire root ID(s) -> model index/indices. Unknown IDs are dropped."""
        if np.isscalar(root_ids):
            return self._id2idx.get(int(root_ids))
        return np.array([self._id2idx[int(r)] for r in root_ids
                         if int(r) in self._id2idx], dtype=np.int64)

    def missing(self, root_ids):
        return [int(r) for r in root_ids if int(r) not in self._id2idx]

    def out_degree(self, idx):
        return self.W.indptr[idx + 1] - self.W.indptr[idx]

    def downstream(self, root_id, top=20):
        """Strongest direct postsynaptic partners of one neuron."""
        i = self.index(root_id)
        row = self.W.getrow(i).tocoo()
        order = np.argsort(-np.abs(row.data))[:top]
        return pd.DataFrame({
            "root_id": self.root_ids[row.col[order]],
            "signed_synapses": row.data[order],
        })

    def __repr__(self):
        return (f"<Connectome {self.n:,} neurons, {self.n_edges:,} edges, "
                f"{self.n_synapses:,} synapses>")
