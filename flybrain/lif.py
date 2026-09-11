"""Leaky integrate-and-fire dynamics on the FlyWire connectome.

A NumPy/SciPy re-implementation of the whole-brain model of
Shiu, Sterne, Spiller et al. (Nature 2024), "A leaky integrate-and-fire
computational model based on the connectome of the entire adult Drosophila
brain reveals insights into sensorimotor processing".

Every neuron is a single LIF unit; every connection's weight is its synapse
count times a single global free parameter, signed by the FlyWire
neurotransmitter prediction. There are no tuned per-synapse weights,
no neuromodulation and no plasticity -- see README for what that costs you.
"""
import numpy as np

# All times in seconds, voltages in volts, to keep units boring and consistent.
DEFAULT_PARAMS = {
    "v_0":   -52e-3,   # resting potential          (Kakaria & de Bivort 2017)
    "v_rst": -52e-3,   # reset potential after spike
    "v_th":  -45e-3,   # spike threshold
    "t_mbr":  20e-3,   # membrane time constant
    "tau":     5e-3,   # synaptic (alpha) time constant  (Juergensen et al. 2021)
    "t_rfc":   2.2e-3, # refractory period               (Lazar et al. 2021)
    "t_dly":   1.8e-3, # synaptic delay                  (Paul et al. 2015)
    "w_syn":   0.275e-3,  # volts of drive per synapse -- the one free parameter
    "f_poi":   250,    # scaling of the external Poisson drive (250 => every
                       # external event is suprathreshold, so a stimulated
                       # neuron fires at the rate you ask for)
    "dt":      0.1e-3, # integration step
}


class LIFNetwork:
    """Stimulate sets of neurons, integrate, and read out who fired.

    Example
    -------
    >>> net = LIFNetwork(connectome)
    >>> res = net.run(stimulate={sugar_idx: 100.0}, t_run=1.0, n_trials=5)
    >>> res.rate(mn9_idx)
    """

    def __init__(self, connectome, params=None, seed=0):
        self.c = connectome
        self.p = dict(DEFAULT_PARAMS, **(params or {}))
        self.rng = np.random.default_rng(seed)
        W = connectome.W
        self.indptr = W.indptr
        self.indices = W.indices
        self.weights = (W.data * np.float32(self.p["w_syn"])).astype(np.float32)

    # ------------------------------------------------------------------
    def run(self, stimulate, t_run=1.0, n_trials=1, silence=(), bin_width=None,
            progress=False):
        """Run the network.

        Parameters
        ----------
        stimulate : dict {neuron_index or array of indices: rate_in_Hz}
            External Poisson drive. Keys may be a single model index or an
            array of them; the value is the firing rate imposed on each.
        t_run : float          trial duration in seconds
        n_trials : int         repeats, averaged over (the model is stochastic
                               only through the external Poisson drive)
        silence : iterable     model indices whose output synapses are set to 0
        bin_width : float or None
            if set, also record spike counts in bins of this width (seconds),
            available as `Result.binned` -- this is what the movies are made from

        Returns
        -------
        Result
        """
        p = self.p
        dt, n = p["dt"], self.c.n
        n_steps = int(round(t_run / dt))
        delay_steps = max(1, int(round(p["t_dly"] / dt)))
        rfc_steps = int(round(p["t_rfc"] / dt))
        decay_v = np.float32(np.exp(-dt / p["t_mbr"]))
        decay_g = np.float32(np.exp(-dt / p["tau"]))
        poi_kick = np.float32(p["w_syn"] * p["f_poi"])

        # external drive: index array + per-step event probability.
        # A rate may be a constant, or a schedule [(t_start, t_stop, hz), ...].
        stim_idx, stim_sched = [], []
        for k, hz in stimulate.items():
            k = np.atleast_1d(np.asarray(k, dtype=np.int64))
            stim_idx.append(k)
            stim_sched.append(_schedule(hz, n_steps, dt)[:, None].repeat(len(k), 1))
        stim_idx = (np.concatenate(stim_idx) if stim_idx
                    else np.zeros(0, dtype=np.int64))
        # (n_steps, n_stim) probability of an external event
        stim_p = (np.concatenate(stim_sched, axis=1) * dt
                  if len(stim_idx) else np.zeros((n_steps, 0)))

        weights = self.weights
        if len(silence):
            weights = weights.copy()
            for i in np.atleast_1d(np.asarray(silence, dtype=np.int64)):
                weights[self.indptr[i]:self.indptr[i + 1]] = 0.0

        counts = np.zeros((n_trials, n), dtype=np.int32)
        spike_times = []
        bin_steps = max(1, int(round(bin_width / dt))) if bin_width else 0
        n_bins = int(np.ceil(n_steps / bin_steps)) if bin_steps else 0
        binned = (np.zeros((n_trials, n_bins, n), dtype=np.uint8)
                  if bin_steps else None)

        for trial in range(n_trials):
            v = np.full(n, p["v_0"], dtype=np.float32)
            g = np.zeros(n, dtype=np.float32)
            refr = np.zeros(n, dtype=np.int32)
            ring = np.zeros((delay_steps, n), dtype=np.float32)
            trial_spikes = []

            for step in range(n_steps):
                slot = step % delay_steps
                # 1. deliver synaptic input that was released delay_steps ago
                g += ring[slot]
                ring[slot] = 0.0

                # 2. external Poisson drive (bypasses the refractory period,
                #    exactly as in the reference model)
                if len(stim_idx):
                    hit = self.rng.random(len(stim_idx)) < stim_p[step]
                    if hit.any():
                        idx = stim_idx[hit]
                        v[idx] += poi_kick
                        refr[idx] = 0

                # 3. integrate (frozen while refractory)
                active = refr == 0
                g[active] *= decay_g
                v_inf = p["v_0"] + g[active]
                v[active] = v_inf + (v[active] - v_inf) * decay_v
                refr[~active] -= 1

                # 4. threshold
                fired = np.flatnonzero(v > p["v_th"])
                if fired.size:
                    v[fired] = p["v_rst"]
                    g[fired] = 0.0
                    refr[fired] = rfc_steps
                    counts[trial, fired] += 1
                    trial_spikes.append((step * dt, fired))
                    if binned is not None:
                        b = binned[trial, step // bin_steps]
                        np.add.at(b, fired, 1)
                    # 5. schedule postsynaptic input
                    starts, ends = self.indptr[fired], self.indptr[fired + 1]
                    if (ends - starts).sum():
                        sel = _ranges(starts, ends)
                        np.add.at(ring[(slot + delay_steps - 1) % delay_steps],
                                  self.indices[sel], weights[sel])

                if progress and step % 1000 == 0:
                    print(f"  trial {trial} step {step}/{n_steps}", flush=True)

            spike_times.append(trial_spikes)

        return Result(self.c, counts, spike_times, t_run, n_trials, stimulate,
                      binned=binned, bin_width=bin_width)


def _schedule(spec, n_steps, dt):
    """Constant rate or [(t_start, t_stop, hz), ...] -> per-step rate array."""
    if np.isscalar(spec):
        return np.full(n_steps, float(spec))
    out = np.zeros(n_steps)
    for t0, t1, hz in spec:
        out[int(round(t0 / dt)):int(round(t1 / dt))] = float(hz)
    return out


def _ranges(starts, ends):
    """Concatenate arange(s, e) for many (s, e) pairs, vectorised."""
    lens = ends - starts
    keep = lens > 0
    starts, ends, lens = starts[keep], ends[keep], lens[keep]
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    out = np.ones(total, dtype=np.int64)
    out[0] = starts[0]
    if len(starts) > 1:
        out[np.cumsum(lens)[:-1]] = starts[1:] - ends[:-1] + 1
    return np.cumsum(out)


class Result:
    """Firing rates and spike times from one experiment."""

    def __init__(self, connectome, counts, spike_times, t_run, n_trials, stimulate,
                 binned=None, bin_width=None):
        self.c = connectome
        self.counts = counts
        self.spike_times = spike_times
        self.t_run = t_run
        self.n_trials = n_trials
        self.stimulate = stimulate
        self.binned = binned                             # (trials, bins, neurons)
        self.bin_width = bin_width
        self.rates = counts.mean(axis=0) / t_run         # Hz, per neuron
        self.rates_std = counts.std(axis=0) / t_run

    def rate(self, idx):
        return float(self.rates[idx])

    def active(self, min_rate=1.0):
        """Model indices firing above `min_rate` Hz, sorted descending."""
        idx = np.flatnonzero(self.rates >= min_rate)
        return idx[np.argsort(-self.rates[idx])]

    def table(self, min_rate=1.0, top=None, names=None):
        import pandas as pd
        idx = self.active(min_rate)
        if top:
            idx = idx[:top]
        df = pd.DataFrame({
            "root_id": self.c.root_ids[idx],
            "rate_hz": self.rates[idx],
            "std_hz": self.rates_std[idx],
        }, index=idx)
        df.index.name = "model_index"
        if names:
            df.insert(0, "name", [names.get(int(r), "") for r in df.root_id])
        return df

    def trace(self, idx, trial=0):
        """Firing rate over time (Hz) for one neuron, from the binned record."""
        if self.binned is None:
            raise ValueError("run(..., bin_width=...) to record a time course")
        return self.binned[trial, :, idx].astype(np.float32) / self.bin_width

    def bin_times(self):
        return np.arange(self.binned.shape[1]) * self.bin_width

    def raster(self, idx):
        """(times, neuron) arrays for the given indices, trial 0."""
        want = set(int(i) for i in np.atleast_1d(idx))
        ts, ns = [], []
        for t, fired in self.spike_times[0]:
            for f in fired:
                if int(f) in want:
                    ts.append(t); ns.append(int(f))
        return np.array(ts), np.array(ns)

    def __repr__(self):
        return (f"<Result {int((self.rates > 0).sum()):,} neurons active, "
                f"{self.n_trials} trial(s) x {self.t_run}s>")
