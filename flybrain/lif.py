"""Leaky integrate-and-fire dynamics on the FlyWire connectome.

A NumPy/SciPy re-implementation of the whole-brain model of Shiu, Sterne,
Spiller et al. (Nature 2024). Every neuron is a single LIF unit; every
connection's weight is its synapse count times one global free parameter,
signed by the FlyWire neurotransmitter prediction. There are no tuned
per-synapse weights, no neuromodulation and no plasticity -- see the README
for what that costs you.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from loguru import logger

from .connectome import Connectome

RateSchedule = Union[float, Sequence[tuple[float, float, float]]]
NeuronGroup = Union[int, Iterable[int]]

# All times in seconds, voltages in volts.
DEFAULT_PARAMS: dict[str, float] = {
    "v_rest": -52e-3,       # resting potential                 (Kakaria & de Bivort 2017)
    "v_reset": -52e-3,      # reset potential after a spike
    "v_threshold": -45e-3,  # spike threshold
    "tau_membrane": 20e-3,  # membrane time constant
    "tau_synapse": 5e-3,    # synaptic (alpha) time constant   (Juergensen et al. 2021)
    "t_refractory": 2.2e-3, # refractory period                (Lazar et al. 2021)
    "t_delay": 1.8e-3,      # synaptic delay                   (Paul et al. 2015)
    "w_synapse": 0.275e-3,  # volts of drive per synapse -- the one free parameter
    "poisson_gain": 250,    # external drive per Poisson event, in units of w_synapse
                            # (250 => every event is suprathreshold, so a stimulated
                            #  neuron fires at exactly the rate you ask for)
    "dt": 0.1e-3,           # integration step
}


def rate_schedule_to_steps(schedule: RateSchedule, n_steps: int, dt: float) -> np.ndarray:
    """
    Expand a constant rate or a list of windows into a per-step rate array.

    :param schedule: Rate in Hz, or [(t_start, t_stop, hz), ...].
    :param n_steps: Number of integration steps.
    :param dt: Step length in seconds.
    :return: Array of length ``n_steps`` with the rate at each step.
    """
    if np.isscalar(schedule):
        return np.full(n_steps, float(schedule))
    per_step = np.zeros(n_steps)
    for t_start, t_stop, hz in schedule:
        per_step[int(round(t_start / dt)):int(round(t_stop / dt))] = float(hz)
    return per_step


def concatenate_ranges(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    Concatenate arange(start, end) for many pairs, without a Python loop.

    :param starts: Range starts.
    :param ends: Range ends (exclusive).
    :return: All the ranges' values, in order.
    """
    lengths = ends - starts
    keep = lengths > 0
    starts, ends, lengths = starts[keep], ends[keep], lengths[keep]
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    steps = np.ones(total, dtype=np.int64)
    steps[0] = starts[0]
    if len(starts) > 1:
        steps[np.cumsum(lengths)[:-1]] = starts[1:] - ends[:-1] + 1
    return np.cumsum(steps)


@dataclass
class ExternalDrive:
    """Poisson drive to a set of neurons, as per-step event probabilities."""
    indices: np.ndarray                   # (n_driven,)
    event_probability: np.ndarray         # (n_steps, n_driven)

    @classmethod
    def from_schedule(cls, stimulate: Mapping[NeuronGroup, RateSchedule],
                      n_steps: int, dt: float) -> "ExternalDrive":
        """
        Build the per-step event probabilities for a stimulation dict.

        :param stimulate: {neuron indices: rate schedule}.
        :param n_steps: Number of integration steps.
        :param dt: Step length in seconds.
        :return: ExternalDrive with one probability column per driven neuron.
        """
        index_blocks: list[np.ndarray] = []
        rate_blocks: list[np.ndarray] = []
        for group, schedule in stimulate.items():
            indices = np.atleast_1d(np.asarray(group, dtype=np.int64))
            per_step_rate = rate_schedule_to_steps(schedule, n_steps, dt)
            index_blocks.append(indices)
            rate_blocks.append(np.repeat(per_step_rate[:, None], len(indices), axis=1))
        if not index_blocks:
            return cls(np.zeros(0, dtype=np.int64), np.zeros((n_steps, 0)))
        return cls(np.concatenate(index_blocks), np.concatenate(rate_blocks, axis=1) * dt)

    def __len__(self) -> int:
        return len(self.indices)


class LIFNetwork:
    """Stimulate sets of neurons, integrate, and read out who fired.

    Example
    -------
    >>> network = LIFNetwork(connectome)
    >>> result = network.run({sugar_indices: 100.0}, t_run=1.0, n_trials=5)
    >>> result.rate(mn9_index)
    """

    def __init__(self, connectome: Connectome, params: Mapping[str, float] | None = None,
                 seed: int = 0) -> None:
        """
        Prepare the network's synaptic weights in volts.

        :param connectome: Loaded connectome.
        :param params: Overrides for DEFAULT_PARAMS.
        :param seed: Seed for the external Poisson drive.
        """
        self.connectome = connectome
        self.params: dict[str, float] = dict(DEFAULT_PARAMS, **(params or {}))
        self.rng = np.random.default_rng(seed)
        weights = connectome.weights
        self._indptr: np.ndarray = weights.indptr
        self._postsynaptic: np.ndarray = weights.indices
        self._weights_volts: np.ndarray = (
            weights.data * np.float32(self.params["w_synapse"])).astype(np.float32)

    def run(self, stimulate: Mapping[NeuronGroup, RateSchedule], t_run: float = 1.0,
            n_trials: int = 1, silence: Iterable[int] = (), bin_width: float | None = None,
            progress: bool = False) -> "Result":
        """
        Run the network and return firing rates and spike times.

        :param stimulate: {neuron indices: rate}. Keys may be a single index or an iterable; a rate is a constant in Hz or a list of (t_start, t_stop, hz) windows.
        :param t_run: Trial duration in seconds.
        :param n_trials: Repeats to average over (stochastic only through the external drive).
        :param silence: Model indices whose output synapses are set to zero.
        :param bin_width: If set, also record spike counts in bins of this width (seconds) as ``Result.binned``.
        :param progress: Log progress every 1000 steps.
        :return: Result.
        """
        p = self.params
        dt = p["dt"]
        n_neurons = self.connectome.n_neurons
        n_steps = int(round(t_run / dt))
        delay_steps = max(1, int(round(p["t_delay"] / dt)))
        refractory_steps = int(round(p["t_refractory"] / dt))
        membrane_decay = np.float32(np.exp(-dt / p["tau_membrane"]))
        synapse_decay = np.float32(np.exp(-dt / p["tau_synapse"]))
        poisson_kick_volts = np.float32(p["w_synapse"] * p["poisson_gain"])
        v_rest = np.float32(p["v_rest"])
        v_reset = np.float32(p["v_reset"])
        v_threshold = np.float32(p["v_threshold"])

        drive = ExternalDrive.from_schedule(stimulate, n_steps, dt)
        weights_volts = self._silenced_weights(silence)

        bin_steps = max(1, int(round(bin_width / dt))) if bin_width else 0
        n_bins = int(np.ceil(n_steps / bin_steps)) if bin_steps else 0
        spike_counts = np.zeros((n_trials, n_neurons), dtype=np.int32)
        binned_counts = (np.zeros((n_trials, n_bins, n_neurons), dtype=np.uint8)
                         if bin_steps else None)
        spike_times: list[list[tuple[float, np.ndarray]]] = []

        logger.debug("run: {} steps x {} trials, {} driven neurons, {} silenced",
                     n_steps, n_trials, len(drive), len(list(silence)))

        for trial in range(n_trials):
            voltage = np.full(n_neurons, v_rest, dtype=np.float32)
            synaptic_drive = np.zeros(n_neurons, dtype=np.float32)
            refractory_left = np.zeros(n_neurons, dtype=np.int32)
            delay_ring = np.zeros((delay_steps, n_neurons), dtype=np.float32)
            trial_spikes: list[tuple[float, np.ndarray]] = []

            for step in range(n_steps):
                slot = step % delay_steps
                # 1. deliver synaptic input released `delay_steps` ago
                synaptic_drive += delay_ring[slot]
                delay_ring[slot] = 0.0

                # 2. external Poisson drive (bypasses the refractory period, as in
                #    the reference model)
                if len(drive):
                    hit = self.rng.random(len(drive)) < drive.event_probability[step]
                    if hit.any():
                        driven_now = drive.indices[hit]
                        voltage[driven_now] += poisson_kick_volts
                        refractory_left[driven_now] = 0

                # 3. integrate (state is frozen while refractory)
                integrating = refractory_left == 0
                synaptic_drive[integrating] *= synapse_decay
                v_target = v_rest + synaptic_drive[integrating]
                voltage[integrating] = v_target + (voltage[integrating] - v_target) * membrane_decay
                refractory_left[~integrating] -= 1

                # 4. threshold, reset, refractory
                fired = np.flatnonzero(voltage > v_threshold)
                if fired.size:
                    voltage[fired] = v_reset
                    synaptic_drive[fired] = 0.0
                    refractory_left[fired] = refractory_steps
                    spike_counts[trial, fired] += 1
                    trial_spikes.append((step * dt, fired))
                    if binned_counts is not None:
                        np.add.at(binned_counts[trial, step // bin_steps], fired, 1)
                    # 5. schedule postsynaptic input
                    edge_starts = self._indptr[fired]
                    edge_ends = self._indptr[fired + 1]
                    edges = concatenate_ranges(edge_starts, edge_ends)
                    if edges.size:
                        target_slot = (slot + delay_steps - 1) % delay_steps
                        np.add.at(delay_ring[target_slot], self._postsynaptic[edges],
                                  weights_volts[edges])

                if progress and step % 1000 == 0:
                    logger.info("trial {} step {}/{}", trial, step, n_steps)

            spike_times.append(trial_spikes)

        return Result(self.connectome, spike_counts, spike_times, t_run, n_trials,
                      dict(stimulate), binned_counts, bin_width)

    def _silenced_weights(self, silence: Iterable[int]) -> np.ndarray:
        """
        Copy of the weights with the silenced neurons' outputs zeroed.

        :param silence: Model indices to silence.
        :return: Weight array in volts (the original if nothing is silenced).
        """
        silence = np.atleast_1d(np.asarray(list(silence), dtype=np.int64))
        if silence.size == 0:
            return self._weights_volts
        weights = self._weights_volts.copy()
        for index in silence:
            weights[self._indptr[index]:self._indptr[index + 1]] = 0.0
        return weights


@dataclass
class Result:
    """Firing rates and spike times from one experiment."""
    connectome: Connectome
    counts: np.ndarray                                  # (trials, neurons) spikes
    spike_times: list[list[tuple[float, np.ndarray]]]   # per trial: [(t, fired indices)]
    t_run: float
    n_trials: int
    stimulate: dict
    binned: np.ndarray | None = None                    # (trials, bins, neurons)
    bin_width: float | None = None
    rates: np.ndarray = field(init=False)               # Hz, per neuron, trial mean
    rates_std: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.rates = self.counts.mean(axis=0) / self.t_run
        self.rates_std = self.counts.std(axis=0) / self.t_run

    def rate(self, index: int) -> float:
        """
        Mean firing rate of one neuron.

        :param index: Model index.
        :return: Rate in Hz.
        """
        return float(self.rates[index])

    def active(self, min_rate_hz: float = 1.0) -> np.ndarray:
        """
        Neurons firing at or above a rate, loudest first.

        :param min_rate_hz: Threshold in Hz.
        :return: Model indices sorted by descending rate.
        """
        indices = np.flatnonzero(self.rates >= min_rate_hz)
        return indices[np.argsort(-self.rates[indices])]

    def table(self, min_rate: float = 1.0, top: int | None = None,
              names: Mapping[int, str] | None = None) -> pd.DataFrame:
        """
        Tabulate the active neurons.

        :param min_rate: Minimum rate in Hz to include.
        :param top: Keep only this many rows.
        :param names: Optional root_id -> label mapping for a name column.
        :return: DataFrame indexed by model index.
        """
        indices = self.active(min_rate)
        if top:
            indices = indices[:top]
        table = pd.DataFrame({"root_id": self.connectome.root_ids[indices],
                              "rate_hz": self.rates[indices],
                              "std_hz": self.rates_std[indices]}, index=indices)
        table.index.name = "model_index"
        if names:
            table.insert(0, "name", [names.get(int(root_id), "") for root_id in table.root_id])
        return table

    def trace(self, index: int, trial: int = 0) -> np.ndarray:
        """
        Firing rate over time for one neuron, from the binned record.

        :param index: Model index.
        :param trial: Trial number.
        :return: Rate in Hz per bin.
        :raises ValueError: if the run did not record bins.
        """
        if self.binned is None or self.bin_width is None:
            raise ValueError("run(..., bin_width=...) to record a time course")
        return self.binned[trial, :, index].astype(np.float32) / self.bin_width

    def bin_times(self) -> np.ndarray:
        """
        Start time of each recorded bin.

        :return: Times in seconds.
        :raises ValueError: if the run did not record bins.
        """
        if self.binned is None or self.bin_width is None:
            raise ValueError("run(..., bin_width=...) to record a time course")
        return np.arange(self.binned.shape[1]) * self.bin_width

    def raster(self, indices: NeuronGroup, trial: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Spike times for a set of neurons.

        :param indices: Model index or indices.
        :param trial: Trial number.
        :return: (spike times, neuron index per spike).
        """
        wanted = set(int(i) for i in np.atleast_1d(indices))
        times: list[float] = []
        neurons: list[int] = []
        for t, fired in self.spike_times[trial]:
            for index in fired:
                if int(index) in wanted:
                    times.append(t)
                    neurons.append(int(index))
        return np.array(times), np.array(neurons)

    def __repr__(self) -> str:
        return (f"<Result {int((self.rates > 0).sum()):,} neurons active, "
                f"{self.n_trials} trial(s) x {self.t_run}s>")
