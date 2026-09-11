# flybrain-sim — a virtual fly you can poke

A working, dependency-light simulator of the **entire adult *Drosophila* brain**, wired from
the FlyWire connectome. You choose neurons, make them fire, and watch what the other 138,000
do about it.

```
138,639 neurons · 15,091,983 connections · 54,492,920 synapses
```

The worked example is taste → **proboscis extension**: sugar on the labellum makes a real fly
stick out its proboscis to drink. Here, sugar-sensing neurons are driven, and a single motor
neuron (**MN9**) is the behavioural readout.

## Quick start

```bash
python3 run_demo.py --quick        # ~3 min, writes results/taste_to_per.png + CSVs
python3 make_movie.py              # ~2 min, writes results/taste_episode.mp4
jupyter lab notebooks/explore_taste_circuit.ipynb
```

```python
from flybrain import Connectome, LIFNetwork, circuits

conn = Connectome()
net  = LIFNetwork(conn)

sugar = tuple(conn.index(circuits.SUGAR_GRN).tolist())
mn9   = conn.index(circuits.MN9)

res = net.run({sugar: 100.0}, t_run=1.0)      # drive sugar GRNs at 100 Hz
print(res.rate(mn9), "Hz")                    # -> ~60 Hz: the fly would extend
res.table(min_rate=1.0, top=20)               # everything else that lit up
```

## What it reproduces

Running the demo gives you three results, all matching the published analysis:

| Experiment | Result |
|---|---|
| Sugar GRNs, 25 → 200 Hz | MN9 rises 0 → ~105 Hz. Graded, with a threshold around 50 Hz. |
| Bitter / Ir94e GRNs, any rate | MN9 stays silent. The model is not just "poke it and it fires". |
| Water GRNs | Silent below ~150 Hz, then drives MN9 — matching the known higher threshold for water. |
| Sugar 100 Hz + bitter | MN9 falls to zero by ~100 Hz of bitter. **Nothing in the code implements this** — the veto falls out of the wiring. |

![taste to PER](results/taste_to_per.png)

## Watching it happen

`make_movie.py` runs one 2.4-second episode — quiet, then sugar, then quiet, then sugar with
bitter — and renders the brain lighting up next to a proboscis driven by MN9:

![the episode](results/taste_episode.gif)

Every dot is a neuron at its **real position** in the FAFB brain volume. The bright cluster
sits on the ventral midline: the sub-esophageal zone, where fly taste processing actually
lives. Nothing in the code puts it there — that is where those neurons are.

A stimulation schedule is just a list of windows:

```python
res = net.run({sugar:  [(0.4, 1.1, 100.0), (1.5, 2.2, 100.0)],
               bitter: [(1.5, 2.2, 150.0)]}, t_run=2.4, bin_width=0.02)
res.trace(mn9)          # MN9's firing rate over time, in 20 ms bins
```

## What's here

```
flybrain/
  connectome.py   load FlyWire v783 into a signed sparse weight matrix
  lif.py          leaky integrate-and-fire dynamics + stimulate/silence/record API
  circuits.py     named neuron sets: sugar / bitter / water / Ir94e GRNs, MN9
  anatomy.py      soma coordinates and cell typing for every neuron
  viz.py          brain rasteriser + the proboscis rig
  minipq.py       a small Parquet reader (so pyarrow isn't needed)
  tcompact.py     Thrift-compact decoder used by minipq
body/             notes and a (untested) FlyGym/MuJoCo integration script
data/             FlyWire v783 connectivity, neuron list, annotations (~107 MB)
notebooks/        interactive walkthrough
run_demo.py       reproduces the figure above
make_movie.py     renders the episode movie + the data behind the interactive page
results/          CSVs, figures, movies, logs
```

## The body

The proboscis here is a **kinematic rig**, not physics: MN9's rate, low-pass filtered, sets two
joint angles. For a real embodied fly you want MuJoCo — FlyGym/NeuroMechFly or flybody.
`body/flygym_per.py` wires MN9 to a FlyGym proboscis, but it has never been run (no package
index in the environment this was built in), so read `body/README.md` before trusting it.

The only hard dependencies are **numpy, scipy, pandas and matplotlib**. No Brian2, no pyarrow,
no FlyWire account, no GPU. A 1-second whole-brain trial takes roughly 8 seconds on one core.

## The model

Each neuron is one LIF unit (Shiu et al. 2024's parameterisation):

| | |
|---|---|
| resting / reset potential | −52 mV |
| spike threshold | −45 mV |
| membrane time constant | 20 ms |
| synaptic time constant | 5 ms |
| refractory period | 2.2 ms |
| synaptic delay | 1.8 ms |
| **weight per synapse** | **0.275 mV** — the single free parameter |

A connection's weight is its synapse count × 0.275 mV, signed by FlyWire's predicted
neurotransmitter (40% of connections are inhibitory). Stimulated neurons receive Poisson
drive strong enough that each event evokes a spike, so "drive at 100 Hz" means what it says.

## What this model does not know

This matters more than anything above. The anatomy is real; the physiology is assumption.

- **No synaptic strengths.** Synapse count is a proxy for weight. It is a decent one, but a
  two-synapse connection onto a sensitive dendrite can outweigh a twenty-synapse one.
- **No neuromodulation.** Dopamine, octopamine and neuropeptides reconfigure fly behaviour
  constantly — hunger state alone changes the sugar→PER gain enormously. None of it is here.
- **Neurotransmitter predictions are ~87% accurate**, and sign is the single most consequential
  number in the model.
- **No intrinsic diversity.** Every neuron gets identical time constants and threshold.
- **Gap junctions are largely unmapped**, and absent here.
- **No plasticity**, so nothing can learn.
- **Positions are anchor points, not morphology.** Each neuron is drawn as one dot — its soma
  where known (85% of cells), otherwise an anchor point on the arbour. Real neurons are trees
  that span neuropils; a dot lighting up says "this cell fired", not "this region is active".
- **One brain, one sex, one moment.** FlyWire is a single adult female; the ventral nerve cord
  is a separate dataset, so anything below the neck is out of scope.
- **Rates are not calibrated.** Trust *which* neurons respond and the *sign* of an effect.
  Do not trust absolute firing rates or millisecond timing.

Treat every output as a hypothesis you would then test in a fly.

## Where to take it

- **Other circuits.** Any FlyWire root IDs work. Find them in [Codex](https://codex.flywire.ai).
- **A body.** MN9's rate is a motor command with nowhere to go. [FlyGym / NeuroMechFly](https://neuromechfly.org)
  is a micro-CT-based physical fly with vision, olfaction and leg adhesion; [flybody](https://github.com/TuragaLab/flybody)
  is a MuJoCo fly with walking and flight. Closing that loop — connectome brain driving a
  physical body in a task — is the actual frontier, and nobody has done it end to end.
- **Better dynamics.** Fit `w_syn`, or give cell types their own time constants.

## Sources and licence

- Soma positions and cell typing: Schlegel et al., *Nature* (2024), whole-brain annotation
  release, [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations) (CC-BY-NC).
- Connectome data and the LIF parameterisation: Shiu, Sterne, Spiller et al., *Nature* (2024),
  "A leaky integrate-and-fire computational model based on the connectome of the entire adult
  *Drosophila* brain reveals insights into sensorimotor processing"; data files and the
  GRN/MN9 identifications come from
  [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) (MIT).
- FlyWire connectome: Dorkenwald et al. and Schlegel et al., *Nature* (2024). FlyWire data is
  CC-BY-NC; cite FlyWire if you publish from this.
- This code is a clean NumPy/SciPy reimplementation, not a fork of the Brian2 original.
