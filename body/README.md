# Giving the brain a body

The proboscis in `make_movie.py` and on the interactive page is a **kinematic rig**: MN9's
firing rate, low-pass filtered, sets two joint angles. It shows you the motor command. It is
not physics — there is no mass, no contact, no gravity, nothing that could fail.

For an actual embodied fly you want MuJoCo. Two options, both installable with `uv`:

| | |
|---|---|
| [FlyGym / NeuroMechFly v2](https://neuromechfly.org) | `uv pip install "flygym"` — micro-CT-based adult female, compound-eye vision, olfaction, leg adhesion, built for closed-loop control |
| [flybody](https://github.com/TuragaLab/flybody) | DeepMind + Janelia MuJoCo fly with RL-trained walking and flight policies |

**Neither could be installed in the session where this project was built** (PyPI was blocked by
network policy), so `flygym_per.py` is written against the published FlyGym API but has never
been executed. Treat it as a starting point, not working code — expect to fix at least the
joint names, which differ between FlyGym releases.

## Running it

```bash
cd flybrain-sim
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt "flygym"
python body/flygym_per.py --seconds 2.4
```

The script prints the actuated DoFs FlyGym exposes on the model before it does anything else.
If no proboscis/rostrum/haustellum joints appear in that list, your FlyGym build doesn't
actuate the mouthparts, and the honest options are to drive a head-pitch DoF as a stand-in or
to add the joints to the MJCF yourself.

## What the coupling actually is

```
sugar GRNs fire  ->  [138,639-neuron LIF network, FlyWire wiring]  ->  MN9 spike rate
MN9 spike rate   ->  low-pass filter (tau ~ 60 ms, muscle + cuticle)  ->  joint targets
```

That second arrow is the honest weak point. Real MN9 output drives muscle 9 through a
neuromuscular junction with its own dynamics, and the proboscis is a hydraulic-ish linkage,
not a servo. A first-order filter onto position targets is the crudest defensible choice;
a Hill-type muscle model would be the next step.
