# Giving the brain a body

The proboscis in `make_movie.py` and on the interactive pages is a kinematic rig. MN9's
firing rate, after a low-pass filter, sets two joint angles. The rig has no mass, no contact
and no gravity.

`flygym_per.py` drives a physical proboscis instead. It uses FlyGym 2.1, which bundles the
FlyBody model from Vaxenburg et al. (2025). FlyBody has actuated proboscis joints: the
rostrum pitch and the haustellum pitch. The fly is tethered, so only the proboscis moves.

## Run it

```bash
cd flybrain-sim
source .venv/bin/activate
uv pip install flygym pyarrow
python body/flygym_per.py --seconds 2.4
```

The script writes `results/flygym_per.mp4` and `results/flygym_per.csv`. The CSV has one row
per physics step with the MN9 drive, the filtered extension, and the distance from the head
to the labellum in millimetres.

## What the script does

1. It runs the brain for the sugar / sugar+bitter episode and records MN9's rate.
2. It builds a tethered FlyBody with position actuators on the two proboscis pitch joints.
3. It probes both ends of each joint range for 0.25 s and keeps the end that moves the
   labellum furthest from the head. This removes any guess about the sign of "extension".
4. It steps the physics. At each step the low-passed MN9 rate sets the joint targets between
   rest and full extension.

## The coupling

```
sugar GRNs fire -> LIF network on the FlyWire wiring -> MN9 spike rate
MN9 spike rate  -> low-pass filter, tau = 60 ms      -> joint angle targets
```

The second arrow is the weak point. Real MN9 output drives muscle 9 through a neuromuscular
junction with its own dynamics. A first-order filter onto position targets is the simplest
defensible choice. A Hill-type muscle model is the next step.
