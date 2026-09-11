#!/usr/bin/env python3
"""Drive a MuJoCo fly's proboscis from the connectome model's MN9 output.

    python body/flygym_per.py --seconds 2.4

NOT TESTED. FlyGym could not be installed in the environment this was written in
(no package-index access), so this is written against the published API and will
probably need small fixes -- most likely the joint names, which differ between
FlyGym releases. Run it once and read the DoF list it prints first.

The brain half of this IS tested: it is the same LIF network as the rest of the
project, and MN9's response to sugar and to sugar+bitter is verified in
results/verify.log.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain import Connectome, LIFNetwork, circuits

BIN = 0.02          # brain readout bin (s)
TAU_MUSCLE = 0.060  # neuromuscular + cuticle low-pass (s)
MN9_SATURATION = 90.0   # Hz that corresponds to a fully extended proboscis

# Candidate joint names, most specific first. FlyGym's MJCF has changed these
# between releases -- the script picks whichever exist and reports what it found.
PROBOSCIS_JOINTS = [
    ("joint_Rostrum", 0.0, 1.05),      # (name, angle at rest, angle extended) radians
    ("joint_Haustellum", 0.0, 0.85),
]
FALLBACK_JOINTS = [
    ("joint_Head", 0.0, 0.35),         # if the mouthparts aren't actuated, pitch the head
]


def brain_trace(seconds, seed):
    """Run the connectome model and return (times, MN9 rate in Hz, extension 0-1)."""
    conn = Connectome()
    net = LIFNetwork(conn, seed=seed)
    S = lambda k: tuple(conn.index(circuits.SETS[k]).tolist())
    half = seconds / 2
    res = net.run(
        {S("sugar"):  [(0.15 * seconds, 0.46 * seconds, 100.0),
                       (0.62 * seconds, 0.92 * seconds, 100.0)],
         S("bitter"): [(0.62 * seconds, 0.92 * seconds, 150.0)]},
        t_run=seconds, bin_width=BIN)
    mn9 = res.trace(conn.index(circuits.MN9))
    return res.bin_times(), mn9, np.clip(mn9 / MN9_SATURATION, 0, 1)


def main(seconds, seed, out):
    from flygym import Fly, Camera, SingleFlySimulation  # noqa: E402

    times, mn9, drive = brain_trace(seconds, seed)
    print(f"brain: MN9 peaks at {mn9.max():.0f} Hz, "
          f"mean {mn9.mean():.1f} Hz over {seconds}s")

    fly = Fly(enable_adhesion=True, spawn_pos=(0, 0, 0.2))
    cam = Camera(attachment_point=fly.model.worldbody, camera_name="camera_front",
                 targeted_fly_names=[fly.name])
    sim = SingleFlySimulation(fly=fly, cameras=[cam])
    obs, info = sim.reset()

    actuated = list(getattr(fly, "actuated_joints", []))
    print(f"\n{len(actuated)} actuated DoFs on this model:")
    for j in actuated:
        print("   ", j)

    joints = [(n, a, b) for (n, a, b) in PROBOSCIS_JOINTS if n in actuated]
    if not joints:
        joints = [(n, a, b) for (n, a, b) in FALLBACK_JOINTS if n in actuated]
        print("\n!! No proboscis DoFs found -- falling back to", [j[0] for j in joints])
    if not joints:
        print("\n!! Nothing to drive. Pick a joint from the list above and put its name "
              "in PROBOSCIS_JOINTS, or add proboscis joints to the MJCF.")
        return
    print("\ndriving:", [j[0] for j in joints])
    slots = {n: actuated.index(n) for n, _, _ in joints}

    dt = sim.timestep
    n_steps = int(seconds / dt)
    ext, k_alpha = 0.0, dt / TAU_MUSCLE
    target = np.array(obs["joints"][0], dtype=float)   # hold the rest pose
    trace = []

    for step in range(n_steps):
        t = step * dt
        cmd = float(drive[min(int(t / BIN), len(drive) - 1)])
        ext += k_alpha * (cmd - ext)                   # muscle low-pass
        for name, rest, full in joints:
            target[slots[name]] = rest + (full - rest) * ext
        obs, reward, term, trunc, info = sim.step({"joints": target})
        sim.render()
        trace.append((t, cmd, ext))
        if term or trunc:
            print(f"episode ended at t={t:.2f}s")
            break

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cam.save_video(out)
    print("wrote", out)
    np.savetxt(out.with_suffix(".csv"), np.array(trace),
               delimiter=",", header="t,mn9_drive,extension", comments="")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2.4)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="results/flygym_per.mp4")
    a = ap.parse_args()
    main(a.seconds, a.seed, a.out)
