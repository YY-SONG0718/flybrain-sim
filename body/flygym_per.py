#!/usr/bin/env python3
"""Drive a MuJoCo fly's proboscis from the connectome model's MN9 output.

    python body/flygym_per.py --seconds 2.4

Written against FlyGym 2.1.0, by reading its source. The FlyBody model in that
release has actuated proboscis joints: the rostrum pitch and the haustellum
pitch. The fly is tethered, so only the proboscis moves.

The brain half is tested: it is the same LIF network as the rest of the
project. The body half ran for the first time on the owner's Mac; this file
records what that run needed.

The pipeline is:

    sugar GRNs fire -> LIF network on the FlyWire wiring -> MN9 spike rate
    MN9 spike rate  -> low-pass filter (muscle)          -> joint angle targets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain import Connectome, LIFNetwork, circuits  # noqa: E402
from flybrain.logging_setup import configure  # noqa: E402

BIN_WIDTH = 0.02             # brain readout bin (s)
TAU_MUSCLE = 0.060           # neuromuscular + cuticle low-pass (s)
MN9_SATURATION_HZ = 90.0     # MN9 rate that corresponds to a fully extended proboscis
SPAWN_POSITION_MM = (0.0, 0.0, 1.5)
CAMERA_OFFSET_MM = (1.2, -3.2, 1.6)
PROBE_SECONDS = 0.25         # settle time when probing which way the proboscis extends


def brain_trace(seconds: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the connectome model for the sugar / sugar+bitter episode.

    :param seconds: Episode length.
    :param seed: Random seed for the external drive.
    :return: (bin times, MN9 rate in Hz, extension command in 0-1).
    """
    connectome = Connectome()
    network = LIFNetwork(connectome, seed=seed)
    sugar = tuple(connectome.index(circuits.SUGAR_GRN).tolist())
    bitter = tuple(connectome.index(circuits.BITTER_GRN).tolist())
    result = network.run(
        {sugar: [(0.15 * seconds, 0.46 * seconds, 100.0), (0.62 * seconds, 0.92 * seconds, 100.0)],
         bitter: [(0.62 * seconds, 0.92 * seconds, 150.0)]},
        t_run=seconds, bin_width=BIN_WIDTH)
    mn9_hz = result.trace(connectome.index(circuits.MN9))
    return result.bin_times(), mn9_hz, np.clip(mn9_hz / MN9_SATURATION_HZ, 0, 1)


def build_simulation():
    """
    Build a tethered FlyBody with position actuators on the proboscis pitch joints.

    :return: (simulation, fly, proboscis joint DoFs in actuator order, joint ranges in radians).
    """
    from flygym import Simulation
    from flygym.compose import ActuatorType, FlyBody, KinematicPosePreset, TetheredWorld
    from flygym.flybody.anatomy_flybody import FlyBodyAxisOrder, FlyBodyJointPreset, FlyBodySkeleton
    from flygym.utils.math import Rotation3D

    fly = FlyBody()
    skeleton = FlyBodySkeleton(axis_order=FlyBodyAxisOrder.YAW_ROLL_PITCH,
                               joint_preset=FlyBodyJointPreset.ALL_BIOLOGICAL)
    joints = fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.FLYBODY_NEUTRAL)
    proboscis_dofs = [dof for dof in joints if dof.child.is_proboscis() and dof.axis.value == "pitch"
                      and dof.child.link in ("rostrum", "haustellum")]
    if not proboscis_dofs:
        raise RuntimeError("FlyBody exposed no proboscis pitch joints; joint names: "
                           + ", ".join(dof.name for dof in joints))
    fly.add_actuators(proboscis_dofs, ActuatorType.POSITION, neutral_input=KinematicPosePreset.FLYBODY_NEUTRAL)
    camera = fly.add_tracking_camera(name="proboscis_cam", pos_offset=CAMERA_OFFSET_MM)

    world = TetheredWorld()
    world.add_fly(fly, spawn_position=SPAWN_POSITION_MM, spawn_rotation=Rotation3D("quat", (1, 0, 0, 0)))
    simulation = Simulation(world)
    simulation.set_renderer(camera, camera_res=(480, 640), playback_speed=1.0, output_fps=30)
    simulation.reset()

    actuated = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    ranges = np.array([[float(joints[dof].range[0]), float(joints[dof].range[1])] for dof in actuated])
    for dof, joint_range in zip(actuated, ranges):
        logger.info("actuated  {:<36s} range {:+.2f} .. {:+.2f} rad", dof.name, *joint_range)
    return simulation, fly, actuated, ranges


def labellum_distance_from_head(simulation, fly) -> float:
    """
    Distance in mm between the head and the tip segment of the proboscis.

    :param simulation: Running simulation.
    :param fly: The fly in it.
    :return: Euclidean distance.
    """
    segments = fly.get_bodysegs_order()
    positions = simulation.get_body_positions(fly.name)
    names = [segment.name for segment in segments]
    tip = "c_labrum" if "c_labrum" in names else "c_haustellum"
    return float(np.linalg.norm(positions[names.index(tip)] - positions[names.index("c_head")]))


def find_extended_targets(simulation, fly, actuated, ranges: np.ndarray) -> np.ndarray:
    """
    Decide which end of each joint range extends the proboscis, by trying both.

    :param simulation: Running simulation.
    :param fly: The fly in it.
    :param actuated: Actuated proboscis DoFs in actuator order.
    :param ranges: (n, 2) joint ranges in radians.
    :return: Target angle per DoF that extends the proboscis furthest.
    """
    from flygym.compose import ActuatorType

    steps = int(PROBE_SECONDS / simulation.timestep)
    distances = {}
    for end in (0, 1):
        simulation.reset()
        simulation.set_actuator_inputs(fly.name, ActuatorType.POSITION, ranges[:, end])
        for _ in range(steps):
            simulation.step()
        distances[end] = labellum_distance_from_head(simulation, fly)
    logger.info("proboscis reach: lower range end {:.3f} mm, upper range end {:.3f} mm", distances[0], distances[1])
    simulation.reset()
    extended_end = 0 if distances[0] > distances[1] else 1
    return ranges[:, extended_end]


def main(seconds: float, seed: int, out_path: str) -> None:
    """
    Drive the MuJoCo fly's proboscis from MN9 and save a video.

    :param seconds: Episode length.
    :param seed: Random seed for the brain model.
    :param out_path: Output video path (a .csv of the command trace is written beside it).
    """
    from flygym.compose import ActuatorType

    _, mn9_hz, extension_command = brain_trace(seconds, seed)
    logger.info("brain: MN9 peaks at {:.0f} Hz, mean {:.1f} Hz over {}s", mn9_hz.max(), mn9_hz.mean(), seconds)

    simulation, fly, actuated, ranges = build_simulation()
    rest_targets = np.zeros(len(actuated))
    extended_targets = find_extended_targets(simulation, fly, actuated, ranges)

    dt = simulation.timestep
    n_steps = int(seconds / dt)
    smoothing = dt / TAU_MUSCLE
    extension = 0.0
    trace_rows = []
    for step in range(n_steps):
        t = step * dt
        command = float(extension_command[min(int(t / BIN_WIDTH), len(extension_command) - 1)])
        extension += smoothing * (command - extension)
        targets = rest_targets + (extended_targets - rest_targets) * extension
        simulation.set_actuator_inputs(fly.name, ActuatorType.POSITION, targets)
        simulation.step()
        simulation.render_as_needed()
        trace_rows.append((t, command, extension, labellum_distance_from_head(simulation, fly)))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    simulation.renderer.save_video(out)
    np.savetxt(out.with_suffix(".csv"), np.array(trace_rows), delimiter=",",
               header="t,mn9_drive,extension,labellum_distance_mm", comments="")
    simulation.close()
    logger.success("wrote {} and {}", out, out.with_suffix(".csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=2.4)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out", default="results/flygym_per.mp4")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    configure(verbose=args.verbose)
    main(args.seconds, args.seed, args.out)
