#!/usr/bin/env python3
"""Drive any named group of neurons and see what the brain does.

    python run_episode.py --drive sugar:100 --window 0.4-1.1 --readout MN9 --render
    python run_episode.py --drive LC4:200 --name escape --readout DNp01@right --render
    python run_episode.py --drive "ORN_DM4+ORN_DM2:100:0.4-1.1" --drive "ORN_DA2:100:1.5-2.2" --name valence
    python run_episode.py --drive sugar:100 --silence re:^PAM --readout MN9
    python run_episode.py --search DNp

--drive NAME:HZ[:t0-t1[:t0-t1...]]   repeatable; windows default to --window
--readout NAME     the neuron whose rate is the behavioural readout
                   (default: the loudest motor or descending neuron in the drive window)
--silence NAME     zero a group's output synapses (repeatable)
--render           write results/<name>/<name>.mp4 (+ .gif) and an interactive .html

Names resolve against the FlyWire annotations: preset (sugar, bitter, water,
ir94e, MN9), exact cell type (LC4, DNp09, PAM01), re:<regex>, class:<super_class>,
id:<root ids>, joined with +, optionally @left / @right.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from flybrain import Anatomy, Connectome, LIFNetwork, Names, Result, page
from flybrain.logging_setup import configure
from flybrain.render import Drive, render_episode

PROJECT_ROOT = Path(__file__).resolve().parent
RUNAWAY_NEURON_COUNT = 200      # more than this still firing at the end => did not settle
RUNAWAY_MIN_RATE_HZ = 5.0
TAIL_SECONDS = 0.2
OUTPUT_CLASSES = ("motor", "descending")
Window = tuple[float, float]


def parse_window(text: str) -> Window:
    """
    Parse "t0-t1" into a (start, stop) pair of seconds.

    :param text: Window text.
    :return: (start, stop).
    """
    start, stop = text.split("-")
    return float(start), float(stop)


def parse_drive(spec: str, default_window: Window, names: Names) -> Drive:
    """
    Parse a --drive spec into a Drive.

    :param spec: "NAME:HZ[:t0-t1[:t0-t1...]]".
    :param default_window: Window used when the spec has none.
    :param names: Resolver for the neuron group name.
    :return: Drive.
    :raises argparse.ArgumentTypeError: if the spec has no rate.
    """
    parts = spec.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(f"--drive needs NAME:HZ, got {spec!r}")
    name, hz = parts[0], float(parts[1])
    windows = [parse_window(part) for part in parts[2:]] or [default_window]
    return Drive(name=name, indices=names.resolve(name),
                 windows=[(start, stop, hz) for start, stop in windows])


def stimulus_mask(times: np.ndarray, drives: list[Drive]) -> np.ndarray:
    """
    Bins during which any drive is on.

    :param times: Bin start times.
    :param drives: The episode's drives.
    :return: Boolean mask over bins.
    """
    mask = np.zeros(len(times), dtype=bool)
    for drive in drives:
        for t_start, t_stop, _ in drive.windows:
            mask |= (times >= t_start) & (times < t_stop)
    return mask


def choose_readout(anatomy: Anatomy, result: Result, mask: np.ndarray) -> int:
    """
    Pick the loudest motor or descending neuron during the stimulus.

    :param anatomy: Loaded anatomy.
    :param result: Recorded result.
    :param mask: Bins during which the stimulus is on.
    :return: Model index.
    """
    super_class = np.asarray(anatomy.super_class, dtype=object)
    candidates = np.flatnonzero(np.isin(super_class, OUTPUT_CLASSES))
    spikes_in_window = result.binned[0][mask][:, candidates].sum(axis=0)
    if spikes_in_window.max() == 0:
        logger.warning("no motor or descending neuron fired during the stimulus; readout is arbitrary")
        return int(candidates[0])
    return int(candidates[np.argmax(spikes_in_window)])


@dataclass
class ResponderTable:
    table: pd.DataFrame
    rate_columns: list[str]        # one per drive window, then outside, then final tail


def responder_table(connectome: Connectome, anatomy: Anatomy, result: Result,
                    drives: list[Drive], readout: int, mask: np.ndarray) -> ResponderTable:
    """
    Tabulate every neuron that spiked, with its rate in each drive window, outside all windows, and in the final 200 ms.

    :param connectome: Loaded connectome.
    :param anatomy: Loaded anatomy.
    :param result: Recorded result.
    :param drives: The episode's drives.
    :param readout: Model index of the readout neuron.
    :param mask: Bins during which any drive is on.
    :return: ResponderTable sorted by rate in the first drive's window.
    """
    binned = result.binned[0].astype(np.float32)
    times = result.bin_times()
    bin_width = result.bin_width
    rate_columns: dict[str, np.ndarray] = {}
    for drive in drives:
        drive_mask = np.zeros(len(times), dtype=bool)
        for t_start, t_stop, _ in drive.windows:
            drive_mask |= (times >= t_start) & (times < t_stop)
        rate_columns[f"hz_{drive.name}"] = binned[drive_mask].sum(0) / max(1, drive_mask.sum()) / bin_width
    outside = ~mask
    rate_columns["hz_outside"] = (binned[outside].sum(0) / max(1, outside.sum()) / bin_width
                                  if outside.any() else np.zeros(binned.shape[1]))
    tail = times >= (times[-1] - TAIL_SECONDS)
    rate_columns["hz_final_200ms"] = binned[tail].sum(0) / max(1, tail.sum()) / bin_width

    driven = set(np.concatenate([drive.indices for drive in drives]).tolist())
    rows = []
    for index in np.flatnonzero(binned.sum(0) > 0):
        row = {"model_index": int(index), "root_id": int(connectome.root_ids[index]),
               "cell_type": anatomy.type_label(index), "super_class": anatomy.super_class[index],
               "side": anatomy.side[index]}
        for column, values in rate_columns.items():
            row[column] = round(float(values[index]), 1)
        row["driven"] = index in driven
        row["is_readout"] = index == readout
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(f"hz_{drives[0].name}", ascending=False)
    return ResponderTable(table, list(rate_columns))


def log_report(responders: ResponderTable, drives: list[Drive]) -> None:
    """
    Log whether the network settled and the loudest responders per class.

    :param responders: Table from ``responder_table``.
    :param drives: The episode's drives.
    """
    table, rate_columns = responders.table, responders.rate_columns
    still_firing = int((table.hz_final_200ms > RUNAWAY_MIN_RATE_HZ).sum())
    if still_firing > RUNAWAY_NEURON_COUNT:
        logger.warning("{} neurons still above {:.0f} Hz in the final 200 ms: the network did NOT settle. "
                       "Treat only the first ~100 ms of each response as meaningful.",
                       still_firing, RUNAWAY_MIN_RATE_HZ)
    else:
        logger.info("network settled: {} neurons above {:.0f} Hz in the final 200 ms", still_firing, RUNAWAY_MIN_RATE_HZ)

    responded = table[~table.driven]
    logger.info("{} neurons responded that were not driven; by class:\n{}", len(responded),
                responded.groupby("super_class").size().sort_values(ascending=False).to_string())
    window_columns = rate_columns[:-2]
    for super_class in ("motor", "descending", "ascending"):
        subset = responded[responded.super_class == super_class].sort_values(window_columns, ascending=False).head(8)
        if len(subset):
            logger.info("loudest {}:\n{}", super_class,
                        subset[["cell_type", "side"] + rate_columns].to_string(index=False))
    if len(drives) > 1:
        descending = responded[responded.super_class == "descending"]
        for drive in drives:
            column = f"hz_{drive.name}"
            top = descending.sort_values(column, ascending=False).head(5)
            summary = ", ".join(f"{cell_type}({str(side)[:1]}) {hz:.0f}"
                                for cell_type, side, hz in top[["cell_type", "side", column]].to_numpy())
            logger.info("loudest descending during {:<22s} {}", drive.name, summary)


def default_name(drives: list[Drive]) -> str:
    """
    Results folder name derived from the drive names.

    :param drives: The episode's drives.
    :return: Filesystem-safe name.
    """
    return "_".join(drive.name for drive in drives).replace(":", "").replace("+", "-")[:40]


def build_parser() -> argparse.ArgumentParser:
    """
    Command-line interface.

    :return: Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drive", action="append", default=[], metavar="NAME:HZ[:t0-t1]")
    parser.add_argument("--window", default="0.4-1.1", help="default stimulus window (s)")
    parser.add_argument("--silence", action="append", default=[], metavar="NAME")
    parser.add_argument("--readout", default=None, metavar="NAME")
    parser.add_argument("--t-run", type=float, default=None, help="default: last window end + 0.3 s")
    parser.add_argument("--bin", type=float, default=0.02, help="recording bin width (s)")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--name", default=None, help="results/<name>/")
    parser.add_argument("--title", default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-gif", dest="gif", action="store_false")
    parser.add_argument("--body", choices=["auto", "on", "off"], default="auto",
                        help="show the proboscis rig (auto: only when the readout is MN9)")
    parser.add_argument("--search", default=None, metavar="TEXT", help="list cell types containing TEXT and exit")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main() -> None:
    """
    Run one episode from the command line.
    """
    parser = build_parser()
    args = parser.parse_args()
    configure(verbose=args.verbose, log_file=PROJECT_ROOT / "results" / "flybrain.log")

    connectome = Connectome()
    anatomy = Anatomy(connectome)
    names = Names(connectome, anatomy)
    if args.search:
        for cell_type, count in names.search(args.search):
            print(f"{count:6d}  {cell_type}")
        return
    if not args.drive:
        parser.error("at least one --drive is required (or --search TEXT)")

    default_window = parse_window(args.window)
    drives = [parse_drive(spec, default_window, names) for spec in args.drive]
    t_run = args.t_run or (max(t_stop for drive in drives for _, t_stop, _ in drive.windows) + 0.3)
    silence = np.concatenate([names.resolve(spec) for spec in args.silence]) if args.silence else np.zeros(0, dtype=np.int64)
    for drive in drives:
        logger.info("drive   {:<24s} {:5d} neurons   {}", drive.name, len(drive.indices),
                    ", ".join(f"{hz:.0f} Hz @ {t0:.2f}-{t1:.2f}s" for t0, t1, hz in drive.windows))
    for spec in args.silence:
        logger.info("silence {:<24s} {:5d} neurons", spec, len(names.resolve(spec)))

    started = time.time()
    network = LIFNetwork(connectome, seed=args.seed)
    stimulate = {tuple(drive.indices.tolist()): drive.windows for drive in drives}
    result = network.run(stimulate, t_run=t_run, bin_width=args.bin, silence=silence)
    logger.info("{}  ({:.0f}s)", result, time.time() - started)

    mask = stimulus_mask(result.bin_times(), drives)
    readout = int(names.resolve(args.readout)[0]) if args.readout else choose_readout(anatomy, result, mask)
    readout_name = "MN9" if args.readout == "MN9" else anatomy.type_label(readout)
    readout_trace = result.trace(readout)
    logger.info("readout {:<24s} {:6.1f} Hz in window, {:5.1f} Hz outside", readout_name,
                readout_trace[mask].mean(), readout_trace[~mask].mean() if (~mask).any() else 0.0)

    responders = responder_table(connectome, anatomy, result, drives, readout, mask)
    log_report(responders, drives)

    name = args.name or default_name(drives)
    out_dir = PROJECT_ROOT / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    responders.table.to_csv(out_dir / "responders.csv", index=False)
    summary = {
        "name": name,
        "drives": [{"name": d.name, "n": int(len(d.indices)), "windows": d.windows} for d in drives],
        "silence": args.silence,
        "readout": {"name": readout_name, "model_index": readout, "rate_in_window": float(readout_trace[mask].mean())},
        "n_active": int((result.rates > 0).sum()), "t_run": t_run, "seed": args.seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    if args.render:
        show_proboscis = (readout_name == "MN9") if args.body == "auto" else (args.body == "on")
        title = args.title or f"{' + '.join(drive.name for drive in drives)} → {readout_name}"
        render_episode(connectome, anatomy, result, drives, readout, out_dir / name, title,
                       gif=args.gif, body=show_proboscis)
        groups = ", ".join(f"{len(drive.indices)} {drive.name} neurons" for drive in drives)
        standfirst = (f"{groups} are made to fire in the FlyWire connectome model. Everything else "
                      f"in the brain is free to respond; {readout_name} is the readout.")
        page.build(out_dir / f"{name}.json", out_dir / f"{name}.html", title, standfirst,
                   readout_name, drives, show_proboscis)
    logger.success("results in {}", out_dir)


if __name__ == "__main__":
    main()
