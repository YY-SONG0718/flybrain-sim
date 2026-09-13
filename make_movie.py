#!/usr/bin/env python3
"""The original worked example, as a thin wrapper over run_episode.py:

    0.40-1.10  sugar GRNs at 100 Hz          -> the fly should extend
    1.50-2.20  sugar 100 Hz + bitter 150 Hz  -> the veto

Writes results/sugar_bitter_proboscis/{.mp4,.gif,.json,.html} and copies the
movie to results/taste_episode.mp4 for the README. Extra arguments are passed
through to run_episode.py (for example --no-gif or --seed 7).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EPISODE_NAME = "sugar_bitter_proboscis"


def main(extra_args: list[str]) -> None:
    """
    Run the sugar/bitter episode through run_episode.py and copy the movie for the README.

    :param extra_args: Extra arguments passed through to run_episode.py.
    """
    command = [sys.executable, str(PROJECT_ROOT / "run_episode.py"),
               "--drive", "sugar:100:0.4-1.1:1.5-2.2", "--drive", "bitter:150:1.5-2.2",
               "--readout", "MN9", "--t-run", "2.4", "--name", EPISODE_NAME,
               "--title", "A virtual fly tastes sugar, then sugar with bitter", "--render", *extra_args]
    subprocess.run(command, check=True)
    episode_dir = PROJECT_ROOT / "results" / EPISODE_NAME
    for extension in ("mp4", "gif"):
        source = episode_dir / f"{EPISODE_NAME}.{extension}"
        if source.exists():
            shutil.copy(source, PROJECT_ROOT / "results" / f"taste_episode.{extension}")
            print(f"copied to results/taste_episode.{extension}")


if __name__ == "__main__":
    main(sys.argv[1:])
