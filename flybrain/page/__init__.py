"""Assemble the interactive scrubber page for any rendered episode."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from ..render import Drive

TEMPLATE_DIR = Path(__file__).resolve().parent
BAND_COLOURS = ["#3987e5", "#d55181", "#199e70", "#c98500", "#9085e9", "#d95926"]


def legend_html(drives: list["Drive"]) -> str:
    """
    Legend entries, one per drive, coloured to match the trace bands.

    :param drives: The episode's drives.
    :return: HTML fragment.
    """
    return "".join(
        f'<span><i class="swatch" style="background:{BAND_COLOURS[position % len(BAND_COLOURS)]}"></i>'
        f'{drive.name} at {drive.windows[0][2]:.0f} Hz</span>'
        for position, drive in enumerate(drives))


def build(payload_path: str | Path, out_path: str | Path, title: str, standfirst: str,
          readout_name: str, drives: list["Drive"], show_proboscis: bool) -> Path:
    """
    Fill the page template with an episode payload and write a self-contained HTML file.

    :param payload_path: JSON written by ``render_episode``.
    :param out_path: Where to write the page.
    :param title: Page title.
    :param standfirst: Introductory paragraph.
    :param readout_name: Label of the readout neuron.
    :param drives: The episode's drives.
    :param show_proboscis: Show the proboscis rig instead of the outputs panel.
    :return: The written path.
    """
    payload = json.loads(Path(payload_path).read_text())
    payload["readout_name"] = readout_name
    payload["body"] = bool(show_proboscis)

    body_title = (f"Proboscis <em>extension driven by {readout_name}</em>" if show_proboscis
                  else "Outputs <em>loudest motor &amp; descending neurons</em>")
    head = (TEMPLATE_DIR / "head.html").read_text().replace("{{TITLE}}", title)
    body = ((TEMPLATE_DIR / "body.html").read_text()
            .replace("{{TITLE}}", title)
            .replace("{{STANDFIRST}}", standfirst)
            .replace("{{BODY_TITLE}}", body_title)
            .replace("{{READOUT}}", readout_name)
            .replace("{{LEGEND}}", legend_html(drives)))
    script = (TEMPLATE_DIR / "script.html").read_text()
    html = (f"{head}\n{body}\n<script>window.__EPISODE__="
            f"{json.dumps(payload, separators=(',', ':'))};</script>\n{script}")
    out_path = Path(out_path)
    out_path.write_text(html)
    logger.success("wrote {} ({:.1f} MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path
