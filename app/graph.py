"""Task graph rendering (Graphviz DOT) from a runner status."""

from __future__ import annotations

from typing import Any

from hgrag.runner import task_summary

COLORS = {"success": "#2e9e5b", "running": "#74c365", "failed": "#d64545", "pending": "#d8d8d8",
          "cancelled": "#e8a33d", "skipped": "#b8c4d6"}
LIGHT_TEXT = {"pending", "skipped"}


def status_to_dot(status: dict[str, Any]) -> str:
    lines = ['digraph G {', 'rankdir=LR; bgcolor="transparent";',
             'node [shape=box, style="rounded,filled", fontname="Helvetica"];']
    for tid in status["graph"]["tasks"]:
        s = task_summary(status["tasks"][tid])
        label = tid + (f"\\n[{s['n_success']}/{s['n']}]" if s["mapped"] else "") + f"\\n{s['state']}"
        font = "black" if s["state"] in LIGHT_TEXT else "white"
        lines.append(f'"{tid}" [label="{label}", fillcolor="{COLORS.get(s["state"], "#d8d8d8")}", fontcolor="{font}"];')
    for a, b in status["graph"]["edges"]:
        lines.append(f'"{a}" -> "{b}";')
    lines.append("}")
    return "\n".join(lines)
