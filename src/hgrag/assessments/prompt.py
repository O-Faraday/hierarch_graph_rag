"""Step 6 - the prompt built from the rebuilt sections: sections left out by the character budget, share of the
budget used, figures sent as images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import Assessment


def assess(msgs: dict[str, Any], *, max_chars: int, max_images: int, send_images: bool = True) -> list[Assessment]:
    """``msgs``: the output of ``generation.build_messages``. Sections are kept best score first; a section that
    does not fit is left out whole (the first one is always kept)."""
    used, dropped = msgs.get("sources") or [], msgs.get("dropped") or []
    chars = len(msgs.get("sources_text") or "")
    use = chars / max_chars if max_chars else 0.0
    figures = [f for s in used for f in s.get("figures") or []]
    on_disk = [f for f in figures if f.get("png_path") and Path(f["png_path"]).is_file()]
    sent = len(msgs.get("images") or []) if send_images else 0
    missing = msgs.get("images_missing") or []
    over_cap = max(0, len(on_disk) - sent) if send_images else 0
    problems = ([f"{len(missing)} figure PNG(s) missing"] if missing else []) + (
        [f"{over_cap} figure(s) beyond the cap of {max_images} images"] if over_cap else [])
    return [
        Assessment("prompt.dropped_sections", len(dropped), passed=not dropped, metadata={"sections": dropped},
                   rationale=f"{len(dropped)} of {len(used) + len(dropped)} sections left out by the budget of "
                             f"{max_chars} characters" + (": " + "; ".join(d[-80:] for d in dropped[:5]) if dropped else "")),
        Assessment("prompt.budget_use", round(use, 3), passed=use <= 1.0, metadata={"chars": chars, "max_chars": max_chars},
                   rationale=f"{chars} characters of sources for a budget of {max_chars}"
                             + (" (the first section alone is over the budget)" if use > 1 else "")),
        Assessment("prompt.images", sent, passed=None if not send_images else not problems,
                   metadata={"figures": len(figures), "on_disk": len(on_disk), "missing": missing, "over_cap": over_cap},
                   rationale=(f"{sent} of {len(figures)} figure(s) of the kept sections sent as images"
                              + (f"; {'; '.join(problems)}" if problems else "")) if send_images
                   else f"images not sent (send_images off): {len(figures)} figure(s) described in text only"),
    ]
