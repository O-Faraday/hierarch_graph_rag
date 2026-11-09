"""Download the arXiv paper of the evaluation set into data/pdf_eval/, and check it by its SHA-256.

It is under CC BY 4.0 (see data/pdf_eval/SOURCES.md for the attribution). It is not committed (2.5 MB of PDF), and a
fixed version (v3) is enough to make the set reproducible. The first 16 characters of the hash are the ``doc_id`` the
pipeline gives it. Its headings are run into their paragraphs, with PDF bookmarks: the control of ``outline.py``.

    uv run python scripts/fetch_eval_arxiv.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import requests

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "pdf_eval"
PAPERS = {
    "2404.02228v3.pdf": "838114ca3dae8c3dacf84fd9e4b39b9b8059f2e7c203cb9cf60e94e330011b96",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, sha256 in PAPERS.items():
        target = OUT / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == sha256:
            print("present", target)
            continue
        resp = requests.get(f"https://arxiv.org/pdf/{name.removesuffix('.pdf')}", timeout=120,
                            headers={"user-agent": "hgrag-eval-fetch"})
        resp.raise_for_status()
        if hashlib.sha256(resp.content).hexdigest() != sha256:
            raise SystemExit(f"{name}: unexpected content (SHA-256 differs), not written")
        target.write_bytes(resp.content)
        print("wrote", target)


if __name__ == "__main__":
    main()
