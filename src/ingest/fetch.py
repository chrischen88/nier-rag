"""Wiki acquisition (SPEC §5.1). The corpus is the fixed Fandom dump snapshot; later wiki edits are ignored."""

from __future__ import annotations

from pathlib import Path

import py7zr
import requests


def download_dump(url: str, dest_dir: Path, user_agent: str) -> Path:
    """Download and extract the dump if needed. Returns the path to the XML file."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / url.rsplit("/", 1)[1]
    xml = dest_dir / archive.name.removesuffix(".7z")
    if xml.exists():
        return xml
    if not archive.exists():
        with requests.get(url, headers={"User-Agent": user_agent}, stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = archive.with_suffix(".part")
            with open(tmp, "wb") as f:
                for block in r.iter_content(1 << 16):
                    f.write(block)
            tmp.rename(archive)
    with py7zr.SevenZipFile(archive) as z:
        z.extractall(dest_dir)
    return xml
