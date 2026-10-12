"""Download the official ALFWorld text data needed by this experiment.

The URLs are the same assets used by the official ``alfworld-download``
script.  The visual detector/checkpoint assets are intentionally omitted:
this project runs only the official TextWorld environment.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import requests


ASSETS = {
    "json": "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip",
    "pddl": "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip",
    "tw_pddl": "https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip",
}


def download(url: str, target: Path, attempts: int = 5) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"[skip] {target} already exists ({target.stat().st_size} bytes)", flush=True)
        return
    for attempt in range(1, attempts + 1):
        temp = target.with_suffix(target.suffix + ".part")
        try:
            print(f"[download {attempt}/{attempts}] {url}", flush=True)
            # A fresh stream is more reliable than loading 70 MB into memory.
            with requests.get(url, stream=True, timeout=(30, 180), allow_redirects=True) as r:
                r.raise_for_status()
                expected = int(r.headers.get("content-length", "0"))
                with temp.open("wb") as f:
                    done = 0
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            done += len(chunk)
                            if done % (10 * 1024 * 1024) < len(chunk):
                                print(f"  {done / 1024 / 1024:.0f} MiB" + (f" / {expected / 1024 / 1024:.0f} MiB" if expected else ""), flush=True)
            if temp.stat().st_size == 0:
                raise RuntimeError("empty download")
            temp.replace(target)
            return
        except Exception as exc:
            print(f"[retry] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
            if attempt == attempts:
                raise
            time.sleep(min(10, attempt * 2))


def extract(zip_path: Path, destination: Path) -> None:
    marker = destination / (zip_path.stem + ".ok")
    if marker.exists():
        print(f"[skip] {zip_path.name} already extracted", flush=True)
        return
    print(f"[extract] {zip_path.name} -> {destination}", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(destination)
    marker.touch()


def copy_logic(data_dir: Path) -> None:
    logic_dir = data_dir / "logic"
    logic_dir.mkdir(parents=True, exist_ok=True)
    try:
        from alfworld.info import ALFRED_PDDL_PATH, ALFRED_TWL2_PATH
    except Exception as exc:
        raise RuntimeError("alfworld package is required to copy its bundled logic files") from exc
    for source in (Path(ALFRED_PDDL_PATH), Path(ALFRED_TWL2_PATH)):
        target = logic_dir / source.name
        if not target.exists():
            shutil.copy2(source, target)
            print(f"[copy] {source} -> {target}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("ALFWORLD_DATA", "./data"))
    args = parser.parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    cache = data_dir / ".downloads"
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in ASSETS.items():
        archive = cache / Path(url).name
        download(url, archive)
        extract(archive, data_dir)
    copy_logic(data_dir)
    print(f"ALFWorld text data is ready at {data_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
