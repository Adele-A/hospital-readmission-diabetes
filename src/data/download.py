"""Download the UCI Diabetes 130-US dataset (id 296) into ``data/raw/``.

Primary path: download the official zip from UCI and extract the two CSVs
(``diabetic_data.csv`` and ``IDS_mapping.csv``), recursing into nested zips.
Fallback: the ``ucimlrepo`` package (encounters only).

The raw CSVs are gitignored; only ``data/raw/MANIFEST.json`` (url + sha256 + date)
is committed, so the provenance is reproducible without shipping the data.

If the network is unreachable the script stops with instructions to place the two
CSVs in ``data/raw/`` manually -- it never fabricates data.

Run:  python -m src.data.download  [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import date
from pathlib import Path

from src.config import load_config
from src.paths import repo_path


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_all_zips(zip_path: Path, dest: Path) -> None:
    """Extract ``zip_path`` into ``dest``, then recursively extract any nested zips."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    for nested in list(dest.rglob("*.zip")):
        inner = nested.with_suffix("")
        inner.mkdir(exist_ok=True)
        with zipfile.ZipFile(nested) as zf:
            zf.extractall(inner)


def _find_one(root: Path, name: str) -> Path:
    """Find exactly one file called ``name`` under ``root`` (case-insensitive)."""
    matches = [p for p in root.rglob("*") if p.is_file() and p.name.lower() == name.lower()]
    if not matches:
        raise FileNotFoundError(f"{name!r} not found inside the downloaded archive")
    return matches[0]


def _download_via_requests(url: str, out_zip: Path) -> None:
    import requests

    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(out_zip, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)


def _try_ucimlrepo(raw_dir: Path, encounters_name: str) -> bool:
    """Fallback: fetch encounters via ucimlrepo. IDS_mapping is NOT available this way."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        return False
    try:
        ds = fetch_ucirepo(id=296)
    except Exception:  # noqa: BLE001 - network/remote failure, caller handles
        return False
    frame = ds.data.original if ds.data.original is not None else ds.data.features
    frame.to_csv(raw_dir / encounters_name, index=False, encoding="utf-8", lineterminator="\n")
    return True


def _manual_placement_stop(raw_dir: Path, enc: str, ids: str) -> None:
    msg = (
        "\nCould not download the dataset automatically (network unreachable or UCI "
        "changed its layout).\n"
        f"Please place the two files manually into: {raw_dir}\n"
        f"  - {enc}\n  - {ids}\n"
        "Source page: https://archive.ics.uci.edu/dataset/296\n"
        "Then re-run:  python -m src.data.download\n"
    )
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def download(force: bool = False) -> dict:
    cfg = load_config()
    raw_dir = repo_path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    enc_name = cfg["data"]["encounters_csv"]
    ids_name = cfg["data"]["ids_mapping_csv"]
    url = cfg["data"]["uci_zip_url"]

    enc_path = raw_dir / enc_name
    ids_path = raw_dir / ids_name

    if enc_path.exists() and ids_path.exists() and not force:
        print(f"Raw files already present in {raw_dir} (use --force to re-download).")
        return _write_manifest(raw_dir, url, enc_path, ids_path, source="cache")

    used_ucimlrepo = False
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        zip_path = tmp_dir / "uci_296.zip"
        try:
            print(f"Downloading {url}")
            _download_via_requests(url, zip_path)
            _extract_all_zips(zip_path, tmp_dir)
            shutil.copyfile(_find_one(tmp_dir, enc_name), enc_path)
            shutil.copyfile(_find_one(tmp_dir, ids_name), ids_path)
        except Exception as exc:  # noqa: BLE001 - fall back, then stop cleanly
            print(f"Direct download failed: {exc}", file=sys.stderr)
            print("Trying ucimlrepo fallback (encounters only)...", file=sys.stderr)
            used_ucimlrepo = _try_ucimlrepo(raw_dir, enc_name)
            if not (used_ucimlrepo and enc_path.exists() and ids_path.exists()):
                _manual_placement_stop(raw_dir, enc_name, ids_name)

    source = "ucimlrepo" if used_ucimlrepo else url
    manifest = _write_manifest(raw_dir, source, enc_path, ids_path, source_kind="download")
    print(f"Done. Wrote {enc_path.name} and {ids_path.name} to {raw_dir}.")
    return manifest


def _write_manifest(
    raw_dir: Path,
    url: str,
    enc_path: Path,
    ids_path: Path,
    source: str = "download",
    source_kind: str | None = None,
) -> dict:
    manifest = {
        "dataset": "UCI Diabetes 130-US Hospitals for Years 1999-2008 (id 296)",
        "source_url": url,
        "downloaded_on": date.today().isoformat(),
        "source_kind": source_kind or source,
        "files": {
            enc_path.name: {
                "sha256": sha256_of(enc_path),
                "size_bytes": enc_path.stat().st_size,
            },
            ids_path.name: {
                "sha256": sha256_of(ids_path),
                "size_bytes": ids_path.stat().st_size,
            },
        },
    }
    manifest_path = raw_dir / "MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UCI Diabetes 130-US dataset.")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    args = parser.parse_args()
    download(force=args.force)


if __name__ == "__main__":
    main()
