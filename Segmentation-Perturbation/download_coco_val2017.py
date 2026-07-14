#!/usr/bin/env python3
"""Download and extract the COCO 2017 validation images and instance annotations."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


BASE_URL = "https://huggingface.co/datasets/pcuenq/coco-2017-mirror/resolve/main"
FILES = {
    "val2017.zip": f"{BASE_URL}/val2017.zip?download=true",
    "annotations_trainval2017.zip": (
        f"{BASE_URL}/annotations_trainval2017.zip?download=true"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("coco"),
        help="COCO destination directory (default: ./coco)",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded ZIP archives after successful extraction",
    )
    return parser.parse_args()


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def download(url: str, destination: Path) -> None:
    """Download to a .part file and atomically publish it when complete."""
    if destination.exists():
        print(f"Using existing archive: {destination}")
        return

    part_path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "coco-val2017-downloader/1.0"},
    )
    print(f"Downloading {destination.name} ...")
    try:
        with urllib.request.urlopen(request) as response, part_path.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024
            while True:
                block = response.read(block_size)
                if not block:
                    break
                output.write(block)
                downloaded += len(block)
                if total:
                    percent = 100.0 * downloaded / total
                    message = (
                        f"\r  {human_size(downloaded)} / {human_size(total)} "
                        f"({percent:5.1f}%)"
                    )
                else:
                    message = f"\r  {human_size(downloaded)}"
                print(message, end="", flush=True)
        print()
        part_path.replace(destination)
    except Exception:
        print(f"\nDownload failed. Partial file retained at {part_path}", file=sys.stderr)
        raise


def extract_validation_images(archive: Path, output_dir: Path) -> None:
    image_dir = output_dir / "val2017"
    if image_dir.is_dir() and len(list(image_dir.glob("*.jpg"))) == 5000:
        print(f"Validation images already extracted: {image_dir}")
        return

    print(f"Extracting {archive.name} ...")
    with zipfile.ZipFile(archive) as zf:
        members = [
            info
            for info in zf.infolist()
            if not info.is_dir()
            and info.filename.startswith("val2017/")
            and info.filename.lower().endswith(".jpg")
        ]
        if len(members) != 5000:
            raise RuntimeError(f"Expected 5000 validation images, found {len(members)}")
        zf.extractall(output_dir, members=members)


def extract_instance_annotations(archive: Path, output_dir: Path) -> None:
    member = "annotations/instances_val2017.json"
    destination = output_dir / member
    if destination.is_file():
        print(f"Instance annotations already extracted: {destination}")
        return

    print(f"Extracting {member} ...")
    with zipfile.ZipFile(archive) as zf:
        if member not in zf.namelist():
            raise RuntimeError(f"{member} was not found in {archive}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    archives: dict[str, Path] = {}
    for filename, url in FILES.items():
        path = output_dir / filename
        download(url, path)
        archives[filename] = path

    extract_validation_images(archives["val2017.zip"], output_dir)
    extract_instance_annotations(
        archives["annotations_trainval2017.zip"], output_dir
    )

    image_count = len(list((output_dir / "val2017").glob("*.jpg")))
    annotation_path = output_dir / "annotations" / "instances_val2017.json"
    if image_count != 5000 or not annotation_path.is_file():
        raise RuntimeError("COCO extraction verification failed")

    if not args.keep_archives:
        for archive in archives.values():
            archive.unlink(missing_ok=True)

    print("\nCOCO 2017 validation data is ready:")
    print(f"  Images:      {output_dir / 'val2017'} ({image_count} files)")
    print(f"  Annotations: {annotation_path}")
    print(f"\nRun with: COCO_ROOT={output_dir} ./run_experiment_venv.sh")


if __name__ == "__main__":
    main()
