"""Download the official NRDS Alloy 617 EDS DAT stack with provenance hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen


BASE = "https://nrds.inl.gov"
ELEMENTS = ["Al", "Cl", "Co", "Cr", "F", "Fe", "Mg", "Mn", "Mo", "Na", "Ni", "O", "Si", "Ti", "W"]
PAGES = {
    "CPS": f"{BASE}/dataset/a617_test6-7_images_ebsd___eds_all_elements",
    **{
        element: f"{BASE}/dataset/a617_test6-7_images_ebsd___eds_{element.lower()}"
        for element in ELEMENTS
    },
}
SLICE_RE = re.compile(r"ebsd-sliceimage-(\d{3})", re.IGNORECASE)
DOWNLOAD_RE = re.compile(r'href="([^"]+/download/[^"]+\.dat)"', re.IGNORECASE)


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "VEER research download/0.2"})
    with urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8", errors="replace")


def discover(output_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for channel, page in PAGES.items():
        html = fetch_text(page)
        urls = sorted(set(DOWNLOAD_RE.findall(html)))
        matches = []
        for url in urls:
            filename = url.rsplit("/", 1)[-1]
            if channel == "CPS" and "_cps.dat" not in filename.lower():
                continue
            if channel != "CPS" and "_cps.dat" in filename.lower():
                continue
            slice_match = SLICE_RE.search(filename)
            if slice_match is not None:
                matches.append((slice_match.group(1), url, filename))
        unique = {slice_id: (url, filename) for slice_id, url, filename in matches}
        if len(unique) != 265:
            raise RuntimeError(f"{channel}: expected 265 DAT slices, found {len(unique)}")
        for slice_id, (url, filename) in sorted(unique.items()):
            if channel == "CPS":
                target = output_root / "all_elements" / f"slice_{slice_id}" / filename
            else:
                target = output_root / "eds" / f"slice_{slice_id}" / "dat" / filename
            records.append(
                {
                    "slice": slice_id,
                    "channel": channel,
                    "source_page": page,
                    "source_url": url,
                    "local_path": target.as_posix(),
                }
            )
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def has_complete_dat_payload(path: Path) -> bool:
    with path.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        return False
    columns, rows = struct.unpack("<II", header)
    return columns > 0 and rows > 0 and path.stat().st_size >= 4 * (2 + columns * rows)


def download(record: dict[str, str], retries: int = 3) -> dict[str, str | int]:
    target = Path(record["local_path"])
    if not target.exists() or not has_complete_dat_payload(target):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        for attempt in range(1, retries + 1):
            try:
                request = Request(
                    record["source_url"],
                    headers={"User-Agent": "VEER research download/0.2"},
                )
                with urlopen(request, timeout=180) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if not has_complete_dat_payload(temporary):
                    raise ValueError(f"incomplete DAT payload returned for {record['source_url']}")
                temporary.replace(target)
                break
            except Exception:
                if temporary.exists():
                    temporary.unlink()
                if attempt == retries:
                    raise
                time.sleep(attempt * 2)
    return {
        **record,
        "length": target.stat().st_size,
        "sha256": sha256(target),
    }


def write_csv(path: Path, records: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["slice", "channel", "source_page", "source_url", "local_path", "length", "sha256"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/alloy617_nrds"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--discover-only", action="store_true")
    arguments = parser.parse_args()
    plan = discover(arguments.out)
    write_csv(arguments.out / "full_stack_download_plan.csv", plan)
    print(f"Discovered {len(plan)} DAT resources across {len(PAGES)} channels.")
    if arguments.discover_only:
        return
    completed: list[dict[str, str | int]] = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {executor.submit(download, record): record for record in plan}
        for index, future in enumerate(as_completed(futures), start=1):
            completed.append(future.result())
            if index % 100 == 0 or index == len(plan):
                print(f"Downloaded/verified {index}/{len(plan)} resources.")
    completed.sort(key=lambda record: (str(record["slice"]), str(record["channel"])))
    write_csv(arguments.out / "full_stack_download_manifest.csv", completed)
    total = sum(int(record["length"]) for record in completed)
    print(f"Manifest written for {len(completed)} resources ({total / (1024 ** 3):.2f} GiB).")


if __name__ == "__main__":
    main()
