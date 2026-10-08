"""Download Backblaze Drive Stats quarters and unzip them into data/raw/.

Each quarter is one zip of daily CSV files (one row per drive per day).
A quarter is roughly 1 GB zipped, so start with one or two.

Usage:
    python -m src.download                 # quarters from config / .env
    python -m src.download Q1_2024 Q2_2024
"""
import shutil
import sys
import zipfile

import requests

from src.config import DATA_RAW, QUARTERS

BASE_URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_{quarter}.zip"


def download_quarter(quarter: str) -> None:
    target_dir = DATA_RAW / f"data_{quarter}"
    if target_dir.exists() and any(target_dir.rglob("*.csv")):
        print(f"{quarter}: already downloaded, skipping")
        return

    zip_path = DATA_RAW / f"data_{quarter}.zip"
    url = BASE_URL.format(quarter=quarter)
    print(f"{quarter}: downloading {url}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        done = 0
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / total:6.1%} of {total / 1e9:.2f} GB", end="")
    print()

    print(f"{quarter}: unzipping")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)

    # macOS zips carry hidden "__MACOSX/._file.csv" copies that would
    # otherwise be read as data
    for junk in target_dir.rglob("__MACOSX"):
        shutil.rmtree(junk, ignore_errors=True)

    zip_path.unlink()
    n_files = len([p for p in target_dir.rglob("*.csv") if not p.name.startswith("._")])
    print(f"{quarter}: {n_files} daily files ready in {target_dir}")


def main(quarters=None) -> None:
    for quarter in quarters or QUARTERS:
        try:
            download_quarter(quarter)
        except requests.HTTPError as err:
            print(
                f"{quarter}: download failed ({err}). Check the quarter exists at "
                "https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data"
            )
            raise


if __name__ == "__main__":
    main(sys.argv[1:] or None)
