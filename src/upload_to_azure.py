"""Upload the raw and processed data to Azure Blob Storage (the data lake).

Needs AZURE_STORAGE_CONNECTION_STRING in your .env. You'll find it in the
Azure portal under your storage account > Security + networking > Access keys.

Usage:
    python -m src.upload_to_azure            # raw + processed
    python -m src.upload_to_azure processed  # just the processed tables
"""
import sys
from pathlib import Path

from src.config import AZURE_CONTAINER, AZURE_STORAGE_CONNECTION_STRING, DATA_PROCESSED, DATA_RAW, ROOT


def upload_folder(container, folder: Path, prefix: str) -> int:
    count = 0
    for path in sorted(folder.rglob("*")):
        if path.is_dir() or path.name.startswith(".") or path.name.endswith(".crc"):
            continue
        blob_name = f"{prefix}/{path.relative_to(folder).as_posix()}"
        with open(path, "rb") as f:
            container.upload_blob(blob_name, f, overwrite=True, max_concurrency=4)
        count += 1
        if count % 50 == 0:
            print(f"  {count} files uploaded")
    return count


def main(what: str = "all") -> None:
    if not AZURE_STORAGE_CONNECTION_STRING:
        raise SystemExit("Set AZURE_STORAGE_CONNECTION_STRING in .env first (see README).")
    from azure.storage.blob import BlobServiceClient

    service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container = service.get_container_client(AZURE_CONTAINER)
    if not container.exists():
        container.create_container()
        print(f"Created container '{AZURE_CONTAINER}'")

    if what in ("all", "raw"):
        n = upload_folder(container, DATA_RAW, "raw")
        print(f"Uploaded {n} raw files")
    if what in ("all", "processed"):
        n = upload_folder(container, DATA_PROCESSED, "processed")
        n += upload_folder(container, ROOT / "reports", "reports")
        print(f"Uploaded {n} processed and report files")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
