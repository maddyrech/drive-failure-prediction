"""Project settings in one place.

Anything here can be overridden with an environment variable or a line in a
local .env file (see .env.example), so the same code runs on a laptop,
in Docker, or against Azure.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Tiny .env reader so we don't need an extra dependency."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

# Folders
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"

# Which Backblaze quarters to download, e.g. "Q1_2024,Q2_2024"
QUARTERS = [q.strip() for q in os.getenv("QUARTERS", "Q1_2024,Q2_2024,Q3_2024").split(",") if q.strip()]

# The question the model answers: "will this drive fail in the next N days?"
HORIZON_DAYS = int(os.getenv("HORIZON_DAYS", "30"))

# We take one snapshot of the fleet per week, and keep only a share of the
# healthy drive-weeks so the training data fits in memory. Every kept healthy
# row gets weight 1 / rate, so the evaluation still reflects the real fleet.
SNAPSHOT_EVERY_DAYS = 7
NEGATIVE_SAMPLE_RATE = float(os.getenv("NEGATIVE_SAMPLE_RATE", "0.05"))

# Hard drives only. Backblaze also logs boot SSDs, which report SMART very
# differently, so they are left out.
MANUFACTURERS = ["Seagate", "HGST", "WDC", "Toshiba"]

# SMART attributes we use. Backblaze's own research found the first five
# are the strongest early-warning signs of failure.
SMART_SIGNALS = {
    "smart_5_raw": "reallocated_sectors",
    "smart_187_raw": "uncorrectable_errors",
    "smart_188_raw": "command_timeouts",
    "smart_197_raw": "pending_sectors",
    "smart_198_raw": "offline_uncorrectable",
    "smart_199_raw": "crc_errors",
}
SMART_CONTEXT = {
    "smart_9_raw": "power_on_hours",
    "smart_12_raw": "power_cycles",
    "smart_194_raw": "temperature_c",
}

# Database: local Postgres by default, Azure Database for PostgreSQL in the cloud.
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://drives:drives@localhost:5432/drives"
)

# Optional: Azure Blob Storage for the raw data lake
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER", "drive-stats")

for folder in (DATA_RAW, DATA_PROCESSED, FIGURES, MODELS):
    folder.mkdir(parents=True, exist_ok=True)
