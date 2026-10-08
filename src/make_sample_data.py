"""Create a small simulated dataset in the same format as Backblaze's files.

This is only for testing that the pipeline runs end to end in a few minutes.
The numbers it produces are made up. Never report results from sample data.

Usage:
    python -m src.make_sample_data            # 3,000 drives, two quarters
    python -m src.make_sample_data --drives 500
"""
import argparse
import shutil
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.config import DATA_RAW

MODELS = [
    # model, capacity (TB), base daily failure hazard
    ("ST12000NM0008", 12, 3.0e-4),
    ("ST16000NM001G", 16, 2.2e-4),
    ("ST4000DM000", 4, 4.5e-4),
    ("HGST HMS5C4040BLE640", 4, 1.2e-4),
    ("HGST HUH721212ALN604", 12, 1.0e-4),
    ("WDC WUH721816ALE6L4", 16, 1.4e-4),
    ("TOSHIBA MG07ACA14TA", 14, 2.0e-4),
    ("TOSHIBA MG08ACA16TEY", 16, 1.6e-4),
    ("CT250MX500SSD1", 0.25, 1.0e-4),  # boot SSD, should be filtered out
]
SMART_IDS = [1, 3, 4, 5, 7, 9, 10, 12, 187, 188, 192, 193, 194, 197, 198, 199, 240, 241, 242]
QUARTERS = {"Q1_2024": (date(2024, 1, 1), 91), "Q2_2024": (date(2024, 4, 1), 91)}


def simulate(n_drives: int, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    n_days = sum(days for _, days in QUARTERS.values())

    model_idx = rng.integers(0, len(MODELS), n_drives)
    models = np.array([MODELS[i][0] for i in model_idx])
    capacity_tb = np.array([MODELS[i][1] for i in model_idx])
    base_hazard = np.array([MODELS[i][2] for i in model_idx])
    serials = np.array([f"SIM{100000 + i}" for i in range(n_drives)])

    start_hours = rng.uniform(0, 6 * 365 * 24, n_drives)
    age_factor = 1 + start_hours / (3 * 365 * 24)  # older drives fail more
    hazard = base_hazard * age_factor

    # Day each drive fails (or never, inside the window)
    fail_day = rng.geometric(np.clip(hazard, 1e-6, 1)) - 1
    fail_day = np.where(fail_day < n_days, fail_day, -1)

    # Most failing drives show warning signs first; some fail without warning
    warns = rng.random(n_drives) < 0.65
    lead_days = rng.integers(5, 35, n_drives)
    healthy_realloc = np.where(rng.random(n_drives) < 0.04, rng.integers(1, 8, n_drives), 0)
    temperature = rng.normal(32, 4, n_drives).round()
    power_cycles = rng.integers(2, 60, n_drives)

    shutil.rmtree(DATA_RAW, ignore_errors=True)
    day_offset = 0
    for q_index, (quarter, (start, days)) in enumerate(QUARTERS.items()):
        out_dir = DATA_RAW / f"data_{quarter}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for d in range(days):
            t = day_offset + d
            alive = (fail_day == -1) | (fail_day >= t)
            idx = np.where(alive)[0]
            failing_soon = (fail_day[idx] >= 0) & (fail_day[idx] - t < lead_days[idx]) & warns[idx]
            ramp = np.where(
                failing_soon,
                (lead_days[idx] - (fail_day[idx] - t)) / lead_days[idx],
                0.0,
            )
            noise = rng.poisson(1, len(idx))
            hours = start_hours[idx] + t * 24

            frame = {
                "date": (start + timedelta(days=d)).isoformat(),
                "serial_number": serials[idx],
                "model": models[idx],
                "capacity_bytes": (capacity_tb[idx] * 1e12).astype(np.int64),
                "failure": (fail_day[idx] == t).astype(int),
            }
            raw = {sid: rng.integers(0, 100, len(idx)) for sid in SMART_IDS}
            raw[5] = healthy_realloc[idx] + (ramp * rng.integers(20, 400, len(idx))).astype(int)
            raw[187] = (ramp * rng.integers(0, 30, len(idx))).astype(int)
            raw[188] = (ramp * rng.integers(0, 5, len(idx)) * rng.integers(0, 2, len(idx))).astype(int)
            raw[197] = (ramp * rng.integers(0, 60, len(idx)) + (noise > 3)).astype(int)
            raw[198] = (ramp * rng.integers(0, 40, len(idx))).astype(int)
            raw[199] = (rng.random(len(idx)) < 0.01).astype(int)
            raw[9] = hours.astype(int)
            raw[12] = power_cycles[idx]
            raw[194] = temperature[idx] + rng.integers(-1, 2, len(idx))

            smart_cols = {}
            for sid in SMART_IDS:
                smart_cols[f"smart_{sid}_normalized"] = 100
                smart_cols[f"smart_{sid}_raw"] = raw[sid]

            # The newer quarter uses the newer file layout, with extra columns
            # and a different column order, like the real data does
            if q_index == 0:
                df = pd.DataFrame({**frame, **smart_cols})
            else:
                extra = {
                    "datacenter": "sim1",
                    "cluster_id": 0,
                    "vault_id": 1000,
                    "pod_id": 1,
                    "pod_slot_num": 1,
                    "is_legacy_format": False,
                }
                df = pd.DataFrame({**frame, **extra, **smart_cols})
            df.to_csv(out_dir / f"{(start + timedelta(days=d)).isoformat()}.csv", index=False)
        day_offset += days
        print(f"{quarter}: wrote {days} daily files to {out_dir}")

    failures = int((fail_day >= 0).sum())
    print(f"Simulated {n_drives} drives with {failures} failures. Sample data only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drives", type=int, default=3000)
    simulate(parser.parse_args().drives)
