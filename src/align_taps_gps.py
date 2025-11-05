#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
align_taps_gps.py
------------------------------------------------------------
Aligns tap timestamps from tappigraphy sessions with GPS coordinates
for the same participants within ±30 seconds.

Outputs one Parquet file per participant + a combined Parquet dataset.
The code is memory-safe, chunked, and reproducible.
------------------------------------------------------------
"""

import json
import re
import gc
import logging
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import bisect

import numpy as np
import pandas as pd
import dask.dataframe as dd # type: ignore
import pyarrow as pa  # noqa
import pyarrow.parquet as pq  # noqa

warnings.filterwarnings("ignore")

# =========================================================
# 1. Setup and Logging
# =========================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.info("Starting GPS–Tap alignment script...")

# =========================================================
# 2. Define Directories (Reproducible Paths)
# =========================================================
# ⚠️ Adjust these three paths only if your structure differs.
# Everything else derives automatically.

data_root = Path("/data/MoT3/")           # main project folder
data_processed = data_root / "processed"  # contains input parquet
data_output = data_root / "aligned"       # where results will be written

data_processed.mkdir(parents=True, exist_ok=True)
data_output.mkdir(parents=True, exist_ok=True)

# Input files
tap_data_path = data_processed / "taps_processed.parquet"
gps_data_path = data_processed / "GeoLocationData.csv"

# Output files
participant_output_dir = data_output / "participants"
participant_output_dir.mkdir(exist_ok=True)
final_output_path = data_output / "Taps_GPS_aligned_30sec.parquet"

# =========================================================
# 3. Load Data
# =========================================================

logger.info(f"Loading tap data from: {tap_data_path}")
ddf = dd.read_parquet(tap_data_path, engine="pyarrow")

logger.info(f"Loading GPS data from: {gps_data_path}")
dtype_gps = {"ts": str, "partId": str, "latitude": str, "longitude": str, "altitude": str}
gps_chunks = pd.read_csv(
    gps_data_path,
    usecols=lambda c: c != "#",
    dtype=dtype_gps,
    chunksize=10_000,
)

unique_part_ids = set(ddf["partId"].unique().compute().tolist())
logger.info(f"Found {len(unique_part_ids)} unique participant IDs.")

# =========================================================
# 4. Build Per-Participant GPS Dictionaries
# =========================================================

def _safe_split_series(series: pd.Series) -> List[str]:
    """Flatten comma-separated strings across a column."""
    return np.concatenate(series.fillna("").astype(str).str.split(",").to_numpy()).tolist()

all_gps: Dict[str, Dict[str, Any]] = {}

for chunk in gps_chunks:
    chunk = chunk[chunk["partId"].isin(unique_part_ids)]
    if chunk.empty:
        continue

    for pid, gps_pid_df in chunk.groupby("partId"):
        try:
            ts_flat = _safe_split_series(gps_pid_df["ts"])
            lat_flat = _safe_split_series(gps_pid_df["latitude"])
            lon_flat = _safe_split_series(gps_pid_df["longitude"])
            alt_flat = _safe_split_series(gps_pid_df["altitude"])

            n = min(len(ts_flat), len(lat_flat), len(lon_flat), len(alt_flat))
            if n == 0:
                continue

            expanded = pd.DataFrame({
                "ts": pd.to_numeric(ts_flat[:n], errors="coerce"),
                "latitude": pd.to_numeric(lat_flat[:n], errors="coerce"),
                "longitude": pd.to_numeric(lon_flat[:n], errors="coerce"),
                "altitude": pd.to_numeric(alt_flat[:n], errors="coerce"),
            }).dropna()

            expanded["ts"] = expanded["ts"].astype(np.int64)
            expanded.sort_values("ts", inplace=True)

            gps_dict = OrderedDict(
                (int(r.ts), {"latitude": float(r.latitude),
                             "longitude": float(r.longitude),
                             "altitude": float(r.altitude)})
                for r in expanded.itertuples(index=False)
            )

            if pid in all_gps:
                all_gps[pid]["dict"].update(gps_dict)
            else:
                all_gps[pid] = {"dict": gps_dict}

        except Exception as e:
            logger.error(f"Error processing GPS for participant {pid}: {e}")
    del chunk
    gc.collect()

# Cache sorted timestamps per participant
for pid, g in all_gps.items():
    g["keys"] = sorted(g["dict"].keys())

logger.info(f"GPS data processed for {len(all_gps)} participants.")

# =========================================================
# 5. Helper Functions
# =========================================================

def parse_taps(v: Any) -> List[int]:
    """Extract sorted unique integer timestamps from any list/string input."""
    if v is None:
        return []
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        arr = pd.to_numeric(pd.Series(v), errors="coerce").dropna().astype(np.int64).tolist()
    elif isinstance(v, str):
        nums = re.findall(r"\d+", v)
        arr = pd.to_numeric(pd.Series(nums), errors="coerce").dropna().astype(np.int64).tolist()
    else:
        arr = pd.to_numeric(pd.Series([v]), errors="coerce").dropna().astype(np.int64).tolist()
    return sorted(set(arr))

def parse_list_like(v: Any) -> List[Any]:
    """Parse JSON list or comma-separated string."""
    if v is None:
        return []
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        return list(v)
    if isinstance(v, str):
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            try:
                parsed = json.loads(v)
                return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
            except Exception:
                pass
        return [x.strip() for x in v.split(",") if x.strip()]
    return [v]

def find_closest_gps(ts: int, gps_dict: Dict[int, Dict[str, float]], gps_keys: List[int], max_interval: int = 30_000) -> Optional[Dict[str, float]]:
    """Find the closest GPS point to a tap timestamp."""
    if not gps_keys:
        return None
    pos = bisect.bisect_left(gps_keys, ts)
    if pos == 0:
        nearest = gps_keys[0]
    elif pos == len(gps_keys):
        nearest = gps_keys[-1]
    else:
        before, after = gps_keys[pos - 1], gps_keys[pos]
        nearest = after if (after - ts) < (ts - before) else before
    return gps_dict.get(nearest) if abs(nearest - ts) <= max_interval else None

def align_gps_with_sessions(df: pd.DataFrame, gps_data: Dict[str, Dict[str, Any]], max_interval: int = 30_000) -> pd.DataFrame:
    """Aligns taps in each session to GPS coordinates within ±30 seconds."""
    aligned = []
    for _, sess in df.iterrows():
        pid = sess.get("partId")
        sid = sess.get("id", "N/A")
        if pid not in gps_data:
            logger.warning(f"Skipping {sid} ({pid}) — no GPS available.")
            continue
        taps = parse_taps(sess.get("taps"))
        if not taps:
            continue
        gps_dict, gps_keys = gps_data[pid]["dict"], gps_data[pid]["keys"]

        applications = parse_list_like(sess.get("application"))
        categories = parse_list_like(sess.get("category"))
        applications = applications[: len(taps)] or [None] * len(taps)
        categories = categories[: len(taps)] or [None] * len(taps)

        lat, lon, alt = [], [], []
        for t in taps:
            gp = find_closest_gps(t, gps_dict, gps_keys, max_interval)
            if gp:
                lat.append(gp["latitude"])
                lon.append(gp["longitude"])
                alt.append(gp["altitude"])
            else:
                lat.append(None); lon.append(None); alt.append(None)

        aligned.append({
            "partId": pid,
            "session_id": sid,
            "tap_ts": taps,
            "application": applications,
            "category": categories,
            "latitude": lat,
            "longitude": lon,
            "altitude": alt
        })
    return pd.DataFrame(aligned)

# =========================================================
# 6. Align and Export
# =========================================================

logger.info("Starting GPS alignment by participant...")
aligned_files = []

for pid in sorted(unique_part_ids):
    subset = ddf[ddf["partId"] == pid].compute()
    if subset.empty:
        continue
    aligned_df = align_gps_with_sessions(subset, all_gps)
    if not aligned_df.empty:
        out_file = participant_output_dir / f"{pid}_aligned.parquet"
        aligned_df.to_parquet(out_file, index=False)
        aligned_files.append(str(out_file))
        logger.info(f"Saved aligned data for {pid}: {len(aligned_df)} rows.")
    del subset, aligned_df
    gc.collect()

# =========================================================
# 7. Combine Results
# =========================================================

if aligned_files:
    ddf_final = dd.read_parquet(participant_output_dir, engine="pyarrow")
    ddf_final.to_parquet(final_output_path, write_index=False)
    logger.info(f"Final aligned dataset saved to: {final_output_path}")
else:
    logger.warning("No aligned data generated.")

gc.collect()
logger.info("GPS–Tap alignment completed successfully.")
