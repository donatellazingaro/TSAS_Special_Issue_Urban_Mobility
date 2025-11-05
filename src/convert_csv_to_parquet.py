#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert_csv_to_parquet.py
------------------------------------------------------------
Efficiently converts large CSV files to Parquet format.

Uses Dask for out-of-core parallel processing and chunked
writing, so the entire dataset is never loaded into memory.

Compatible with HPC/cluster workflows.
------------------------------------------------------------
"""

import logging
import dask.dataframe as dd # type: ignore
from pathlib import Path
import time

# =========================================================
# 1. Setup logging
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =========================================================
# 2. Define input and output paths
# =========================================================
# ⚙️ Adjust these paths as needed
data_root = Path("/data/MoT3/")
data_root.mkdir(parents=True, exist_ok=True)

csv_file = data_root / "taps_raw.csv"
parquet_output = data_root / "taps_processed.parquet"

# =========================================================
# 3. Conversion with Dask
# =========================================================
start_time = time.time()

logger.info(f"Starting CSV → Parquet conversion")
logger.info(f"Input CSV:  {csv_file}")
logger.info(f"Output Parquet: {parquet_output}")

# Read the CSV lazily (out-of-core)
try:
    ddf = dd.read_csv(csv_file, blocksize="256MB", assume_missing=True)
    logger.info(f"Columns detected: {ddf.columns.tolist()}")
except Exception as e:
    logger.error(f"Error reading CSV file: {e}")
    raise SystemExit(1)

# Write to Parquet
try:
    ddf.to_parquet(
        parquet_output,
        compression="snappy",
        engine="pyarrow",
        write_index=False,
        schema="infer"
    )
except Exception as e:
    logger.error(f"Error writing Parquet file: {e}")
    raise SystemExit(1)

elapsed = time.time() - start_time
logger.info(f"✅ CSV successfully converted to Parquet!")
logger.info(f"Elapsed time: {elapsed:.2f} seconds")
