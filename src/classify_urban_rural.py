#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
classify_urban_rural.py
------------------------------------------------------------
Classify points as Urban vs Rural using a GHSL DUC polygon layer.

- Reads input CSV/Parquet containing latitude/longitude.
- Accepts scalar or list-like columns (stringified lists ok).
- Picks the first valid coordinate per row as representative.
- Spatial join with GHSL DUC shapefile to set `is_urban` (bool).
- Writes CSV/Parquet with the added `is_urban` column.
------------------------------------------------------------
"""

import argparse
import logging
import warnings
from pathlib import Path
import re
import json
import ast

import numpy as np
import pandas as pd
import geopandas as gpd # type: ignore

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("urban_rural")

# --------------------------- CLI ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Classify points as Urban vs Rural with GHSL DUC.")
    p.add_argument("--input", type=Path, required=True, help="Input CSV/Parquet with latitude/longitude columns.")
    p.add_argument("--ghsl-shp", type=Path, required=True, help="GHSL DUC shapefile (.shp).")
    p.add_argument("--lat-col", type=str, default="latitude", help="Latitude column name.")
    p.add_argument("--lon-col", type=str, default="longitude", help="Longitude column name.")
    p.add_argument("--out", type=Path, required=True, help="Output file path (.csv or .parquet).")
    p.add_argument("--keep-all-cols", action="store_true", help="Keep all input columns in the output.")
    return p.parse_args()

# ------------------------- Helpers -------------------------

def to_float_or_nan(v):
    if v is None:
        return np.nan
    if isinstance(v, (float, int)) and np.isfinite(v):
        return float(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"nan", "null", ""}:
            return np.nan
        if re.match(r"^-?\d+(\.\d+)?$", s):
            try:
                return float(s)
            except Exception:
                return np.nan
    return np.nan

def parse_maybe_list(val):
    """
    Parse list-like (true list or stringified list) into a Python list of floats.
    Returns [] if unparseable. Scalars become [float_or_nan].
    """
    if isinstance(val, list):
        return [to_float_or_nan(v) for v in val]
    if isinstance(val, (float, int)):
        return [to_float_or_nan(val)]
    if isinstance(val, str):
        # try literal list
        try:
            val_clean = val.replace("null", "None")
            out = ast.literal_eval(val_clean)
            if isinstance(out, list):
                return [to_float_or_nan(v) for v in out]
        except Exception:
            pass
        # try JSON list
        try:
            parsed = json.loads(val.replace("'", '"'))
            if isinstance(parsed, list):
                return [to_float_or_nan(v) for v in parsed]
        except Exception:
            pass
        # fallback: single scalar string
        return [to_float_or_nan(val)]
    # other scalars
    return [to_float_or_nan(val)]

def first_valid(seq):
    """Return first finite number from a sequence; else np.nan."""
    for x in seq:
        if np.isfinite(x):
            return float(x)
    return np.nan

# --------------------------- Core --------------------------

def classify_urban_rural(df: pd.DataFrame, ghsl_path: Path, lat_col="latitude", lon_col="longitude") -> pd.DataFrame:
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Columns '{lat_col}' and/or '{lon_col}' not found in input.")

    # parse lat/lon possibly as lists
    lat_parsed = df[lat_col].apply(parse_maybe_list)
    lon_parsed = df[lon_col].apply(parse_maybe_list)

    # representative point per row (first valid)
    df = df.copy()
    df["_lat0"] = lat_parsed.apply(first_valid)
    df["_lon0"] = lon_parsed.apply(first_valid)

    # drop rows with no valid coordinates
    before = len(df)
    df = df[np.isfinite(df["_lat0"]) & np.isfinite(df["_lon0"])]
    after = len(df)
    if after < before:
        logger.info(f"Dropped {before - after} rows without valid lat/lon.")

    # load GHSL and ensure WGS84
    ghsl = gpd.read_file(ghsl_path)
    if ghsl.crs is None:
        logger.warning("GHSL shapefile has no CRS; assuming EPSG:4326.")
        ghsl = ghsl.set_crs(epsg=4326)
    ghsl = ghsl.to_crs(epsg=4326)

    # make GeoDataFrame of points
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["_lon0"], df["_lat0"]), crs="EPSG:4326")

    # spatial join: point within polygon → urban
    joined = gpd.sjoin(gdf, ghsl, how="left", predicate="within")

    # choose an ID-like field if present, else index_right
    duci_col = next((c for c in ghsl.columns if c.lower().startswith("id_duc")), None)
    if duci_col and duci_col in joined.columns:
        joined["is_urban"] = joined[duci_col].notna()
    else:
        joined["is_urban"] = joined["index_right"].notna()

    # clean up
    out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
    return out

# ---------------------------- Main -------------------------

def main():
    args = parse_args()
    logger.info(f"Input: {args.input}")
    logger.info(f"GHSL:  {args.ghsl_shp}")
    logger.info(f"Out:   {args.out}")

    # load input
    if args.input.suffix.lower() == ".csv":
        df_in = pd.read_csv(args.input)
    else:
        df_in = pd.read_parquet(args.input)

    # keep a lightweight set of columns if not requested to keep all
    if args.keep_all_cols:
        df_work = df_in
    else:
        keep = {args.lat_col, args.lon_col}
        # keep identifiers if common names found
        for k in ("partId", "id", "session_id"):
            if k in df_in.columns:
                keep.add(k)
        df_work = df_in[list(keep.intersection(df_in.columns))].copy()

    # classify
    df_out = classify_urban_rural(df_work, args.ghsl_shp, lat_col=args.lat_col, lon_col=args.lon_col)

    # quick summary
    if "is_urban" in df_out:
        share = float(df_out["is_urban"].mean() * 100.0)
        logger.info(f"Urban share: {share:.2f}% (n={len(df_out)})")

    # write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".csv":
        df_out.to_csv(args.out, index=False)
    else:
        df_out.to_parquet(args.out, index=False)

    logger.info("Done.")

if __name__ == "__main__":
    main()
