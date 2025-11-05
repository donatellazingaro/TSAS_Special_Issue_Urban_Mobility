#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
urbanform_features.py
------------------------------------------------------------
Compute street-network features (closeness, betweenness, degree,
orientation entropy, circuity, n_nodes, n_edges) around a representative
GPS point per session, for multiple buffer sizes.

Reproducibility improvements:
- Single, consistent folder layout (data_root/processed, data_root/output, data_root/cache)
- Local graph tile cache; once downloaded, reused deterministically
- Circuity computed in meters (projected CRS), not degrees
- “Nearest node” chosen from nodes INSIDE the buffered subgraph
- Defensive parsing for list-like columns; stable logging; fixed filenames with timestamp

NOTE: OSM is a live dataset; to ensure bit-for-bit reproducibility over time, the
cached graphs in `cache/osm_tiles` must be preserved/committed alongside results.
------------------------------------------------------------
"""

import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
import pickle
import re
import json
import ast

import numpy as np
import pandas as pd
import geopandas as gpd # type: ignore
from shapely.geometry import Point, box # type: ignore

import dask.dataframe as dd # type: ignore
import osmnx as ox # type: ignore
import networkx as nx # type: ignore

warnings.filterwarnings("ignore")

# ------------------------- logging -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("urbanform")


# ------------------------- config --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Compute urban form features around session GPS points.")
    p.add_argument("--data-root", type=Path, default=Path("/data/MoT3/"), help="Project root folder.")
    p.add_argument("--input", type=Path, default=None,
                   help="Aligned taps+GPS parquet. Default: <data_root>/processed/Taps_GPS_aligned_30sec.parquet")
    p.add_argument("--ghsl-shp", type=Path, default=None,
                   help="Optional GHSL DUC shapefile for urban/rural classification.")
    p.add_argument("--tile-size-deg", type=float, default=0.01, help="Tile size in degrees for OSM caching.")
    p.add_argument("--buffers", type=int, nargs="+", default=[200, 500, 1000], help="Buffer sizes in meters.")
    p.add_argument("--network-type", type=str, default="all", help="OSMnx network_type (drive, walk, all, ...).")
    p.add_argument("--npartitions", type=int, default=4, help="Dask partitions for row-wise feature extraction.")
    p.add_argument("--max-workers", type=int, default=0, help="Reserved for future parallel graph ops (unused).")
    p.add_argument("--timestamp", type=str, default=None, help="Fixed timestamp in filenames for reproducibility.")
    return p.parse_args()


# ---------------------- utils: parsing ----------------------
def safe_parse_list(val):
    """Parse list-like (list or stringified list) to Python list of floats with NaN for bad items."""
    if isinstance(val, list):
        return [to_float_or_nan(v) for v in val]
    if isinstance(val, str):
        try:
            val_clean = val.replace("null", "None")
            out = ast.literal_eval(val_clean)
            if isinstance(out, list):
                return [to_float_or_nan(v) for v in out]
        except Exception:
            try:
                parsed = json.loads(val.replace("'", '"'))
                if isinstance(parsed, list):
                    return [to_float_or_nan(v) for v in parsed]
            except Exception:
                pass
    return []


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


def clean_numeric_list(lst):
    return [to_float_or_nan(v) for v in lst]


# ---------------------- utils: tiles ------------------------
def round_tile_center(lat, lon, tile_size_deg):
    lat_tile = round(lat / tile_size_deg) * tile_size_deg
    lon_tile = round(lon / tile_size_deg) * tile_size_deg
    return lat_tile, lon_tile


def tile_id_from_coords(lat, lon, tile_size_deg):
    lat_tile, lon_tile = round_tile_center(lat, lon, tile_size_deg)
    return f"{lat_tile:.5f}_{lon_tile:.5f}", lat_tile, lon_tile


def tile_graph_path(cache_dir: Path, tile_id: str) -> Path:
    return cache_dir / f"tile_{tile_id}.graph.pkl"


def download_and_cache_tile_graph(cache_dir: Path, lat_tile, lon_tile, tile_id, tile_size_deg, network_type="all"):
    """Download OSM graph for tile polygon (if not cached), add bearings, and cache it."""
    path = tile_graph_path(cache_dir, tile_id)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)

    min_lat = lat_tile - tile_size_deg / 2
    max_lat = lat_tile + tile_size_deg / 2
    min_lon = lon_tile - tile_size_deg / 2
    max_lon = lon_tile + tile_size_deg / 2
    poly = box(min_lon, min_lat, max_lon, max_lat)

    try:
        G = ox.graph_from_polygon(poly, network_type=network_type, simplify=True)
        G = ox.bearing.add_edge_bearings(G)
    except Exception as e:
        logger.error(f"[GraphError] tile_id={tile_id} → {e}")
        G = None

    with open(path, "wb") as f:
        pickle.dump(G, f)
    return G


# ---------------------- feature funcs -----------------------
def compute_orientation_entropy(angles, bins=36):
    if not angles:
        return np.nan
    hist, _ = np.histogram(angles, bins=bins, range=(0, 180), density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist)) if hist.size else np.nan


def extract_graph_features(lat, lon, tile_size_deg, cache, cache_dir, buffer_m=500, network_type="all"):
    """
    Compute features using the buffered subgraph around (lat, lon).
    - Nearest node picked FROM the subgraph (not the full tile graph).
    - Circuity computed in meters using projected geometry.
    """
    cols = [
        "closeness", "betweenness", "degree", "orientation_entropy", "circuity",
        "n_nodes", "n_edges", "tile_id", "error"
    ]
    if np.isnan(lat) or np.isnan(lon):
        return pd.Series({k: np.nan for k in cols[:-2]} | {"tile_id": None, "error": "NaN lat/lon"}, index=cols)

    tile_id, lat_tile, lon_tile = tile_id_from_coords(lat, lon, tile_size_deg)

    try:
        if tile_id not in cache:
            cache[tile_id] = download_and_cache_tile_graph(cache_dir, lat_tile, lon_tile, tile_id, tile_size_deg, network_type)
        G = cache[tile_id]
        if not G:
            return pd.Series({k: np.nan for k in cols[:-2]} | {"tile_id": tile_id, "error": "empty_graph"}, index=cols)

        # Convert to GeoDataFrames
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
        if nodes_gdf.empty:
            return pd.Series({k: np.nan for k in cols[:-2]} | {"tile_id": tile_id, "error": "no_nodes"}, index=cols)

        # Project to a local CRS in meters
        nodes_proj = nodes_gdf.to_crs(epsg=3857)  # Web Mercator ~meter units
        point_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(nodes_proj.crs)

        # Build buffer, filter nodes to subgraph
        buffer_poly = point_gdf.geometry.iloc[0].buffer(buffer_m)
        sub_node_index = nodes_proj[nodes_proj.geometry.within(buffer_poly)].index

        if len(sub_node_index) < 1:
            return pd.Series({k: np.nan for k in cols[:-2]} | {"tile_id": tile_id, "error": "empty_subgraph"}, index=cols)

        H = G.subgraph(sub_node_index)

        # Build projected geometry map for circuity in meters
        nodes_proj_map = nodes_proj.geometry  # index aligns with node IDs

        # Pick nearest node INSIDE the subgraph
        # Compute distances in meters from the point to each subgraph node
        try:
            sub_points = nodes_proj_map.loc[list(H.nodes)]
            dists = sub_points.distance(point_gdf.geometry.iloc[0])
            nearest_node = dists.idxmin()
        except Exception:
            nearest_node = next(iter(H.nodes))

        # Centralities (guard small graphs)
        closeness = nx.closeness_centrality(H) if H.number_of_nodes() > 2 else {}
        betweenness = nx.betweenness_centrality(H, normalized=True, endpoints=True) if H.number_of_nodes() > 2 else {}
        degree_dict = dict(H.degree())

        # Edges geodataframe for H (projected for bearings optional, but we use original bearings from G)
        edges_H = ox.graph_to_gdfs(H, edges=True, nodes=False)
        bearings = list(edges_H["bearing"].values) if "bearing" in edges_H.columns else []

        # Circuity in meters: edge length (meters) / straight-line (meters)
        circuity_vals = []
        for u, v, d in H.edges(data=True):
            length_m = d.get("length", np.nan)
            if not np.isfinite(length_m):
                continue
            try:
                p1 = nodes_proj_map.loc[u]
                p2 = nodes_proj_map.loc[v]
                straight_m = p1.distance(p2)
                if straight_m and straight_m > 0:
                    circuity_vals.append(length_m / straight_m)
            except Exception:
                continue

        return pd.Series({
            "closeness": closeness.get(nearest_node, np.nan),
            "betweenness": betweenness.get(nearest_node, np.nan),
            "degree": float(degree_dict.get(nearest_node, np.nan)),
            "orientation_entropy": compute_orientation_entropy(bearings),
            "circuity": float(np.nanmean(circuity_vals)) if len(circuity_vals) else np.nan,
            "n_nodes": int(H.number_of_nodes()),
            "n_edges": int(H.number_of_edges()),
            "tile_id": tile_id,
            "error": np.nan
        }, index=cols)

    except Exception as e:
        return pd.Series({k: np.nan for k in cols[:-2]} | {"tile_id": tile_id, "error": str(e)}, index=cols)


# ------------------------ main -----------------------------
def main():
    args = parse_args()

    # Folder structure
    data_root = args.data_root
    processed_dir = data_root / "processed"
    output_dir = data_root / "output"
    cache_dir = data_root / "cache" / "osm_tiles"

    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Inputs
    input_parquet = args.input or (processed_dir / "Taps_GPS_aligned_30sec.parquet")
    ghsl_shp = args.ghsl_shp  # optional

    # Timestamp for filenames (fixed if provided for reproducibility)
    nowstr = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M")

    # OSMnx config (caching on; log to python logger)
    ox.settings.use_cache = True
    ox.settings.log_console = False

    logger.info(f"Input:  {input_parquet}")
    logger.info(f"GHSL:   {ghsl_shp if ghsl_shp else '(skipped)'}")
    logger.info(f"Cache:  {cache_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Buffers: {args.buffers} (meters), Tile size: {args.tile_size_deg}°")

    # Load data
    if input_parquet.suffix == ".csv":
        df = pd.read_csv(input_parquet)
    else:
        df = pd.read_parquet(input_parquet)

    # Parse coordinates (list-like columns)
    for col in ["latitude", "longitude", "altitude"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_parse_list)
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric_list)

    # Keep rows with at least one valid lat/lon
    df = df[df["latitude"].apply(lambda l: any(np.isfinite(x) for x in l))]
    df = df[df["longitude"].apply(lambda l: any(np.isfinite(x) for x in l))]

    # Representative point = first non-NaN
    df["lat0"] = df["latitude"].apply(lambda l: next((x for x in l if np.isfinite(x)), np.nan))
    df["lon0"] = df["longitude"].apply(lambda l: next((x for x in l if np.isfinite(x)), np.nan))

    # Optional: urban/rural classification
    if ghsl_shp and Path(ghsl_shp).exists():
        logger.info("Adding urban/rural classification via GHSL DUC polygons...")
        ghsl = gpd.read_file(ghsl_shp)
        if ghsl.crs is None:
            logger.warning("GHSL shapefile has no CRS; assuming EPSG:4326.")
            ghsl = ghsl.set_crs(epsg=4326)
        ghsl = ghsl.to_crs(epsg=4326)

        gps_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon0"], df["lat0"]), crs="EPSG:4326")
        gps_with_duc = gpd.sjoin(gps_gdf, ghsl, how="left", predicate="within")

        # A generic indicator—adapt field name as needed (ID_DUC_G0 is in some GHSL layers)
        col_duci = next((c for c in ghsl.columns if c.lower().startswith("id_duc")), None)
        gps_with_duc["is_urban"] = gps_with_duc[col_duci].notna() if col_duci else gps_with_duc.index_right.notna()

        perc_urban = float(gps_with_duc["is_urban"].mean() * 100.0)
        perc_rural = 100.0 - perc_urban
        logger.info(f"Global urban share: {perc_urban:.2f}% (rural {perc_rural:.2f}%)")

        df = pd.DataFrame(gps_with_duc.drop(columns="geometry"))
    else:
        df["is_urban"] = np.nan
        logger.info("No GHSL shapefile provided; skipping urban/rural classification.")

    # Sensitivity analysis over buffers
    summary_frames = []
    for buf in args.buffers:
        logger.info(f"Running features for buffer={buf} m")
        graph_cache = {}

        ddf = dd.from_pandas(df, npartitions=max(1, args.npartitions))

        def _apply_row(row):
            return extract_graph_features(
                row["lat0"], row["lon0"], args.tile_size_deg, graph_cache, cache_dir,
                buffer_m=buf, network_type=args.network_type
            )

        features_ddf = ddf.apply(_apply_row, axis=1, meta=pd.DataFrame)
        features_df = features_ddf.compute()

        result_df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

        out_parquet = output_dir / f"Taps_GPS_urbanform_buf{buf}_{nowstr}.parquet"
        out_csv = output_dir / f"Taps_GPS_urbanform_buf{buf}_{nowstr}.csv"
        result_df.to_parquet(out_parquet, index=False)
        result_df.to_csv(out_csv, index=False)
        logger.info(f"Saved buffer {buf} m results → {out_parquet} & {out_csv}")

        # Summary (basic)
        summary = result_df.describe(include="all").T
        summary["buffer_m"] = buf

        # Urban share
        perc_urban_buf = float(result_df["is_urban"].mean() * 100.0) if "is_urban" in result_df else np.nan
        perc_rural_buf = 100.0 - perc_urban_buf if np.isfinite(perc_urban_buf) else np.nan
        summary.loc["urban_rural_share", "mean"] = perc_urban_buf
        summary.loc["urban_rural_share", "std"] = perc_rural_buf

        summary_frames.append(summary)

    # Combined summary
    summary_all = pd.concat(summary_frames, axis=0)
    summary_out = output_dir / f"Taps_GPS_urbanform_summary_{nowstr}.csv"
    summary_all.to_csv(summary_out)
    logger.info(f"Saved sensitivity + urban share summary → {summary_out}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
