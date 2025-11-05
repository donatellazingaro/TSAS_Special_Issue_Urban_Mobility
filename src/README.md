# Preprocessing Pipeline for MapOnTap Datasets

This directory documents the preprocessing workflow that was used to
transform raw smartphone behavior and GPS traces into the anonymized,
analysis-ready datasets used in the study.

---

### Data Availability

For privacy and ethical reasons, the raw GPS trajectories and
tappigraphy event streams collected via the MapOnTap platform cannot be
distributed publicly. These data contain sensitive human mobility
information and are protected under the GIVA Lab data governance policy
and ethical approvals.

Accordingly, the scripts in this directory are provided for
transparency and reproducibility in principle, but cannot be executed
end-to-end without authorized access to the restricted raw datasets.

Where possible, anonymized and aggregated derivatives of the final
feature tables are shared separately via the Supplementary Materials and
the OSF repository associated with this project.

---

## Overview of Scripts

Each Python script corresponds to a specific stage of the preprocessing
pipeline, from ingestion to GPS–tap alignment and computation of
urban-form metrics. All scripts are written for scalability and memory
safety (for example, chunked reads and cached OSM network tiles).

---

### `convert_csv_to_parquet.py`

Efficiently converts large CSV logs into Parquet format using Dask.
Supports parallel processing and minimizes memory usage.

---

### `align_taps_gps.py`

Aligns tappigraphy events with the nearest available GPS point within a
±30-second matching window. Produces per-participant aligned datasets
and preserves session structure.

---

### `urbanform_features.py`

Derives street-network metrics at multiple spatial scales using OSMnx
and network analysis. The extracted features include:

- Closeness centrality
- Betweenness centrality
- Degree
- Orientation entropy
- Circuity
- Counts of nodes and edges

Optional integration with GHSL DUC allows classification of sessions
with respect to urbanicity.

---

### `classify_urban_rural.py`

Classifies each session or GPS point as urban or rural using GHSL
polygon data. Adds a binary `is_urban` column to the table and supports
robust parsing of list-like coordinate formats.

---

## Reproducibility

Although the original raw data are protected, the full computational
methodology is openly shared:

- Full source code available in this repository
- Data structures and variable definitions documented in the project
- All analytic results in the manuscript are reproducible using the
shared anonymized derivatives
- Urban-form computation is deterministic once the OSM tile cache is fixed

Researchers seeking controlled access to restricted raw data should
contact the GIVA Lab at the University of Zurich.

---

## Dependencies

- Python 3.10 or later
- pandas, numpy, dask
- geopandas, shapely, osmnx, networkx
- pyarrow (Parquet I/O)

Dependency versions are pinned in `requirements.txt`.