# Preprocessing Pipeline for MapOnTap Datasets

This directory contains the preprocessing workflow used to prepare the **MapOnTap (MoT)** datasets for analysis.  
Each script performs a specific step in the pipeline — from raw CSV conversion to GPS alignment and the computation of urban form metrics.

All datasets are anonymized and preprocessed derivatives of the original tappigraphy, GPS, and questionnaire data collected by the **GIVA Lab** at the University of Zurich.

---

##  Pipeline Overview

### `convert_csv_to_parquet.py`

Converts large raw CSV files into optimized **Parquet** format using [Dask](https://dask.org/).  
This enables scalable, memory-efficient data processing for subsequent steps.

- Handles large datasets using chunked reads.  
- Uses the `pyarrow` backend for fast I/O.  
- Outputs `optimized_explosion_mot3.parquet` to the `/processed/` directory.

**Usage**
```bash
python convert_csv_to_parquet.py 
```

### `align_taps_gps.py`

Aligns touchscreen interaction timestamps (tappigraphy) with GPS coordinates, linking behavioral and spatial data at the session level.

- Matches each tap timestamp to the nearest GPS point within ±30 seconds.  
- Expands GPS sequences into individual coordinate entries.  
- Produces per-participant aligned Parquet files and a combined dataset.  
- Reports coverage statistics before and after alignment.

**Usage**
```bash
python align_taps_gps.py 
```

### `urbanform_features.py`

Computes street-network metrics for each session’s representative GPS point using **OSMnx**.

**Extracted features include:**
- Closeness  
- Betweenness  
- Degree  
- Orientation Entropy  
- Circuity  
- Number of Nodes  
- Number of Edges  

Performs multi-scale analysis with configurable buffer sizes (default: 200 m, 500 m, 1000 m).  
Optionally integrates GHSL DUC shapefiles to classify sessions as urban or rural.

**Usage**
```bash
python urbanform_features.py \
  --data-root /data/MoT3/ \
  --input /data/MoT3/processed/Taps_GPS_aligned_30sec.parquet \
  --buffers 200 500 1000
```

### `classify_urban_rural.py`

Classifies each GPS point or session as **urban** or **rural** using the GHSL DUC global urban grid.

- Loads aligned GPS data from `Taps_GPS_aligned_30sec.parquet`.  
- Performs spatial join with GHSL polygons.  
- Adds a binary `is_urban` column to the dataset.  
- Exports both CSV and Parquet outputs.

**Usage**
```bash
python classify_urban_rural.py \
  --input /data/MoT3/processed/Taps_GPS_aligned_30sec.parquet \
  --ghsl-shp /path/to/GHSL_DUC.shp \
  --out /data/MoT3/output/aligned_with_urban.parquet
```

### Reproducibility Notes

- All scripts are modular and can be executed independently.  
- Graph tiles from OpenStreetMap are cached locally under `/cache/osm_tiles/`.  
- Derived datasets are anonymized and non-sensitive.  
- Original tappigraphy and GPS data are protected under the GIVA Lab ethical protocol.

Researchers seeking access to the original data should contact:

> **GIVA Lab – University of Zurich**  
> [https://www.geo.uzh.ch/en/units/giva.html](https://www.geo.uzh.ch/en/units/giva.html)

---

##  Core Packages

- pandas  
- numpy  
- pyarrow  
- dask  
- geopandas  
- shapely  
- osmnx  
- networkx  

---

##  Recommended Environment

- Python ≥ 3.9  
- ≥ 8 GB RAM  
