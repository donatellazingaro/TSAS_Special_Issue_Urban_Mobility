# Data Dictionary and Access Information

This directory contains **anonymized and pre-processed session-level features** derived from tappigraphy logs, GPS traces, and questionnaire data collected through the MapOnTap platform. For privacy protection, no raw trajectories or identifying metadata are included.

---

## File Contents

| Filename | Dimensions | Unit of Analysis |
|----------|------------|-----------------|
| `pre-processed_anonimized_final_clean.csv` | 2,442 rows × 23 columns | GPS-embedded mobile map sessions |

Each row represents a single mobile map app session during real-world navigation.  
All columns have been inspected for direct identifiers and standardized where necessary.

---

## Column Definitions

| Column | Description | Type |
|--------|-------------|------|
| **partId** | Anonymous participant identifier | Categorical |
| **sessionId** | Unique session identifier (anonymous) | Categorical |
| **tap_rate_z** | Standardized tap frequency (taps/s) | Continuous |
| **cv_iti_z** | Standardized temporal variability in inter-touch intervals (CV–ITI) | Continuous |
| **SA_z** | Standardized spatial anxiety (higher = more anxious) | Continuous |
| **SBSOD_z** | Standardized self-reported sense of direction (higher = better) | Continuous |
| **Gender_M** | Dummy variable: 1 = male, 0 = female | Binary |
| **orientation_entropy_z** | Standardized orientation entropy (500 m buffer) | Continuous |
| **betweenness_log_z** | Standardized log-transformed street network betweenness | Continuous |
| **closeness_log_z** | Standardized log-transformed street network closeness | Continuous |
| **degree_mean_z** | Standardized mean node degree | Continuous |
| **circuity_log_z** | Standardized log-transformed circuity | Continuous |
| **lat_c** | Session centroid latitude (decimal degrees, offset for privacy) | Continuous |
| **lon_c** | Session centroid longitude (decimal degrees, offset for privacy) | Continuous |
| **session_duration_sec** | Duration of the session in seconds | Continuous |
| **num_taps** | Number of touchscreen interactions in the session | Continuous |
| **timestamp_start_utc** | Session start time (with uniform anonymizing temporal shift) | Timestamp |

---

## Privacy and Access

Raw GPS trajectories and tappigraphy logs are **not shared** to ensure participant anonymity.  
A fully anonymized version of the feature table above is also openly available through OSF:
 OSF Repository → https://osf.io/nz3s8/

---

##  Reproducibility Notes

- Standardization was performed **within-sample** prior to modeling.
- All urban form metrics were extracted within **500 m buffers** from session centroids using OSMnx street graphs.
- Temporal shifts preserve order and duration while preventing deanonymization.

---

If further derived datasets or updated versions are released, they will be added to this directory with version tags and described here.
