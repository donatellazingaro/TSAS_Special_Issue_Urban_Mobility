FILE CONTENTS
-------------

Filename:
pre-processed_anonimized_final_clean.csv

Dimensions:
2,442 rows × 23 columns

Unit of Analysis:
Individual GPS-embedded mobile map sessions


COLUMN DEFINITIONS
------------------

Column                  | Description                                                                  | Type
-------------------------------------------------------------------------------------------------------------
partId                  | Anonymous participant identifier                                             | Categorical
sessionId               | Unique session identifier (anonymous)                                        | Categorical
tap_rate_z              | Standardized tap frequency (taps/s), session-level                           | Continuous
cv_iti_z                | Standardized temporal variation in touch events (CV of inter-touch intervals)| Continuous
SA_z                    | Standardized spatial anxiety score                                           | Continuous
SBSOD_z                 | Standardized self-reported sense of direction                                | Continuous
Gender_M                | Dummy variable: 1 = male, 0 = female                                         | Binary
orientation_entropy_z   | Standardized street-orientation entropy (500 m buffer)                       | Continuous
betweenness_log_z       | Standardized log-transformed street network betweenness                      | Continuous
closeness_log_z         | Standardized log-transformed street network closeness                        | Continuous
degree_mean_z           | Standardized mean node degree                                                | Continuous
circuity_log_z          | Standardized log-transformed circuity                                        | Continuous
lat_c                   | Session centroid latitude (decimal degrees)                                  | Continuous
lon_c                   | Session centroid longitude (decimal degrees)                                 | Continuous
session_duration_sec    | Raw session duration in seconds                                              | Continuous
num_taps                | Raw number of taps during the session                                        | Continuous
timestamp_start_utc     | Session start time (with anonymized temporal shift applied)                  | Timestamp
