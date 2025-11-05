#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Supplementary Table S1:
Spearman correlations and mean differences across
urban-form buffer sizes (200m, 500m, 1000m).

This script loads the processed sensitivity table and
exports a publication-ready PDF. Designed to be run locally
or on any machine with the project structure below:

TSAS_Special_Issue_Urban_Mobility/
│
├── data/
│   └── Table_S1_Sensitivity_Results.csv
└── outputs/
    └── sensitivity/
        └── Table_S1_Sensitivity_Results.pdf

Requires:
- Python 3.10+
- pandas, matplotlib
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("../data/Table_S1_Sensitivity_Results.csv")
OUT_DIR = Path("../outputs/sensitivity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = OUT_DIR / "Table_S1_Sensitivity_Results.pdf"

CAPTION = (
    "Table S1. Spearman rank correlations (ρ) and percentage mean "
    "differences (ΔMean%) for urban-form metrics computed using 200 m, "
    "500 m, and 1000 m network buffers (pairwise comparisons). These "
    "checks confirm that 500 m provides stable and interpretable "
    "estimates for the primary analyses."
)

RENAME_METRICS = {
    "closeness": "Closeness Centrality",
    "betweenness": "Betweenness Centrality",
    "degree": "Degree Centrality",
    "orientation_entropy": "Orientation Entropy",
    "circuity": "Circuity"
}

def main():
    print("\n=== Generating Supplementary Table S1 ===")


    if not DATA_PATH.exists():
        raise FileNotFoundError(f"❌ Data not found: {DATA_PATH.resolve()}")


    df = pd.read_csv(DATA_PATH, comment="#")


    df["Metric"] = df["Metric"].map(RENAME_METRICS)
    df["From_Buffer"] = df["From_Buffer"].astype(str) + " m"
    df["To_Buffer"] = df["To_Buffer"].astype(str) + " m"
    df["Spearman_rho"] = df["Spearman_rho"].round(3)
    df["Delta_Mean_pct"] = df["Delta_Mean_pct"].round(1)


    df = df.rename(columns={
        "Metric": "Urban Form Metric",
        "From_Buffer": "From",
        "To_Buffer": "To",
        "Spearman_rho": "ρ (Spearman)",
        "Delta_Mean_pct": "ΔMean (%)"
    })


    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")


    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )


    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_linewidth(1.0)


    plt.figtext(
        0.5, -0.05, CAPTION,
        wrap=True, ha="center", fontsize=9
    )


    plt.tight_layout()
    plt.savefig(PDF_PATH, bbox_inches="tight", dpi=300)
    plt.close()


    print(f"✅ PDF exported successfully → {PDF_PATH.resolve()}\n")


if __name__ == "__main__":
    main()
