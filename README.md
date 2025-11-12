# How Sense of Direction, Spatial Anxiety, and Urban Form Shape Interaction with GPS-Embedded Mobile Map Applications  
### Repository for Zingaro et al., ACM TSAS (2025, in review)

[![Build Status](https://github.com/DonatellaZingaro/TSAS_Special_Issue_Urban_Mobility/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/DonatellaZingaro/TSAS_Special_Issue_Urban_Mobility/actions/workflows/python-package-conda.yml)


This repository contains the full analysis pipeline supporting the manuscript:

> **How Sense of Direction, Spatial Anxiety, and Urban Form Shape Interaction with GPS-Embedded Mobile Map Applications**  
> **Donatella Zingaro** [![ORCID](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0002-3658-6045),  
> **Tumasch Reichenbacher** [![ORCID](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0002-7833-185X),  
> **Sara I. Fabrikant** [![ORCID](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0003-1263-8792)  
> *ACM Transactions on Spatial Algorithms and Systems* (in review)

---

## Abstract

How do the built environment and individual differences jointly shape real-world navigation behavior?  
To investigate this question, we analyzed **everyday interaction with GPS-embedded mobile map applications**, combining:

- touchscreen dynamics (tappigraphy)
- graph-based street network metrics
- self-reported spatial anxiety (SA), sense of direction (SBSOD), and gender

We quantified **engagement frequency** (*tap rate*) and **engagement rhythm** (*inter-tap variability*, CV–ITI) from smartphone map use during navigation.

Key findings reveal that:

- **Spatial anxiety** consistently predicts reduced and more fragmented interaction
- **Urban form matters**:
  - Higher **orientation entropy** and **degree centrality** → fewer touches
  - Higher **closeness centrality** → more regular contact patterns  
- These environmental effects **depend on the navigator**:
  - High-anxiety users show stronger disruption in complex, winding layouts
  - Individuals with strong orientation skills interact more **selectively and strategically**

Taken together, these results highlight the need for **context-aware and cognitively informed mobile map design**. By introducing a privacy-preserving behavioral dataset from real-world navigation, we aim to support **adaptive navigation interfaces** that encourage more spatially engaged and sustainable wayfinding.

---

### Keywords

Urban Mobility · Spatial Cognition · Urban Morphology · Street Network Analysis · Mobile Map Interaction · Tappigraphy

---

## Repository Contents

| Directory / File | Description |
|-----------------|-------------|
| `notebooks/` | sensitivity analysis, robust regressions, moderation analyses, Bayesian multivariate model |
| `src/` | Modular Python scripts for preprocessing, urban-form computation, and diagnostics |
| `data/` | data + links to anonymized OSF datasets |
| `requirements.txt` | Python dependencies and pinned versions for reproducibility |
| `CITATION.cff` | Machine-readable citation metadata |

> Python requirement: **Python 3.10+** is needed for the analysis code in `src/`

---

## Data Availability & Privacy

Raw GPS trajectories and tappigraphy logs are **not** publicly released to protect participant privacy.

Instead, **fully anonymized and pre-processed** feature tables, supplementary documentation, and figures are openly available via OSF:

OSF Repository → https://osf.io/nz3s8/

A detailed data dictionary is included in `data/README_DATA.md`.

---

## Reproducing the Analysis

To reproduce the full analysis pipeline locally:

```bash
# Clone this repository
git clone https://github.com/donatellazingaro/TSAS_Special_Issue_Urban_Mobility.git
cd TSAS_Special_Issue_Urban_Mobility

#  Install Python dependencies
pip install -r requirements.txt

# Launch the analysis environment
jupyter lab
