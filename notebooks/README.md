# Analysis Notebooks

The notebooks are organized as follows:

---

### `robust_regression_models.ipynb`

This notebook investigates how **urban morphology** and **individual spatial traits** contribute to two key indicators of mobile map engagement:

- **Tap rate** (frequency of interaction)
- **CV-ITI** (rhythmic variability of touch events)

Using **Robust Linear Models (RLM)** with Huber weighting, it:

- Implements a cross-validated evaluation workflow
- Computes performance metrics (MAD, pseudo-R², CV-MAD)
- Generates diagnostic plots of residuals
- Saves detailed model summaries to `../outputs/regression/summaries/`

This analysis supports the central claim that **street network form predicts engagement**, especially for users experiencing heightened spatial anxiety.

---

### `moderation_analysis.ipynb`

This notebook explores **interaction effects**—how the impact of the environment on mobile map use depends on who the navigator is.

It evaluates moderation by:

- Spatial Anxiety
- Sense of Direction (SBSOD)
- Gender

using interaction terms within a robust regression framework.

High-quality visualizations illustrate:

- Closeness × SA (CV-ITI)
- Entropy × SA (Tap rate)
- Closeness × SBSOD (CV-ITI)
- Degree × SBSOD (Tap rate)
- SA × SBSOD (Tap rate)
- Gender × Closeness / Entropy

Figures are saved to `../outputs/moderation/figs/`.

These results strengthen the narrative that **cognitive and emotional traits modulate interaction behavior in real-world environments**.

---

### `bayesian_multivariate_models.ipynb`

This notebook estimates a **Bayesian multivariate regression model** in which:

- Tap rate and CV-ITI are modeled jointly
- Shared residual structure captures the co-dependence of engagement frequency and rhythm

Outputs include:

- Posterior parameter summaries
- Residual correlations between outcomes
- Trace and convergence diagnostics

Figures and summaries are stored under:  
`../outputs/bayesmv/figs/` and `../outputs/bayesmv/summaries/`.

This analysis provides a more comprehensive account of **how environment and traits shape interaction dynamics as a coupled system**.

---

## Data Dependencies

All notebooks rely on the anonymized session-level dataset:

