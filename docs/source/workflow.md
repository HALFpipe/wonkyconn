# Workflow

Wonkyconn evaluates the quality of functional connectivity matrices derived from fMRIPrep outputs at the group level. Given a set of pre-computed connectivity matrices, a phenotype file (age, gender), and atlas information, it computes a battery of benchmarking metrics and produces a summary report.

The processing pipeline computes the following metrics for each group of connectivity matrices:

- **QC-FC correlation** — correlation between framewise displacement and each edge, with median absolute value and percentage of significant edges.
- **Distance dependence** — correlation between QC-FC values and Euclidean distance between atlas nodes.
- **Global correlation (GCOR)** — average correlation across all pairs of time series, summarised as mean and SEM.
- **DMN similarity** — within-network mean connectivity for each Yeo 7 network, and correlation of seed-based connectivity patterns with the default mode network template.
- **DMN distance t-statistic** — paired t-test comparing DMN–visual vs DMN–FPN mean connectivity distances.
- **Degrees of freedom loss** — mean and standard deviation of temporal degrees of freedom lost to confound regression.
- **Gradient similarity** — alignment of group-level connectivity gradients with a reference decomposition (optional, disabled in light mode).
- **Age and sex prediction** — cross-validated AUC for sex classification and MAE / R² for age prediction from connectivity features (optional, disabled in light mode).
