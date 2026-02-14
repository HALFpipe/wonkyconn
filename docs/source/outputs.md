# Outputs

Wonkyconn performs group-level analysis and writes the following files to the output directory.

## `metrics.tsv`

A tab-separated file containing group-level quality control and connectivity metrics, indexed by the grouping variable (e.g. atlas segmentation). Columns include:

- `median_absolute_qcfc` — Median absolute QC-FC correlation across edges.
- `percentage_significant_qcfc` — Percentage of edges with significant QC-FC correlation.
- `distance_dependence` — Correlation between QC-FC values and inter-node distance.
- `gcor` — Global correlation (mean and SEM across subjects).
- `dmn_vis_distance_vs_dmn_fpn` — Paired t-statistic comparing DMN–visual vs DMN–FPN mean connectivity distance.
- `dmn_similarity_mean` — Mean correlation of individual connectivity patterns with the DMN template.
- `dmn_similarity_std` — Standard deviation of the DMN similarity across subjects.
- `degrees_of_freedom_loss_mean` — Mean degrees of freedom lost to denoising.
- `degrees_of_freedom_loss_std` — Standard deviation of degrees of freedom loss.
- `sex_auc` — AUC for sex classification from connectivity (with CI bounds).
- `sex_accuracy` — Accuracy for sex classification.
- `age_mae` — Mean absolute error for age prediction from connectivity (with CI bounds).
- `age_r2` — R² for age prediction.
- `gradients_similarity` — Similarity of group-level connectivity gradients to a reference.

## `metrics.png`

A summary visualization of the group-level metrics.

## `dmn_similarity_*.tsv`

Per-subject DMN similarity statistics, including within-network mean connectivity, standard deviation, and correlation with the DMN template for each Yeo 7 network.
