from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import (
    fetch_atlas_schaefer_2018,
    fetch_atlas_yeo_2011,
    fetch_development_fmri,
)
from nilearn.image import iter_img, load_img, math_img, resample_to_img
from nilearn.interfaces.fmriprep import load_confounds, load_confounds_strategy
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from scipy import stats

dataset = fetch_development_fmri(reduce_confounds=False)
yeo_networks = fetch_atlas_yeo_2011(n_networks=7, thickness="thick")
schaefer400 = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
# mist444 = fetch_atlas_basc_multiscale_2015(resolution=400, version='asym')

# create a dictionary to summarise yeo network seed coverage
yeo7_nii = load_img(yeo_networks.maps)
yeo7_nii = list(iter_img(yeo7_nii))[0]  # for some reason there's a fourth dimension
network_labels = np.unique(yeo7_nii.dataobj)[1:]
atlas_nii = load_img(schaefer400.maps)
region_labels = np.unique(atlas_nii.dataobj)[1:]

yeo7_nii = resample_to_img(yeo7_nii, schaefer400.maps, interpolation="nearest")
region_membership = pd.DataFrame(0, index=region_labels, columns=[f"yeo7-{int(n)}" for n in network_labels])
network_masker = NiftiLabelsMasker(yeo7_nii).fit()

for n in network_labels:
    cur_region = math_img(f"img=={n}", img=yeo7_nii)
    masker = NiftiMasker(cur_region)
    atlas_parcel_in_network = masker.fit_transform(schaefer400.maps)
    atlas_parcel_in_network = np.unique(atlas_parcel_in_network)[1:]
    region_membership.loc[atlas_parcel_in_network, f"yeo7-{int(n)}"] = 1

# look for regions that's only in dmn
roi_index_single_network_membership = region_membership.index[region_membership.T.sum() == 1]
roi_index_single_and_dmn = roi_index_single_network_membership[
    region_membership.loc[roi_index_single_network_membership, "yeo7-7"] == 1
]
# add og label as a reference
region_membership["original_labels"] = schaefer400.labels[1:]
region_membership.to_csv("sandbox/reports/region_membership.tsv", sep="\t")

# initialise objects
atlas_masker = NiftiLabelsMasker(schaefer400.maps, labels=schaefer400.labels).fit()
connectome = ConnectivityMeasure(kind="correlation")

denoising = {
    "simple": {"denoise_strategy": "simple", "motion": "basic"},
    "simple+grs": {
        "denoise_strategy": "simple",
        "motion": "basic",
        "global_signal": "basic",
    },
    "scrubbing": {
        "denoise_strategy": "scrubbing",
        "fd_threshold": 0.5,
        "motion": "basic",
        "wm_csf": "basic",
    },
    "scrubbing+grs": {
        "denoise_strategy": "scrubbing",
        "fd_threshold": 0.5,
        "motion": "basic",
        "wm_csf": "basic",
        "global_signal": "basic",
    },
    "compcor": None,
    "baseline": {"strategy": ["high_pass"]},
}

for dk in denoising:
    # extract schaefer 400 and denoise
    if dk == "baseline":
        confounds, sample_masks = load_confounds(dataset.func, **denoising[dk])
        connectomes_prepro = []
        for img, c, s in zip(dataset.func, confounds, sample_masks, strict=False):
            ts = atlas_masker.transform(img, confounds=c, sample_mask=s)
            conn = connectome.fit_transform(ts)
            del ts
            connectomes_prepro.append(conn)

    elif dk == "compcor":
        dataset = fetch_development_fmri(reduce_confounds=True)
        connectomes_prepro = []
        for img, pc in zip(dataset.func, dataset.confounds, strict=False):
            c = pd.read_csv(pc, sep="\t")
            ts = atlas_masker.transform(img, confounds=c)
            conn = connectome.fit_transform(ts)
            del ts
            connectomes_prepro.append(conn)
    else:
        confounds, sample_masks = load_confounds_strategy(dataset.func, **denoising[dk])
        connectomes_prepro = []
        for img, c, s in zip(dataset.func, confounds, sample_masks, strict=False):
            ts = atlas_masker.transform(img, confounds=c, sample_mask=s)
            conn = connectome.fit_transform(ts)
            del ts
            connectomes_prepro.append(conn)

    # calculate similarity of the thresholded individual level seed based connectivity with the binary mask
    average_connectivity_within_network = []
    std_connectivity_within_network = []
    corr_wtih_dmn = []
    for _i, conn in enumerate(connectomes_prepro):
        conn = conn.squeeze()
        subj_average_connectivity_within_network = []
        subj_variance_connectivity_within_network = []
        subj_corr_wtih_dmn = []
        for idx_roi in roi_index_single_and_dmn - 1:
            seed_based_map = conn[int(idx_roi), :]

            networks = [region_membership[f"yeo7-{int(n)}"].values * seed_based_map for n in network_labels]
            networks = np.array(networks)
            networks[networks == 0] = np.nan

            subj_average_connectivity_within_network.append(np.nanmean(networks, axis=1))
            subj_variance_connectivity_within_network.append(np.nanstd(networks, axis=1))
            subj_corr_wtih_dmn.append(np.corrcoef(seed_based_map, region_membership["yeo7-7"].values)[1, 0])

        subj_average_connectivity_within_network = pd.DataFrame(subj_average_connectivity_within_network).mean()
        subj_variance_connectivity_within_network = pd.DataFrame(subj_variance_connectivity_within_network).mean()
        subj_corr_wtih_dmn = pd.DataFrame(subj_corr_wtih_dmn).mean()

        average_connectivity_within_network.append(subj_average_connectivity_within_network.values)
        std_connectivity_within_network.append(subj_variance_connectivity_within_network.values)
        corr_wtih_dmn.append(subj_corr_wtih_dmn.values)

    average_connectivity_within_network = pd.DataFrame(
        average_connectivity_within_network, columns=[f"mean_{r}" for r in region_membership.columns[:-1]]
    )
    std_connectivity_within_network = pd.DataFrame(
        std_connectivity_within_network, columns=[f"sd_{r}" for r in region_membership.columns[:-1]]
    )
    corr_wtih_dmn = pd.DataFrame(corr_wtih_dmn, columns=["corr_with_dmn"])

    summary = pd.concat([average_connectivity_within_network, std_connectivity_within_network, corr_wtih_dmn], axis=1)
    if dk != "compcor":
        # add motion metrics
        meanfd = []
        for conf in dataset.confounds:
            fd = pd.read_csv(conf, sep="\t")["framewise_displacement"].mean()
            meanfd.append(fd)
        summary["mean_framewise_displacement"] = meanfd
    summary["participant_id"] = dataset.phenotypic["participant_id"].values

    # calculate distance
    summary["mean-diff_dmn_visual"] = summary["mean_yeo7-7"] - summary["mean_yeo7-1"]
    summary["mean-diff_dmn_motor"] = summary["mean_yeo7-7"] - summary["mean_yeo7-2"]
    summary["mean-diff_dmn_fpn"] = summary["mean_yeo7-7"] - summary["mean_yeo7-6"]
    weird_subjects = summary.index[summary["mean-diff_dmn_visual"] < summary["mean-diff_dmn_fpn"]]
    summary = summary.set_index("participant_id")
    summary.to_csv(f"sandbox/reports/{dk}_average-connectivity-within-network.tsv", sep="\t")

    # plot the distribution
    plt.figure()
    sns.histplot(summary[["mean-diff_dmn_visual", "mean-diff_dmn_fpn"]])
    plt.savefig(f"sandbox/reports/{dk}_dmn-visual-fpn_distance.png")
    plt.figure()
    sns.histplot(summary[["mean_yeo7-7"]])
    plt.savefig(f"sandbox/reports/{dk}_yeo7-7.png")
    plt.figure()
    sns.histplot(summary.corr_with_dmn)
    plt.savefig(f"sandbox/reports/{dk}_corr-dmn.png")

    # # plot all subject with pcc seed connectivity
    # cur_region = math_img('img==7', img=yeo7_nii)
    # cur_region = resample_to_img(cur_region, atlas_nii, interpolation='nearest')
    # for i, conn in enumerate(connectomes_prepro):
    #     if i in weird_subjects:
    #         fp = Path(f"sandbox/subject_fc_maps/{dk}_{dataset.phenotypic.iloc[i]['participant_id']}.png")
    #         if not fp.exists():
    #             conn = conn.squeeze()
    #             seed_based_map = conn[int(roi_index_single_and_dmn[-1] - 1), :]
    #             nii = atlas_masker.inverse_transform(seed_based_map)
    #             display = plot_stat_map(
    #               nii, cmap='cold_hot', vmin=-1, vmax=1,
    #               title=dataset.phenotypic.iloc[i]['participant_id']
    #             )
    #             display.add_contours(cur_region, color='k')
    #             plt.savefig(fp)

# Group stats time!

stats_path = list(Path("sandbox/reports/").glob("*_average-connectivity-within-network.tsv"))
denoising = {
    "baseline": {},
    "simple": {},
    "simple+grs": {},
    "scrubbing": {},
    "scrubbing+grs": {},
    "compcor": {},
}

for p in stats_path:
    dstrategy_name = p.name.split("_")[0]
    df = pd.read_csv(p, sep="\t")
    mean_corr_with_dmn = df["corr_with_dmn"].mean()
    t, _ = stats.ttest_rel(df["mean-diff_dmn_visual"], df["mean-diff_dmn_fpn"])
    denoising[dstrategy_name] = {
        "Similarity between DMN and the seed-based connectivity map\n(Pearson's correlations)": mean_corr_with_dmn,
        "Difference of Distance between DMN vs Visual network and DMN vs FPN\n(t-score)": t,
    }

summary = pd.DataFrame(denoising)

for i in range(summary.shape[0]):
    plt.figure()
    plt.title(summary.index[i].split("\n")[0])
    sns.barplot(summary.iloc[i, :])
    plt.ylabel(summary.index[i].split("\n")[1])
    plt.savefig(f"sandbox/reports/groupstat{i + 1}.png")
