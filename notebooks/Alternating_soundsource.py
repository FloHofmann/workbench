# %%
import itertools
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scikit_posthocs as sp
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import (
    friedmanchisquare,
    gaussian_kde,
    rankdata,
)

from workbench.data.preprocess import combTableCreate, expand_dict_columns

# %matplotlib QtAgg
# plt.ion()
individuals_flag = False

# --------------------------------------------------------------------
# HELPER: Tukey-Kramer on Friedman Ranks (Matches MATLAB multcompare)
# --------------------------------------------------------------------
def tukey_kramer_friedman(data, labels):
    """
    Ranks the data within each block (row) and applies Tukey's HSD, 
    replicating MATLAB's multcompare(..., 'CType', 'tukey-kramer') 
    after a Friedman test.
    """
    # Rank data across columns (axis=1) for each row
    ranked_data = rankdata(data, axis=1)
    df = pd.DataFrame(ranked_data, columns=labels)
    melted = df.melt(var_name='group', value_name='rank')
    
    # Apply Tukey HSD on the ranked data
    res = sp.posthoc_tukey(melted, val_col='rank', group_col='group')
    return res

# %%
conn = sqlite3.connect(
    r"\\172.25.250.112\burgalossi\lab share\Data\Florian\Recordings_FH.db"
)
sql = """
SELECT * FROM Recordings WHERE (Animal_Id, Cell_Id) IN (
SELECT Animal_Id, Cell_Id FROM Recordings WHERE Condition IN ("Baseline", "soso") GROUP BY Animal_Id, Cell_Id
HAVING COUNT(DISTINCT Condition) >= 2)
AND Condition IN ("Baseline","soso") AND use = 1 AND Folders_generated = 1
"""

datatable = pd.read_sql_query(sql, conn)
conn.close()

# %%
try:
    comb_table = pl.read_parquet(
        Path(
            r"\\172.25.250.112\burgalossi\lab share\Data\Florian\comb_tables\soso_comb.parquet"
        )
    )
except:
    print("comb_table doesnt exist, start creation")
    comb_table = combTableCreate(datatable)
    comb_df = pd.DataFrame(comb_table)
    dict_list = ["pupil_psth", "whisk_psth", "eye_psth"]
    comb_df = expand_dict_columns(comb_df, dict_columns=dict_list, flatten_2d=False)
    # clean and reshape the dataframe
    comb_table = pl.from_dataframe(comb_df)
    conditions = comb_table["Condition"].unique().to_list()
    # --------------------------------------------------------------------
    comb_a = comb_table.filter(pl.col("Condition") == conditions[0]).drop("Condition")
    comb_b = comb_table.filter(pl.col("Condition") == conditions[1]).drop("Condition")
    # --------------------------------------------------------------------
    comb_joined = comb_a.join(comb_b, on=["Animal_Id", "Cell_Id"], how="inner")
    # --------------------------------------------------------------------
    comb_joined = comb_joined[
        [s.name for s in comb_joined if not (s.null_count() == comb_joined.height)]
    ]
    comb_table = comb_joined.rename(lambda c: c[:-6] if c.endswith("_right") else c)
    comb_table.write_parquet(
        r"\\172.25.250.112\burgalossi\lab share\Data\Florian\comb_tables\soso_comb.parquet",
        use_pyarrow=True,
    )
    print("table created and saved")

# filter comb_table for low total number of stimulations (at least 10 stims per speaker)
comb_table = comb_table.filter(pl.col("n_stims") > 40)

# Filter for max raster rows > 10 for the first speaker ('a') to match MATLAB logic
valid_rows = []
for i in range(len(comb_table)):
    r_rows = comb_table["RasterRows"][i]
    a_rows = r_rows.get('a', [])
    if len(a_rows) > 0 and np.max(a_rows) > 10:
        valid_rows.append(True)
    else:
        valid_rows.append(False)
comb_table = comb_table.filter(valid_rows)
print(f"{len(comb_table)} cells after n_stims > 40 and raster_rows > 10 filter.")

# %%
# analysis
speaker_position = {"a": 93, "w": 178, "e": 272, "r": 356}
rads_x = np.cos(np.deg2rad(list(speaker_position.values())))
rads_y = np.sin(np.deg2rad(list(speaker_position.values())))

# %%
# cell by cell sound source representation

half_window = 1500  # in ms
time_bin = 0.009
nbins = int(np.round(half_window / (time_bin * 1000)))
raster_edges = np.linspace(-half_window, half_window, nbins * 2 + 1)

# mimic the medfilt1
bins_plot = np.concatenate(
    [
        [raster_edges[0]],
        np.median(
            np.vstack(
                [raster_edges[:-1], raster_edges[1:]]
            ),
            axis=0,
        ),
    ]
)
bins_plot = bins_plot[1::]
resp_window = np.flatnonzero(
    (bins_plot > 1) & (bins_plot < 300)
)

speakers = ["a", "w", "e", "r"]
raster_times = comb_table["RasterTimes"]
raster_rows = comb_table["RasterRows"]
raster_rate = comb_table["RasterRate"]
HDRateSmooth = comb_table["hdRateSmooth"]
DIRECTIONS = np.linspace(0, 360, 37)
stim_keys = list(raster_times[0].keys())
stim_key_to_idx = {k: i for i, k in enumerate(stim_keys)}
speaker_idx = [stim_key_to_idx[k] for k in speakers]
trigger_time = np.array(comb_table["trigger_time"][0].to_list())
time_pre = (trigger_time < 0) & (trigger_time >= -0.5)

whisk_psth = {
    "a": comb_table["whisk_psth_a"],
    "w": comb_table["whisk_psth_w"],
    "e": comb_table["whisk_psth_e"],
    "r": comb_table["whisk_psth_r"],
}
new_rate = np.zeros((len(raster_times), len(raster_edges) - 1, len(stim_keys)))

whisk_avg_orig = np.array(comb_table["whisk_avg"].to_list())
whisk_avg = np.stack(
    [np.stack([cell[k] for k in stim_keys], axis=1) for cell in whisk_avg_orig], axis=0
)

baseline_subtract_whisk = np.empty_like(whisk_avg)

# %%

for i in range(len(raster_times)):
    for idx, j in enumerate(stim_keys):
        if not len(raster_times[i][j]) == 0:
            counts, _ = np.histogram(raster_times[i][j], bins=raster_edges)
            new_rate[i, :, idx] = np.divide(
                counts, (np.max(raster_rows[i][j]) * time_bin)
            )

for idx, stim in enumerate(stim_keys):
    baseline_median = np.nanmedian(
        whisk_avg[:, trigger_time <= 0, idx],
        axis=1,
        keepdims=True,
    )
    baseline_subtract_whisk[:, :, idx] = whisk_avg[:, :, idx] - baseline_median

new_rate = new_rate[:, :, speaker_idx]
whisk_avg = whisk_avg[:, :, speaker_idx]
baseline_subtract_whisk = baseline_subtract_whisk[:, :, speaker_idx]

# %%
# peak fr for all soundsources
preferred = np.array(comb_table["HDAngle"])
speaker_angles = np.array(list(speaker_position.values()))

diff = preferred[:, None] - speaker_angles[None, :]
wrapped = (diff + 180) % 360 - 180
dist_deg = np.abs(wrapped)

idx_sorted = np.argsort(dist_deg, axis=1)
dist_sorted = np.take_along_axis(dist_deg, idx_sorted, axis=1)
speakers_sorted = np.take_along_axis(speaker_angles[None, :], idx_sorted, axis=1)

resp_peak_fr = np.array(
    [np.max(new_rate[:, resp_window, i], axis=1) for i in range(0, 4)]
).T
resp_peak_sorted = np.take_along_axis(resp_peak_fr, idx_sorted, axis=1)

fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)

xlab = ["Closest", "Second closest", "Second Farthest", "Farthest"]
for i, key in enumerate(speaker_position):
    axes[i].scatter(resp_peak_sorted[:, i], dist_sorted[:, i], s=10, alpha=0.7)
    axes[i].set_xlabel(xlab[i])
    if i == 0:
        axes[i].set_ylabel("Circular distance to speaker (deg)")

plt.tight_layout()
plt.show(block=False)

# %%
resp_peak_df = pd.DataFrame(resp_peak_fr, columns=list("awer"))
long_df = resp_peak_df.reset_index().melt(
    id_vars="index", var_name="speaker", value_name="max_psth"
)
plt.figure(figsize=(10, 5))
sns.boxplot(data=long_df, x="speaker", y="max_psth", fliersize=0)
sns.lineplot(
    data=long_df,
    x="speaker",
    y="max_psth",
    hue="index",
    estimator=None,
    linewidth=1,
    alpha=0.6,
    legend=False,
)

plt.gca().margins(y=0.3)
plt.xlabel("Speaker")
plt.ylabel("Max Psth")
plt.title("max psth")
plt.tight_layout()
plt.show(block=False)

stat, p = friedmanchisquare(
    resp_peak_df["a"], resp_peak_df["w"], resp_peak_df["e"], resp_peak_df["r"]
)
print("-" * 70)
print("Max FR not sorted")
print(f"Friedman Chi^2: {stat:.2f}; p: {p:.4f}")
print("-" * 70)

# %%
# max psth (firing rate) sorted by distance

fr_resp_sorted = np.take_along_axis(resp_peak_fr, idx_sorted, axis=1)
fr_labels = ["closest", "second_closest", "second_farthest", "farthest"]
fr_resp_df_sort = (
    pd.DataFrame(fr_resp_sorted, columns=fr_labels)
    .reset_index()
    .melt(id_vars="index", var_name="speaker", value_name="fr_psth")
)

plt.figure(figsize=(12, 5))
plt.title("Max FR sorted by speaker distance")
ax = sns.boxplot(data=fr_resp_df_sort, x="speaker", y="fr_psth")
ax.set_xticks([0, 1, 2, 3])
ax.set_ylabel("Fr [Hz]")
ax.set_xlabel("Speaker position relative to PD")
ax.set_xticklabels(fr_labels)
ax.margins(y=0.3)
sns.lineplot(
    data=fr_resp_df_sort,
    x="speaker",
    y="fr_psth",
    hue="index",
    linewidth=1,
    legend=False,
)
plt.show(block=False)

fr_stat, fr_p = friedmanchisquare(
    fr_resp_sorted[:, 0],
    fr_resp_sorted[:, 1],
    fr_resp_sorted[:, 2],
    fr_resp_sorted[:, 3],
)
print("-" * 70)
print("Max FR sorted")
print(f"Friedman Chi square: {fr_stat:.2f}, p = {fr_p:.4f}")
print("Pairwise (Tukey-Kramer on ranks):")

tukey_fr = tukey_kramer_friedman(fr_resp_sorted, fr_labels)
for c1, c2 in itertools.combinations(range(4), 2):
    print(f"{fr_labels[c1]} vs {fr_labels[c2]}: p = {tukey_fr.iloc[c1, c2]:.3f}")

print("-" * 70)

# %%
# calculate the integral over the bins
auc_resp = (new_rate[:, resp_window, :] * time_bin).sum(axis=1)
auc_resp_df = (
    pd.DataFrame(auc_resp, columns=["a", "w", "e", "r"])
    .reset_index()
    .melt(id_vars="index", var_name="speaker", value_name="auc_psth")
)
auc_stat, auc_p = friedmanchisquare(
    auc_resp[:, 0], auc_resp[:, 1], auc_resp[:, 2], auc_resp[:, 3]
)
plt.figure(figsize=(10, 5))
plt.title("auc unsorted")
ax = sns.boxplot(data=auc_resp_df, x="speaker", y="auc_psth")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(speakers)
ax.margins(y=0.3)
sns.lineplot(
    data=auc_resp_df, x="speaker", y="auc_psth", hue="index", linewidth=1, legend=False
)
plt.show(block=False)
print("-" * 70)
print("AUC unsorted")
print(f"Friedman p = {auc_p:.4f}")
print("Pairwise (Tukey-Kramer on ranks):")

tukey_auc = tukey_kramer_friedman(auc_resp, speakers)
for c1, c2 in itertools.combinations(range(4), 2):
    print(f"{speakers[c1]} vs {speakers[c2]}: p = {tukey_auc.iloc[c1, c2]:.3f}")

print("-" * 70)


# %%
auc_resp_sorted = np.take_along_axis(auc_resp, idx_sorted, axis=1)
auc_sort_labels = ["closest", "second_closest", "second_farthest", "farthest"]
auc_resp_df_sort = (
    pd.DataFrame(auc_resp_sorted, columns=auc_sort_labels)
    .reset_index()
    .melt(id_vars="index", var_name="speaker", value_name="auc_psth")
)
plt.figure(figsize=(12, 5))
plt.title("auc sorted")
ax = sns.boxplot(data=auc_resp_df_sort, x="speaker", y="auc_psth")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(auc_sort_labels)
ax.margins(y=0.3)
sns.lineplot(
    data=auc_resp_df_sort,
    x="speaker",
    y="auc_psth",
    hue="index",
    linewidth=1,
    legend=False,
)
plt.show(block=False)

auc_stat, auc_p = friedmanchisquare(
    auc_resp_sorted[:, 0],
    auc_resp_sorted[:, 1],
    auc_resp_sorted[:, 2],
    auc_resp_sorted[:, 3],
)
print("-" * 70)
print("AUC Sorted")
print(f"Friedman chi-square: {auc_stat:.2f}; p: {auc_p:.4f}")
print("Pairwise (Tukey-Kramer on ranks):")

tukey_auc_sort = tukey_kramer_friedman(auc_resp_sorted, auc_sort_labels)
for c1, c2 in itertools.combinations(range(4), 2):
    print(f"{auc_sort_labels[c1]} vs {auc_sort_labels[c2]}: p = {tukey_auc_sort.iloc[c1, c2]:.3f}")

print("-" * 70)

# %% [markdown]
# Computation of correlations for each cell

new_rate_sort = np.take_along_axis(new_rate, idx_sorted[:, None, :], axis=2)
avg_whisker_sort = np.take_along_axis(whisk_avg, idx_sorted[:, None, :], axis=2)

baseline_time_idx = (bins_plot > -1000) & (bins_plot < 0)
baseline_mean = new_rate_sort[:, baseline_time_idx, :].mean(axis=1, keepdims=True)
new_rate_bs = new_rate_sort - baseline_mean

baseline_whisk_time_idx = (trigger_time > -1000) & (trigger_time < 0)
baseline_mean_whisk = avg_whisker_sort[:, baseline_whisk_time_idx, :].mean(
    axis=1, keepdims=True
)
avg_whisker_bs = avg_whisker_sort - baseline_mean_whisk

new_rate_bs_smooth = new_rate_bs 
response_data = new_rate_bs_smooth[:, resp_window, :]

# %%
n_cells = response_data.shape[0]
n_speakers = response_data.shape[2]

corr_all = np.zeros((n_cells, n_speakers, n_speakers))

for cell in range(n_cells):
    corr_all[cell] = np.corrcoef(response_data[cell].T)

mean_corr_matrix = corr_all.mean(axis=0)

# %%
triu_idx = np.triu_indices(n_speakers, k=1)
corr_pairs = corr_all[:, triu_idx[0], triu_idx[1]]
corr_values = corr_pairs.flatten()

# %%
mean_corr_per_cell = corr_pairs.mean(axis=1)

# %% average response, sorted by speaker distance
mean_psth = new_rate_bs_smooth.mean(axis=0)
sd_psth = new_rate_bs_smooth.std(axis=0)
plot_resp_show = np.arange(resp_window[0] - 50, resp_window[-1] + 100, step=1)

mean_whisk = avg_whisker_bs.mean(axis=0)
sd_psth_whisk = avg_whisker_bs.std(axis=0)
plot_resp_show_whisk = (trigger_time <= 0.5) & (trigger_time >= -0.1)

# %%
n_cells, n_spk, _ = corr_all.shape
i_idx, j_idx = np.triu_indices(n_spk, k=1)

pair_labels = [
    f"{distance_label[i]} – {distance_label[j]}" for i, j in zip(i_idx, j_idx)
]

pair_rank_dist = j_idx - i_idx
sort_idx = np.argsort(pair_rank_dist, kind="stable")

corr_pairs = corr_pairs[:, sort_idx]
pair_labels = [pair_labels[i] for i in sort_idx]

np.random.seed(42)
fig, ax = plt.subplots(figsize=(8, 5))
n_pairs = corr_pairs.shape[1]
x_positions = np.arange(n_pairs)

ax.boxplot(
    [corr_pairs[:, i] for i in range(n_pairs)],
    positions=x_positions,
    widths=0.5,
    showfliers=False,
)

colors = plt.cm.tab20(np.linspace(0, 1, n_cells))
for i in range(n_cells):
    jitter = np.random.uniform(-0.15, 0.15, size=n_pairs)
    ax.scatter(
        x_positions + jitter,
        corr_pairs[i, :],
        color=colors[i],
        edgecolors="black",
        s=50,
        linewidth=0.8,
        alpha=0.9,
    )

ax.set_xticks(x_positions)
ax.set_xticklabels(pair_labels, rotation=45, ha="right")
plt.xticks(rotation=65)

ax.set_ylabel("PSTH correlation (r)")
ax.set_xlabel("Speaker pair")
ax.margins(y=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show(block=False)

stat, p = friedmanchisquare(*[corr_pairs[:, i] for i in range(n_pairs)])
print(f"Friedman chi² = {stat:.3f}, p = {p:.3f}")
print("\nPairwise comparisons (Tukey-Kramer on ranks):")

tukey_corr = tukey_kramer_friedman(corr_pairs, pair_labels)
for i, j in itertools.combinations(range(n_pairs), 2):
    print(f"({pair_labels[i]}) vs ({pair_labels[j]}): p = {tukey_corr.iloc[i, j]:.3f}")

print("-" * 70)