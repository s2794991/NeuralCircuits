from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


csv_file = Path("data") / "master_detailed_comment.csv"
output_folder = Path("q1_outputs")
output_folder.mkdir(exist_ok=True)

# Use axon length as proj measurement as closer to projection strength than count whether a region has any endpoint
measure = "length"

# More clusters more groups not good
n_clusters = 3

# 1% whether a region is a real target
# 5% small regions do not make the label too long
target_cutoff = 0.01
label_cutoff = 0.05

neurons = pd.read_csv(csv_file)

comments = neurons["comment"].fillna("").str.lower()
neurons = neurons.loc[~comments.str.contains("bad tracing")].copy()

comments = neurons["comment"].fillna("").str.lower()
neurons = neurons.loc[~comments.str.contains("outside of ecl5a")].copy()





projection_columns = []
regions = []
for column in neurons.columns:
    if column.endswith("_" + measure):
        projection_columns.append(column)
        regions.append(column.replace("_" + measure, ""))

projection_length = neurons[projection_columns].fillna(0)
projection_length.columns = regions

neurons["total_length"] = projection_length.sum(axis=1)
neurons = neurons.loc[neurons["total_length"] > 0].copy()
projection_length = projection_length.loc[neurons.index]


relative_projection = projection_length.copy()
for i in relative_projection.index:
    total = projection_length.loc[i].sum()
    for region in regions:
        relative_projection.loc[i, region] = projection_length.loc[i, region] / total

n_targets = []
for i in relative_projection.index:
    row = relative_projection.loc[i]
    n_targets.append((row >= target_cutoff).sum())

neurons["n_target_regions"] = n_targets

neurons["projection_breadth_type"] = "multi-region"
neurons.loc[neurons["n_target_regions"] == 1, "projection_breadth_type"] = "single-region"


# Check k(2 3 4 5) for the Kmeans result
# k = 3 is good?
cluster_number_rows = []
for k in range(2, 6):
    check_model = KMeans(n_clusters=k, random_state=42, n_init=20)
    check_id = check_model.fit_predict(relative_projection)
    cluster_sizes = pd.Series(check_id).value_counts()

    cluster_number_rows.append(
        {
            "k": k,
            "inertia": check_model.inertia_,
            "silhouette": silhouette_score(relative_projection, check_id),
            "smallest_cluster": int(cluster_sizes.min()),
            "largest_cluster": int(cluster_sizes.max()),
            "chosen_main_result": k == n_clusters,
        }
    )

cluster_number_check = pd.DataFrame(cluster_number_rows)


model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
cluster_id = model.fit_predict(relative_projection)

# PCA for figure
pca = PCA(n_components=2, random_state=42)
pca_coordinates = pca.fit_transform(relative_projection)

neurons["q1_cluster"] = ["cluster_" + str(x + 1) for x in cluster_id]
neurons["PC1"] = pca_coordinates[:, 0]
neurons["PC2"] = pca_coordinates[:, 1]


mean_proj = relative_projection.copy()
mean_proj["q1_cluster"] = neurons["q1_cluster"]
mean_proj = mean_proj.groupby("q1_cluster").mean()




cluster_labels = []
for cluster_name in mean_proj.index:
    row = mean_proj.loc[cluster_name]
    strong_regions = []

    for region in regions:
        if row[region] >= label_cutoff:
            strong_regions.append(region)

    if len(strong_regions) == 0:
        strong_regions = list(row.sort_values(ascending=False).head(3).index)

    cluster_labels.append(
        {
            "cluster": cluster_name,
            "n_neurons": int((neurons["q1_cluster"] == cluster_name).sum()),
            "label": "/".join(strong_regions),
        }
    )

label_table = pd.DataFrame(cluster_labels)

breadth_summary = pd.crosstab(
    neurons["q1_cluster"],
    neurons["projection_breadth_type"],
    normalize="index",
)
for column in ["single-region", "multi-region"]:
    if column not in breadth_summary.columns:
        breadth_summary[column] = 0
breadth_summary = breadth_summary[["single-region", "multi-region"]]



columns_to_save = [
    "neuron_ID",
    "x",
    "y",
    "z",
    "injection",
    "proj",
    "n_target_regions",
    "projection_breadth_type",
    "q1_cluster",
]

neurons[columns_to_save].to_csv(
    output_folder / "neurons_with_connectivity_group.csv",
    index=False,
    float_format="%.2f",
)
mean_proj.to_csv(output_folder / "cluster_projection_proportion.csv", float_format="%.2f")
label_table.to_csv(output_folder / "connectivity_patterns.csv", index=False, float_format="%.2f")
breadth_summary.to_csv(output_folder / "single_or_multi_projection.csv", float_format="%.2f")


plt.figure(figsize=(7, 5))
for cluster_name in sorted(neurons["q1_cluster"].unique()):
    subset = neurons.loc[neurons["q1_cluster"] == cluster_name]
    plt.scatter(subset["PC1"], subset["PC2"], label=cluster_name, alpha=0.8)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Projection clusters")
plt.legend()
plt.tight_layout()
plt.savefig(output_folder / "clusters_pca.png", dpi=300)
plt.close()
plt.figure(figsize=(12, 3 + n_clusters))
plt.imshow(mean_proj, aspect="auto", cmap="Blues")
plt.colorbar(label="mean relative projection")
plt.yticks(range(len(mean_proj.index)), mean_proj.index)
plt.xticks(range(len(regions)), regions, rotation=90)
plt.title("Mean projection by cluster")
plt.tight_layout()
plt.savefig(output_folder / "cluster_projection_proportion.png", dpi=300)
plt.close()



# Plot k check
fig, axis1 = plt.subplots(figsize=(7, 4))
axis1.plot(
    cluster_number_check["k"],
    cluster_number_check["inertia"],
    marker="o",
    color="tab:blue",
)
axis1.set_xlabel("number of clusters")
axis1.set_ylabel("inertia", color="tab:blue")
axis1.tick_params(axis="y", labelcolor="tab:blue")

axis2 = axis1.twinx()
axis2.plot(
    cluster_number_check["k"],
    cluster_number_check["silhouette"],
    marker="s",
    color="tab:orange",
)
axis2.set_ylabel("silhouette score", color="tab:orange")
axis2.tick_params(axis="y", labelcolor="tab:orange")

axis1.axvline(n_clusters, color="grey", linestyle="--", linewidth=1)
plt.title("Cluster number check")
fig.tight_layout()
plt.savefig(output_folder / "cluster_number_check.png", dpi=300)
plt.close()





print("Q1 done")