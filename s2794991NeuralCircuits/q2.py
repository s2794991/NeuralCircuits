from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway



cluster_file = Path("q1_outputs") / "neurons_with_connectivity_group.csv"
output_folder = Path("q2_outputs")
output_folder.mkdir(exist_ok=True)


neurons = pd.read_csv(cluster_file)
neurons = neurons[["neuron_ID", "x", "y", "z", "q1_cluster", "injection", "proj"]].copy()


coordinates = ["x", "y", "z"]
clusters = sorted(neurons["q1_cluster"].unique())

coordinate_rows = []
for cluster_name in clusters:
    subset = neurons.loc[neurons["q1_cluster"] == cluster_name]
    coordinate_rows.append(
        {
            "q1_cluster": cluster_name,
            "n_neurons": len(subset),
            "x_mean": subset["x"].mean(),
            "y_mean": subset["y"].mean(),
            "z_mean": subset["z"].mean(),
            "x_median": subset["x"].median(),
            "y_median": subset["y"].median(),
            "z_median": subset["z"].median(),
        }
    )

coordinate_table = pd.DataFrame(coordinate_rows)
coordinate_table.to_csv(
    output_folder / "soma_position_by_cluster.csv",
    index=False,
    float_format="%.2f",
)


rows = []
for coordinate in coordinates:
    groups = []
    for cluster_name in clusters:
        values = neurons.loc[neurons["q1_cluster"] == cluster_name, coordinate]
        groups.append(values)

    result = f_oneway(*groups)
    rows.append(
        {
            "coordinate": coordinate,
            "F_statistic": result.statistic,
            "p_value": result.pvalue,
        }
    )

coordinate_test = pd.DataFrame(rows)


# 3 tests, Bonferroni correction
coordinate_test["p_adj"] = (coordinate_test["p_value"] * len(coordinates)).clip(upper=1)

coordinate_test_out = coordinate_test.copy()
coordinate_test_out["F_statistic"] = coordinate_test_out["F_statistic"].round(2)
for col in ["p_value", "p_adj"]:
    coordinate_test_out[col] = coordinate_test_out[col].map(lambda x: f"{x:.2e}")

coordinate_test_out.to_csv(output_folder / "soma_position_anova.csv", index=False)


# Classify check can soma xyz predict Q1 clusters; nearest-centroid classifier
n_folds = 5
rng = np.random.default_rng(42)
fold_id = pd.Series(index=neurons.index, dtype=int)


for cluster_name in clusters:
    cluster_index = neurons.loc[neurons["q1_cluster"] == cluster_name].index.to_numpy()
    cluster_index = rng.permutation(cluster_index)

    for place, row_index in enumerate(cluster_index):
        fold_id.loc[row_index] = place % n_folds

predicted_cluster = pd.Series(index=neurons.index, dtype=object)


for fold in range(n_folds):
    train_rows = fold_id != fold
    test_rows = fold_id == fold

    train_coordinates = neurons.loc[train_rows, coordinates]
    coordinate_mean = train_coordinates.mean()
    coordinate_std = train_coordinates.std(ddof=0).replace(0, 1)

    scaled_train = (train_coordinates - coordinate_mean) / coordinate_std
    scaled_test = (neurons.loc[test_rows, coordinates] - coordinate_mean) / coordinate_std

    centroids = {}
    for cluster_name in clusters:
        in_cluster = neurons.loc[train_rows, "q1_cluster"] == cluster_name
        centroids[cluster_name] = scaled_train.loc[in_cluster].mean().to_numpy()

    for row_index, row in scaled_test.iterrows():
        distances = {}
        for cluster_name in clusters:
            distances[cluster_name] = np.linalg.norm(row.to_numpy() - centroids[cluster_name])

        predicted_cluster.loc[row_index] = min(distances, key=distances.get)

observed_cluster = neurons["q1_cluster"]

# Compare
confusion = pd.crosstab(
    observed_cluster,
    predicted_cluster,
    rownames=["observed_cluster"],
    colnames=["predicted_cluster"],
).reindex(index=clusters, columns=clusters, fill_value=0)


accuracy = (observed_cluster == predicted_cluster).mean()
majority_cluster = observed_cluster.value_counts().idxmax()
majority_accuracy = (observed_cluster == majority_cluster).mean()


pairs = [("x", "y"), ("x", "z"), ("y", "z")]
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for axis, pair in zip(axes, pairs):
    x_axis, y_axis = pair

    for cluster_name in clusters:
        subset = neurons.loc[neurons["q1_cluster"] == cluster_name]
        axis.scatter(subset[x_axis], subset[y_axis], label=cluster_name, alpha=0.8)

    axis.set_xlabel(x_axis)
    axis.set_ylabel(y_axis)
    axis.set_title(x_axis + " vs " + y_axis)

axes[0].legend()
plt.tight_layout()
plt.savefig(output_folder / "soma_position_by_cluster.png", dpi=300)
plt.close()




print("Q2 done")