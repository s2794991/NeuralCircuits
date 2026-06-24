from pathlib import Path
import matplotlib.pyplot as plt
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
coordinate_test["p_adj"] = (coordinate_test["p_value"] * len(coordinates)).clip(upper=1)

coordinate_test_out = coordinate_test.copy()
coordinate_test_out["F_statistic"] = coordinate_test_out["F_statistic"].round(2)
for col in ["p_value", "p_adj"]:
    coordinate_test_out[col] = coordinate_test_out[col].map(lambda x: f"{x:.2e}")

coordinate_test_out.to_csv(output_folder / "soma_position_anova.csv", index=False)






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



injection_table = pd.crosstab(neurons["q1_cluster"], neurons["injection"], normalize="index")
injection_table.to_csv(output_folder / "injection_by_cluster.csv", float_format="%.2f")







print("Q2 done")