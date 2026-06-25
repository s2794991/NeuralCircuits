from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import false_discovery_control, f_oneway
from sklearn.neighbors import NearestNeighbors


axon_file = Path("q1_outputs") / "neurons_with_connectivity_group.csv"
merfish_file = Path("q3_outputs") / "cells_pc.csv"
output_folder = Path("q4_outputs")
output_folder.mkdir(exist_ok=True)

# 10 nearby cells as average. The distance is not that far
k = 10


axon = pd.read_csv(axon_file)
merfish = pd.read_csv(merfish_file)

pc_cols = []
for column in merfish.columns:
    if column.startswith("expression_PC"):
        pc_cols.append(column)


nearest_cells = NearestNeighbors(n_neighbors=k)
nearest_cells.fit(merfish[["x", "y", "z"]])
distances, neighbour_index = nearest_cells.kneighbors(axon[["x", "y", "z"]])


# Each SNR neuron gets the average expression PC though 10 nearest cells
matched = []
for neuron_number in range(len(axon)):
    neighbours = merfish.iloc[neighbour_index[neuron_number]]

    row = {
        "nearest_distance": distances[neuron_number, 0],
        "kth_neighbour_distance": distances[neuron_number, k - 1],
    }

    for pc in pc_cols:
        row["local_" + pc] = neighbours[pc].mean()

    matched.append(row)

local_expr = pd.DataFrame(matched)
axon_expr = pd.concat([axon.reset_index(drop=True), local_expr], axis=1)
axon_expr.to_csv(output_folder / "neuron_local_pc.csv", index=False, float_format="%.2f")


local_pcs = []
for pc in pc_cols:
    local_pcs.append("local_" + pc)

cluster_pcs = axon_expr.groupby("q1_cluster")[local_pcs].mean()
cluster_pcs_for_table = cluster_pcs.copy()
cluster_pcs_for_table.columns = [col.replace("local_expression_", "") for col in cluster_pcs_for_table.columns]
cluster_pcs_for_table.to_csv(output_folder / "cluster_local_pc.csv", float_format="%.2f")


tests = []
for pc in local_pcs:
    groups = []
    for cluster_name in sorted(axon_expr["q1_cluster"].unique()):
        groups.append(axon_expr.loc[axon_expr["q1_cluster"] == cluster_name, pc])

    result = f_oneway(*groups)
    tests.append(
        {
            "expression_PC": pc.replace("local_", ""),
            "F_statistic": result.statistic,
            "p_value": result.pvalue,
        }
    )

pc_tests = pd.DataFrame(tests).sort_values("p_value")

# FDR instead of p
pc_tests["q_value"] = false_discovery_control(pc_tests["p_value"], method="bh")
pc_tests = pc_tests.sort_values("q_value")

pc_tests_out = pc_tests.copy()
pc_tests_out["F_statistic"] = pc_tests_out["F_statistic"].round(2)
for col in ["p_value", "q_value"]:
    pc_tests_out[col] = pc_tests_out[col].map(lambda x: f"{x:.2e}")

pc_tests_out.to_csv(output_folder / "pc_anova.csv", index=False)


pc_pair = list(pc_tests.head(2)["expression_PC"])
x_pc = "local_" + pc_pair[0]
y_pc = "local_" + pc_pair[1]

plt.figure(figsize=(7, 5))
for cluster_name in sorted(axon_expr["q1_cluster"].unique()):
    subset = axon_expr.loc[axon_expr["q1_cluster"] == cluster_name]
    plt.scatter(subset[x_pc], subset[y_pc], label=cluster_name, alpha=0.8)

plt.xlabel(pc_pair[0].replace("_", " "))
plt.ylabel(pc_pair[1].replace("_", " "))
plt.title("Local expression PCs by projection cluster")
plt.legend()
plt.tight_layout()
plt.savefig(output_folder / "pc_cluster.png", dpi=300)
plt.close()



print("Q4 done")