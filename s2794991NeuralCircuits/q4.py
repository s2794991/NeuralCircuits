from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
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


# Control check to remove xyz and injection effects
control_columns = ["x", "y", "z"]
injection_dummy = pd.get_dummies(axon_expr["injection"], drop_first=True)
control_table = pd.concat([axon_expr[control_columns], injection_dummy], axis=1)
control_matrix = np.column_stack(
    [np.ones(len(control_table)), control_table.to_numpy(dtype=float)]
)

controlled_tests = []
rng = np.random.default_rng(42)
cluster_labels = axon_expr["q1_cluster"].to_numpy()
cluster_names = sorted(axon_expr["q1_cluster"].unique())


def cluster_f_stat(values, labels):
    grand_mean = values.mean()
    ss_between = 0
    ss_within = 0

    for cluster_name in cluster_names:
        group = values[labels == cluster_name]
        ss_between += len(group) * (group.mean() - grand_mean) ** 2
        ss_within += ((group - group.mean()) ** 2).sum()

    df_between = len(cluster_names) - 1
    df_within = len(values) - len(cluster_names)
    return (ss_between / df_between) / (ss_within / df_within)


for pc in local_pcs:
    # Whether local expression still not the same after the control step
    y = axon_expr[pc].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(control_matrix, y, rcond=None)
    residual = y - control_matrix @ beta

    residual_column = pc + "_residual"
    axon_expr[residual_column] = residual

    observed_f = cluster_f_stat(residual, cluster_labels)
    permutation_f = []
    for _ in range(5000):
        permutation_f.append(cluster_f_stat(residual, rng.permutation(cluster_labels)))

    permutation_f = np.asarray(permutation_f)
    p_value = (1 + (permutation_f >= observed_f).sum()) / (len(permutation_f) + 1)

    controlled_tests.append(
        {
            "expression_PC": pc.replace("local_", ""),
            "F_statistic": observed_f,
            "p_value_permutation": p_value,
        }
    )

controlled_pc_tests = pd.DataFrame(controlled_tests).sort_values("p_value_permutation")
controlled_pc_tests["q_value_permutation"] = false_discovery_control(
    controlled_pc_tests["p_value_permutation"],
    method="bh",
)
controlled_pc_tests = controlled_pc_tests.sort_values("q_value_permutation")


controlled_pc_tests_out = controlled_pc_tests.copy()
controlled_pc_tests_out["F_statistic"] = controlled_pc_tests_out["F_statistic"].round(2)
for col in ["p_value_permutation", "q_value_permutation"]:
    controlled_pc_tests_out[col] = controlled_pc_tests_out[col].map(lambda x: f"{x:.2e}")

controlled_pc_tests_out.to_csv(
    output_folder / "pc_anova_after_position_injection_control.csv",
    index=False,
)


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