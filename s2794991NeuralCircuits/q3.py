from pathlib import Path
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression




h5ad_file = Path("data") / "ec_obj_imputed_log2.h5ad"
output_folder = Path("q3_outputs")
output_folder.mkdir(exist_ok=True)

n_pcs = 10
n_genes_to_show = 10


adata = ad.read_h5ad(h5ad_file)

position = adata.obs[["x_ccf", "y_ccf", "z_ccf"]].copy()
position.columns = ["x", "y", "z"]





for column in ["x", "y", "z"]:
    position[column] = position[column].astype(float)


expression = adata.X

# The first 10 PCs are used as a compact summary of the expression matrix.
pca = PCA(n_components=n_pcs, random_state=42)
expression_pcs = pca.fit_transform(expression)





pc_names = []
for i in range(n_pcs):
    pc_names.append("expression_PC" + str(i + 1))

pc_table = pd.DataFrame(expression_pcs, columns=pc_names, index=position.index)
pc_table = pd.concat([position, pc_table], axis=1)
pc_table.to_csv(output_folder / "cells_pc.csv", float_format="%.2f")

pca_variance = pd.DataFrame(
    {
        "PC": pc_names,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }
)
pca_variance.to_csv(output_folder / "pc_variance.csv", index=False, float_format="%.2f")





coordinate_model = LinearRegression()
pc_rows = []
for pc in pc_names:
    coordinate_model.fit(position[["x", "y", "z"]], pc_table[pc])
    r2 = coordinate_model.score(position[["x", "y", "z"]], pc_table[pc])
    pc_rows.append({"expression_PC": pc, "R2_from_coordinates": r2})

pc_test = pd.DataFrame(pc_rows)
pc_test.to_csv(output_folder / "pc_r2.csv", index=False, float_format="%.2f")






gene_tests = []
for gene_number, gene in enumerate(adata.var_names):
    values = np.asarray(expression[:, gene_number]).ravel()

    for axis in ["x", "y", "z"]:
        r, p = spearmanr(position[axis], values)
        gene_tests.append({"gene": gene, "axis": axis, "spearman": r, "p_value": p})

gene_table = pd.DataFrame(gene_tests)
gene_table["q_value"] = false_discovery_control(gene_table["p_value"], method="bh")
gene_table["abs_spearman"] = gene_table["spearman"].abs()

top_rows = []
genes_seen = set()
for _, row in gene_table.sort_values("abs_spearman", ascending=False).iterrows():
    if row["gene"] not in genes_seen:
        top_rows.append(row)
        genes_seen.add(row["gene"])
    if len(top_rows) == n_genes_to_show:
        break

top_genes = pd.DataFrame(top_rows)[["gene", "axis", "spearman", "q_value"]]
top_genes_out = top_genes.copy()
top_genes_out["spearman"] = top_genes_out["spearman"].round(2)
top_genes_out["q_value"] = top_genes_out["q_value"].map(lambda x: f"{x:.2e}")
top_genes_out.to_csv(output_folder / "top_spatial_genes.csv", index=False)


plot_pc = pc_test.sort_values("R2_from_coordinates", ascending=False).iloc[0]["expression_PC"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for axis, coordinate in zip(axes, ["x", "y", "z"]):
    axis.scatter(pc_table[coordinate], pc_table[plot_pc], alpha=0.4)
    axis.set_xlabel(coordinate)
    axis.set_ylabel(plot_pc)
    axis.set_title(coordinate + " vs " + plot_pc)

plt.tight_layout()
plt.savefig(output_folder / "pc_position.png", dpi=300)
plt.close()







print("Q3 done")