import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sim_matrix: (N x N) numpy array of cosine similarities
# patient_labels: list of patient IDs, sorted by cluster
# cluster_boundaries: list of indices where clusters start

fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(sim_matrix, ax=ax, cmap='Blues', vmin=0.4, vmax=1.0,
            xticklabels=patient_labels, yticklabels=patient_labels,
            linewidths=0, square=True, cbar_kws={'label': 'Cosine Similarity'})

# Draw cluster boundary lines
for b in cluster_boundaries:
    ax.axhline(b, color='white', linewidth=2)
    ax.axvline(b, color='white', linewidth=2)

ax.set_title('Pairwise Patient Similarity Matrix (BioClinicalBERT Embeddings)', fontsize=13)
plt.tight_layout()
plt.savefig('pairwise_similarity_heatmap.png', dpi=300)