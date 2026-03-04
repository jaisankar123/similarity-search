import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

# =====================================================
# 1. LOAD CONFIG AND DATA FROM MONGO
# =====================================================
load_dotenv()


def fetch_embeddings_from_mongo():
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    client = MongoClient(uri)
    db = client[os.getenv("DB_NAME")]
    coll = db[os.getenv("EMBEDDINGS_COLLECTION")]

    cursor = coll.find({}, {"embedding": 1, "p_id": 1})
    embeddings, p_ids = [], []

    for doc in cursor:
        embeddings.append(doc["embedding"])
        p_ids.append(doc["p_id"])

    return np.array(embeddings), p_ids, coll


# =====================================================
# 2. DYNAMIC K SELECTION (STATISTICAL)
# =====================================================
def find_optimal_k(data, max_k=10):
    """Statistically determines the best K using Silhouette Score."""
    print("Finding optimal number of clusters (K)...")
    sil_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(data)
        score = silhouette_score(data, labels)
        sil_scores.append(score)
        print(f"Testing K={k} | Silhouette Score: {score:.4f}")

    best_k = k_range[np.argmax(sil_scores)]
    print(f"\n>>> Statistically Optimal K selected: {best_k}\n")
    return best_k


# =====================================================
# 3. DEC MODEL COMPONENTS
# =====================================================
class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, latent_dim=10, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.Tensor(n_clusters, latent_dim))

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.clusters) ** 2, 2)
        q = 1.0 / (1.0 + norm_squared / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


class DEC(nn.Module):
    def __init__(self, n_clusters, input_dim=768):
        super(DEC, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, 10),
        )
        self.clustering_layer = ClusteringLayer(n_clusters, latent_dim=10)

    def forward(self, x):
        z = self.encoder(x)
        return self.clustering_layer(z), z


# =====================================================
# 4. TRAINING LOGIC
# =====================================================
def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_dec(data, n_clusters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tensor = torch.tensor(data, dtype=torch.float).to(device)

    model = DEC(n_clusters=n_clusters, input_dim=data.shape[1]).to(device)

    print(f"Initializing DEC centers for K={n_clusters} on {device}...")
    with torch.no_grad():
        initial_z = model.encoder(x_tensor).cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        kmeans.fit(initial_z)
        model.clustering_layer.clusters.data = torch.tensor(kmeans.cluster_centers_).to(
            device
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.KLDivLoss(reduction="batchmean")
    model.train()

    for epoch in range(101):
        q, _ = model(x_tensor)
        p = target_distribution(q).detach()

        optimizer.zero_grad()
        loss = criterion(q.log(), p)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch} | KL Loss: {loss.item():.6f}")
    return model


# =====================================================
# 5. SYNC RESULTS TO MONGODB
# =====================================================
def sync_clusters_to_db(collection, p_ids, labels):
    """
    Uses Bulk Write to efficiently update MongoDB documents with cluster labels.
    """
    print(f"Syncing {len(p_ids)} labels back to MongoDB...")
    updates = []

    for p_id, label in zip(p_ids, labels):
        # We use UpdateOne to target the patient ID and set the new cluster field
        updates.append(
            UpdateOne({"p_id": p_id}, {"$set": {"dec_cluster_label": int(label)}})
        )

    if updates:
        result = collection.bulk_write(updates)
        print(f"Done! Modified {result.modified_count} documents.")


# =====================================================
# 6. VISUALIZATION AND EVALUATION
# =====================================================
def evaluate_and_visualize(latent_features, cluster_labels, best_k):
    sil_score = silhouette_score(latent_features, cluster_labels)
    db_index = davies_bouldin_score(latent_features, cluster_labels)

    print("\n--- FINAL PERFORMANCE METRICS ---")
    print(f"Silhouette Score (↑): {sil_score:.4f}")
    print(f"Davies-Bouldin Index (↓): {db_index:.4f}")

    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(latent_features)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=pca_results[:, 0],
        y=pca_results[:, 1],
        hue=cluster_labels,
        palette="viridis",
        s=70,
        alpha=0.8,
    )
    plt.title(f"Final Patient Clusters (K={best_k})\nSil: {sil_score:.2f}")
    plt.show()


# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":
    features, patient_ids, collection = fetch_embeddings_from_mongo()

    if len(features) < 3:
        print("Insufficient data for clustering.")
    else:
        # 1. Dynamically find best K
        optimal_k = find_optimal_k(features, max_k=10)

        # 2. Train DEC
        dec_model = train_dec(features, n_clusters=optimal_k)

        # 3. Inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dec_model.eval()
        with torch.no_grad():
            x_input = torch.tensor(features).float().to(device)
            q, z = dec_model(x_input)

            latent_z = z.cpu().numpy()
            # Assign final labels based on highest probability
            final_clusters = torch.argmax(q, dim=1).cpu().numpy()

        # 4. Sync Labels to Database
        sync_clusters_to_db(collection, patient_ids, final_clusters)

        # 5. Visualize
        evaluate_and_visualize(latent_z, final_clusters, optimal_k)

        # tell me this answer
        print(
            f"silouette for original features {silhouette_score(features, final_clusters)}"
        )
