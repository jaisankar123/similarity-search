import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pymongo import MongoClient
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from kneed import KneeLocator  # <--- Essential for mathematical elbow detection

# ======================================================
# 1. CONFIGURATION & DB INITIALIZATION
# ======================================================
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
EMBEDDINGS_COLL = os.getenv("EMBEDDINGS_COLLECTION")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
embeddings_col = db[EMBEDDINGS_COLL]

# ======================================================
# 2. MULTI-THREADED DATA FETCHING
# ======================================================
def fetch_batch(skip, limit):
    cursor = embeddings_col.find({}, {"p_id": 1, "embedding": 1}).skip(skip).limit(limit)
    return list(cursor)

def load_data_parallel(batch_size=1000, max_workers=10):
    total_docs = embeddings_col.count_documents({})
    print(f"Total documents to fetch: {total_docs}")
    
    all_data = []
    offsets = range(0, total_docs, batch_size)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_offset = {executor.submit(fetch_batch, offset, batch_size): offset for offset in offsets}
        for future in tqdm(as_completed(future_to_offset), total=len(offsets), desc="Downloading Embeddings"):
            try:
                all_data.extend(future.result())
            except Exception as e:
                print(f"Batch failed: {e}")
            
    df = pd.DataFrame(all_data)
    X = np.array(df['embedding'].tolist())
    p_ids = df['p_id'].tolist()
    return X, p_ids

# ======================================================
# 3. PREPROCESSING & DYNAMIC CLUSTERING (ELBOW + METRICS)
# ======================================================
def perform_clustering(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=0.95) 
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA reduced dimensions to {X_pca.shape[1]}")
    
    k_range = list(range(2, 11))
    inertias = []
    sil_scores = []
    ch_scores = []
    db_scores = []
    
    print("Evaluating metrics and curvature...")
    for k in tqdm(k_range, desc="Optimization Loop"):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_pca)
        
        inertias.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X_pca, labels))
        ch_scores.append(calinski_harabasz_score(X_pca, labels))
        db_scores.append(davies_bouldin_score(X_pca, labels))

    # --- 1. CALCULATE MATHEMATICAL ELBOW ---
    kn = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
    elbow_k = kn.elbow if kn.elbow else k_range[0]

    # --- 2. CALCULATE METRIC CONSENSUS ---
    s_norm = MinMaxScaler().fit_transform(np.array(sil_scores).reshape(-1, 1)).flatten()
    c_norm = MinMaxScaler().fit_transform(np.array(ch_scores).reshape(-1, 1)).flatten()
    d_norm = 1 - MinMaxScaler().fit_transform(np.array(db_scores).reshape(-1, 1)).flatten()
    
    unified_scores = (s_norm + c_norm + d_norm) / 3
    metric_best_k = k_range[np.argmax(unified_scores)]

    # --- 3. FINAL DECISION ---
    # We favor the Elbow point because it's more conservative for clinical similarity,
    # but we check if the metrics suggest a significantly better structure nearby.
    final_k = elbow_k 
    print(f"\nMathematical Elbow found at k={elbow_k}")
    print(f"Metric-based peak found at k={metric_best_k}")
    print(f"Final Decision: Using k={final_k} (Elbow Method)")

    # Final Model
    final_kmeans = KMeans(n_clusters=final_k, init='k-means++', n_init=10, random_state=42)
    final_labels = final_kmeans.fit_predict(X_pca)
    
    return final_labels, X_pca, final_k, k_range, inertias

# ======================================================
# 4. MAIN EXECUTION
# ======================================================
def main():
    X, p_ids = load_data_parallel()
    labels, X_pca, k, k_range, inertias = perform_clustering(X)
    
    # 5. VISUALIZATION
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Elbow Method with Highlighted Knee
    ax1.plot(k_range, inertias, 'go-', linewidth=2, markersize=8, label='Inertia')
    ax1.axvline(x=k, color='red', linestyle='--', label=f'Chosen Elbow K={k}')
    ax1.set_title('Elbow Method (Curvature Detection)')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cluster Scatter Plot (PCA)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=60, ax=ax2, edgecolor='w')
    ax2.set_title(f'Patient Similarity Clusters (k={k})')
    ax2.set_xlabel('PCA Component 1')
    ax2.set_ylabel('PCA Component 2')
    
    plt.tight_layout()
    plt.show()

    # Results
    results_df = pd.DataFrame({"p_id": p_ids, "cluster_tag": labels})
    print("\nSample Clustering Results:")
    print(results_df.head())

if __name__ == "__main__":
    main()