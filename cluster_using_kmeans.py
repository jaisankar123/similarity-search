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
from kneed import KneeLocator 

# ======================================================
# 1. CONFIGURATION & DB INITIALIZATION
# ======================================================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
EMBEDDINGS_COLL = os.getenv("EMBEDDINGS_COLLECTION")
PATIENT_COLL = os.getenv("PATIENT_COLLECTION")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
embeddings_col = db[EMBEDDINGS_COLL]
patient_col = db[PATIENT_COLL]

# ======================================================
# 2. DATA FETCHING (JOINING COLLECTIONS)
# ======================================================
def fetch_patient_details(p_ids):
    """Fetch conditions from the patient collection"""
    cursor = patient_col.find({"p_id": {"$in": p_ids}}, {"p_id": 1, "conditions": 1})
    details = {}
    for doc in cursor:
        conds = doc.get("conditions", [])
        if isinstance(conds, list) and len(conds) > 0:
            label = conds[0].get("item", "Unknown") if isinstance(conds[0], dict) else str(conds[0])
        else:
            label = "Unknown"
        details[doc["p_id"]] = label
    return details

def fetch_batch(skip, limit):
    """Fetches embeddings and joins with clinical labels"""
    cursor = list(embeddings_col.find({}, {"p_id": 1, "embedding": 1}).skip(skip).limit(limit))
    p_ids = [d["p_id"] for d in cursor]
    condition_map = fetch_patient_details(p_ids)
    
    for doc in cursor:
        doc["condition"] = condition_map.get(doc["p_id"], "Unknown")
    return cursor

def load_data_parallel(batch_size=3000, max_workers=8):
    total_docs = embeddings_col.count_documents({})
    print(f"Total documents to fetch: {total_docs}")
    
    all_data = []
    offsets = range(0, total_docs, batch_size)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_offset = {executor.submit(fetch_batch, offset, batch_size): offset for offset in offsets}
        for future in tqdm(as_completed(future_to_offset), total=len(offsets), desc="📥 Downloading & Joining"):
            all_data.extend(future.result())
            
    df = pd.DataFrame(all_data)
    X = np.array(df['embedding'].tolist())
    p_ids = df['p_id'].tolist()
    conditions = df['condition'].tolist()
    return X, p_ids, conditions

# ======================================================
# 3. CLUSTERING & OPTIMIZATION
# ======================================================
def calculate_k_metrics(k, X_pca):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels, sample_size=2000, random_state=42)
    ch = calinski_harabasz_score(X_pca, labels)
    db = davies_bouldin_score(X_pca, labels)
    return k, kmeans.inertia_, sil, ch, db

def perform_clustering(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.95) 
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA reduced dimensions to {X_pca.shape[1]}")
    
    k_range = list(range(2, 11))
    results = []
    print("Evaluating metrics via ThreadPool (Memory Safe)...")
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
        futures = [executor.submit(calculate_k_metrics, k, X_pca) for k in k_range]
        for future in tqdm(as_completed(futures), total=len(k_range), desc="⚙️ Optimizing"):
            results.append(future.result())

    results.sort(key=lambda x: x[0])
    _, inertias, sil_scores, ch_scores, db_scores = zip(*results)
    kn = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
    final_k = kn.elbow if kn.elbow else 3
    print(f"\nOptimal K selected: {final_k}")

    final_kmeans = KMeans(n_clusters=final_k, init='k-means++', n_init=10, random_state=42)
    final_labels = final_kmeans.fit_predict(X_pca)
    
    final_metrics = {
        "silhouette": sil_scores[k_range.index(final_k)],
        "calinski": ch_scores[k_range.index(final_k)],
        "davies": db_scores[k_range.index(final_k)]
    }
    return final_labels, X_pca, final_k, k_range, inertias, final_metrics

# ======================================================
# 4. MAIN EXECUTION
# ======================================================
def main():
    # 1. Data Fetching and Clustering
    X, p_ids, conditions = load_data_parallel()
    labels, X_pca, k, k_range, inertias, metrics = perform_clustering(X)

    # 2. Initialize Analysis DataFrame early to avoid UnboundLocalError
    analysis_df = pd.DataFrame({
        'p_id': p_ids,
        'Cluster': labels,
        'Condition': conditions
    })

    # 3. Fetch Demographic Data (Age and City)
    print("📊 Fetching demographic data for advanced analysis...")
    demo_cursor = patient_col.find({"p_id": {"$in": p_ids}}, {"p_id": 1, "age": 1, "city": 1})
    demo_map = {d["p_id"]: {"age": d.get("age", 0), "city": d.get("city", "Unknown")} for d in demo_cursor}
    
    # Map demographic data back to analysis_df
    analysis_df['Age'] = analysis_df['p_id'].map(lambda x: demo_map.get(x, {}).get("age", 0))
    analysis_df['City'] = analysis_df['p_id'].map(lambda x: demo_map.get(x, {}).get("city", "Unknown"))
    analysis_df['Age'] = pd.to_numeric(analysis_df['Age'], errors='coerce').fillna(0)
    
    if not os.path.exists('analysis_plots'):
        os.makedirs('analysis_plots')

    # ======================================================
    # BASIC CLUSTERING PLOTS (1-9)
    # ======================================================
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, 'go-')
    plt.title('1. Elbow Method')
    plt.savefig('analysis_plots/01_elbow_method.png')
    plt.close()

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=30, alpha=0.5)
    plt.title(f'2. Patient Clusters (k={k})')
    plt.savefig('analysis_plots/02_cluster_scatter.png')
    plt.close()

    top_15_conds = pd.Series(conditions).value_counts().nlargest(15).index
    
    plt.figure(figsize=(12, 7))
    filtered_df = analysis_df[analysis_df['Condition'].isin(top_15_conds)]
    pivot_pct = filtered_df.groupby(['Cluster', 'Condition']).size().unstack(fill_value=0)
    pivot_pct = pivot_pct.div(pivot_pct.sum(axis=1), axis=0) * 100
    pivot_pct.plot(kind='bar', stacked=True, colormap='tab20', ax=plt.gca())
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small')
    plt.title('7. Clinical Composition (Top 15 Conditions)')
    plt.tight_layout()
    plt.savefig('analysis_plots/07_condition_composition.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    ctab = pd.crosstab(analysis_df['Cluster'], analysis_df['Condition'])
    sns.heatmap(ctab.div(ctab.sum(axis=1), axis=0), cmap='YlGnBu')
    plt.title('9. Cluster Purity Heatmap')
    plt.savefig('analysis_plots/09_purity_heatmap.png')
    plt.close()

    # ======================================================
    # ADVANCED DEMOGRAPHIC PLOTS (10-13)
    # ======================================================
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Cluster', y='Age', data=analysis_df, palette='muted', inner="quartile")
    plt.title('10. Age Demographics per Patient Cluster')
    plt.savefig('analysis_plots/10_age_violin_distribution.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 8))
    top_cities = analysis_df['City'].value_counts().nlargest(10).index
    city_df = analysis_df[analysis_df['City'].isin(top_cities)]
    city_ctab = pd.crosstab(city_df['Cluster'], city_df['City'])
    sns.heatmap(city_ctab, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('11. Geographic Cluster Concentration (Top 10 Cities)')
    plt.savefig('analysis_plots/11_city_cluster_heatmap.png', dpi=300)
    plt.close()

    plt.figure(figsize=(14, 8))
    bubble_df = analysis_df[analysis_df['Condition'].isin(top_15_conds[:10])]
    bubble_data = bubble_df.groupby(['Age', 'Condition', 'Cluster']).size().reset_index(name='PatientCount')
    sns.scatterplot(data=bubble_data, x='Age', y='Condition', size='PatientCount', 
                    hue='Cluster', sizes=(20, 500), alpha=0.6, palette='viridis')
    plt.title('12. Age vs Condition Correlation within Clusters')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Cluster')
    plt.tight_layout()
    plt.savefig('analysis_plots/12_age_condition_bubbles.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 7))
    top_5_cities = analysis_df['City'].value_counts().nlargest(5).index
    top_5_conds_list = analysis_df['Condition'].value_counts().nlargest(5).index
    special_df = analysis_df[analysis_df['City'].isin(top_5_cities) & analysis_df['Condition'].isin(top_5_conds_list)]
    pivot_special = special_df.groupby(['City', 'Condition']).size().unstack(fill_value=0)
    pivot_special.plot(kind='bar', stacked=True, colormap='Set3', ax=plt.gca())
    plt.title('13. Condition Prevalence across Top 5 Cities')
    plt.ylabel('Patient Count')
    plt.xticks(rotation=45)
    plt.legend(title='Condition', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('analysis_plots/13_city_condition_stack.png', dpi=300)
    plt.close()

    print("\n" + "="*35)
    print(f"      CLUSTERING RESULTS (k={k})")
    print("="*35)
    print(f"Silhouette Score:   {metrics['silhouette']:.4f}")
    print(f"Calinski-Harabasz:  {metrics['calinski']:.2f}")
    print(f"Davies-Bouldin:     {metrics['davies']:.4f}")
    print("="*35)
    print(f"\n✅ All 13 plots saved. Process complete.")

if __name__ == "__main__":
    main()