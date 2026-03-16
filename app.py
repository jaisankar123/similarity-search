import streamlit as st
import pandas as pd
import faiss
import json
import numpy as np
import os
import time
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv
from streamlit_lottie import st_lottie

# ======================================================
# 1. CONFIGURATION & STYLING
# ======================================================
st.set_page_config(page_title="FHIR Patient Similarity", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    
    /* Search Bar Layout */
    .stButton>button { 
        margin-top: 28px; 
        background-color: #6c63ff; 
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 8px;
    }
    
    /* Loading Area Background */
    .loading-container {
        background-color: #1a1a1a; /* Dark background for the ECG animation */
        border-radius: 15px;
        padding: 20px;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        color: white;
        margin-bottom: 20px;
    }

    /* Patient Detail Card: White background, Black text */
    .patient-card { 
        background-color: white !important; 
        color: black !important;
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        margin-bottom: 20px;
        border: 1px solid #ddd;
    }
    .patient-card strong, .patient-card p {
        color: black !important;
    }

    /* Table styling with visible borders */
    .styled-table-container {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ccc;
    }
    .styled-table-container table { color: black !important; }
    .styled-table-container td, .styled-table-container th {
        border: 1px solid #ddd !important;
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# ======================================================
# 2. DATA & ASSET LOADERS
# ======================================================
def load_lottie_file(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except:
        return None

@st.cache_resource
def init_resources():
    faiss_dir = Path(os.getenv("FAISS_INDEX_PATH", BASE_DIR / "faiss"))
    index = faiss.read_index(str(faiss_dir / "patient.index"))
    with open(str(faiss_dir / "index_mapping.json"), "r") as f:
        mapping = json.load(f)
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("DB_NAME")]
    collection = db[os.getenv("NARRATIVE_COLLECTION")]
    return index, mapping, collection

# Asset Initialization
dna_helix = load_lottie_file("search.json")
success_anim = load_lottie_file("success.json")
heartbeat_loader = load_lottie_file("search.json")

index, mapping, collection = init_resources()
idx_to_p_id = {int(k): v["p_id"] for k, v in mapping.items()}

# ======================================================
# 3. SEARCH LOGIC
# ======================================================
def get_patient_similarity(query_id, k=6):
    target_data = collection.find_one({"p_id": query_id})
    if not target_data:
        return None, None

    emb_coll = MongoClient(os.getenv("MONGO_URI"))[os.getenv("DB_NAME")][os.getenv("EMBEDDINGS_COLLECTION")]
    vector_data = emb_coll.find_one({"p_id": query_id})
    if not vector_data:
        return None, None

    query_vector = np.array([vector_data["embedding"]], dtype="float32")
    faiss.normalize_L2(query_vector)
    scores, ids = index.search(query_vector, k)
    
    results = []
    for i in range(len(ids[0])):
        hit_idx = ids[0][i]
        score = scores[0][i]
        if hit_idx == -1: continue
        hit_p_id = idx_to_p_id.get(int(hit_idx))
        patient_doc = collection.find_one({"p_id": hit_p_id})
        results.append({
            "Patient ID": hit_p_id,
            "Similarity Score": round(float(score), 4),
            "Clinical Narrative": patient_doc.get("clinical_narrative", "N/A")
        })
    return results[0], results[1:]

# ======================================================
# 4. UI LAYOUT
# ======================================================
with st.sidebar:
    if dna_helix:
        st_lottie(dna_helix, speed=1, height=200)
    st.title("Settings")
    top_k = st.slider("Number of neighbors", 1, 10, 5)

st.title("🧬 FHIR Patient Similarity Finder")
st.write("Search clinically similar patients using Bio-ClinicalBERT embeddings.")

# Search bar logic (Half screen width)
col_search_1, col_search_2, col_spacer = st.columns([2, 1, 3]) 

with col_search_1:
    query_id_input = st.text_input("Enter Patient ID (e.g., 101):", "")

with col_search_2:
    search_clicked = st.button("Search Similar Patients")

# --- Results Area ---
if search_clicked:
    if query_id_input:
        try:
            query_id = int(query_id_input)
            
            # 1. Loading Animation (ECG Heartbeat)
            placeholder = st.empty()
            with placeholder.container():
                st.markdown('<div class="loading-container">', unsafe_allow_html=True)
                if heartbeat_loader:
                    st_lottie(heartbeat_loader, speed=1, height=300, key="loader")
                st.markdown('<h3>Analyzing Clinical Patterns...</h3></div>', unsafe_allow_html=True)
                
            # Perform Search
            target, neighbors = get_patient_similarity(query_id, k=top_k + 1)
            time.sleep(1.5) # Ensuring the animation is visible to the user
            
            # Clear loader
            placeholder.empty()

            if target:
                # 2. Success Animation
                if success_anim:
                    st_lottie(success_anim, speed=1, loop=False, height=150)
                
                # 3. Display Results
                st.subheader("🎯 Selected Patient Details")
                st.markdown(f"""
                <div class="patient-card">
                    <p><strong>Patient ID:</strong> {target['Patient ID']}</p>
                    <p><strong>Clinical Narrative:</strong> {target['Clinical Narrative']}</p>
                </div>
                """, unsafe_allow_html=True)

                st.subheader(f"👥 Top {top_k} Similar Patients")
                df_results = pd.DataFrame(neighbors)
                
                st.markdown('<div class="styled-table-container">', unsafe_allow_html=True)
                st.table(df_results)
                st.markdown('</div>', unsafe_allow_html=True)

                # Download Button
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"similar_patients_{query_id}.csv",
                    mime="text/csv",
                )
                
            else:
                st.error("Patient ID not found in database.")
        except ValueError:
            st.error("Please enter a valid numerical Patient ID.")
    else:
        st.warning("Please enter a Patient ID to proceed.")

# ======================================================
# 5. VISUALIZATIONS SECTION
# ======================================================
st.divider()
st.subheader("📊 Clinical Population Analytics")
show_viz = st.button("Toggle Clustering Visualizations")

if show_viz:
    plot_path = Path("analysis_plots")
    
    if not plot_path.exists():
        st.error("Analysis plots not found. Please run the clustering script first.")
    else:
        # Define the plots we want to show and their titles
        viz_files = [
            ("01_elbow_method.png", "Mathematical Optimization (Elbow)"),
            ("02_cluster_scatter.png", "2D Patient Projection (PCA)"),
            ("03_cluster_distribution.png", "Population per Cluster"),
            ("07_condition_composition.png", "Clinical Condition Purity"),
            ("09_purity_heatmap.png", "Condition Specialization Heatmap"),
            ("10_age_violin_distribution.png", "Age Demographics per Cluster"),
            ("11_city_cluster_heatmap.png", "Geographic Concentration"),
            ("12_age_condition_bubbles.png", "Age vs Condition Correlation")
        ]

        # Create a 2-column layout for a "Dashboard" feel
        rows = [viz_files[i:i + 2] for i in range(0, len(viz_files), 2)]
        
        for row in rows:
            cols = st.columns(2)
            for i, (file_name, title) in enumerate(row):
                full_path = plot_path / file_name
                if full_path.exists():
                    with cols[i]:
                        st.markdown(f"**{title}**")
                        st.image(str(full_path), use_container_width=True)
                else:
                    with cols[i]:
                        st.info(f"Plot {file_name} is not yet generated.")