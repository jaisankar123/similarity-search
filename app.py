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
st.markdown("""
<style>

/* ============================================================
   APP BACKGROUND
============================================================ */

.stApp {
    background: linear-gradient(
        180deg,
        #FFD8A8 0%,
        #FFE8CC 40%,
        #FFF4E6 75%,
        #FFFFFF 100%
    );
}

.main{
    background: transparent;
}

/* ============================================================
   TOP HEADER
============================================================ */

header[data-testid="stHeader"]{
    background: linear-gradient(
        90deg,
        #FFB366,
        #FFE8CC
    ) !important;
}

[data-testid="stToolbar"]{
    background: transparent !important;
}

/* ============================================================
   SIDEBAR
============================================================ */

[data-testid="stSidebar"]{
    background: linear-gradient(
        180deg,
        #FFE8CC,
        #FFF7F0
    );
}

button[kind="header"] svg{
    fill:#000000 !important;
    color:#000000 !important;
}

/* ============================================================
   TEXT
============================================================ */

h1, h2, h3, h4, h5, h6,
label,
p,
span,
div{
    color:#222222 !important;
}

/* ============================================================
   TEXT INPUT
============================================================ */

.stTextInput input{
    background-color:white !important;
    color:black !important;
    border:1px solid #cccccc !important;
    border-radius:8px;
}

.stTextInput input::placeholder{
    color:#666666 !important;
}

/* ============================================================
   BUTTONS
============================================================ */

.stButton > button{
    background:#FF8C42 !important;
    color:white !important;
    border:none;
    border-radius:8px;
}

.stButton > button:hover{
    background:#F97316 !important;
}

.stDownloadButton > button{
    background:white !important;
    color:black !important;
    border:1px solid #cccccc !important;
    border-radius:8px;
}

.stDownloadButton > button:hover{
    background:#FFE8CC !important;
}


/* ============================================================
   PATIENT CARD
============================================================ */

.patient-card{
    background:white !important;
    color:black !important;
    border-radius:12px;
    padding:20px;
    border:1px solid #dddddd;
    box-shadow:0 3px 8px rgba(0,0,0,0.1);
}

.patient-card strong,
.patient-card p{
    color:black !important;
}

/* ============================================================
   TABLE (UPDATED WITH EXPLICIT WIDTH FIXES)
============================================================ */

div[data-testid="stTable"] {
    background: white !important;
    border-radius: 10px;
    width: 100% !important;
    overflow-x: auto !important;
}

div[data-testid="stTable"] table {
    background: white !important;
    color: black !important;
    border-collapse: collapse !important;
    width: 100% !important;
    table-layout: fixed !important; /* Forces strict adherence to cell widths */
}

div[data-testid="stTable"] th {
    background: #FFE8CC !important;
    color: black !important;
    border: 1px solid #dddddd !important;
    padding: 10px !important;
}

div[data-testid="stTable"] td {
    background: white !important;
    color: black !important;
    border: 1px solid #dddddd !important;
    padding: 10px !important;
    word-wrap: break-word !important;
    white-space: normal !important;
}

/* Precise column-by-column widths */
/* Index column */
div[data-testid="stTable"] th:nth-child(1), 
div[data-testid="stTable"] td:nth-child(1) {
    width: 50px !important;
}

/* Patient ID Column */
div[data-testid="stTable"] th:nth-child(2), 
div[data-testid="stTable"] td:nth-child(2) {
    width: 80px !important;
}

/* Similarity Score Column */
div[data-testid="stTable"] th:nth-child(3), 
div[data-testid="stTable"] td:nth-child(3) {
    width: 90px !important;
}

/* Clinical Narrative Column (gets the remaining wide width) */
div[data-testid="stTable"] th:nth-child(4), 
div[data-testid="stTable"] td:nth-child(4) {
    width: auto !important;
}


/* ============================================================
   DATAFRAME
============================================================ */

[data-testid="stDataFrame"]{
    background:white !important;
    color:black !important;
}

[data-testid="stDataFrame"] table{
    color:black !important;
}

/* ============================================================
   ALERTS
============================================================ */

.stAlert{
    background:white !important;
    color:black !important;
}

/* ============================================================
   REMOVE BLACK BACKGROUND
============================================================ */

.block-container{
    background:transparent !important;
}

</style>
""", unsafe_allow_html=True)

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# ======================================================
# 2. DATA & ASSET LOADERS
# # ======================================================
# def load_lottie_file(filepath: str):
#     try:
#         with open(filepath, "r") as f:
#             return json.load(f)
#     except:
#         return None

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
success_image = "Sucesso.svg"
search_svg = "search.svg"

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
    st.image(search_svg, width=220)
    
    st.title("Settings")
    top_k = st.slider("Number of neighbors", 1, 10, 5)

st.title("🧬 FHIR Patient Similarity Finder")
st.write("Search clinically similar patients using Bio-ClinicalBERT embeddings.")

# Search bar logic (Half screen width)
col_search_1, col_search_2, col_spacer = st.columns([2, 1, 2]) 

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
                # st.markdown('<div class="loading-container">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])

                with col2:
                    st.image(search_svg, width=180)
                st.markdown('<h3>Analyzing Clinical Patterns...</h3></div>', unsafe_allow_html=True)
                
            # Perform Search
            target, neighbors = get_patient_similarity(query_id, k=top_k + 1)
            time.sleep(1.5) # Ensuring the animation is visible to the user
            
            # Clear loader
            placeholder.empty()

            if target:
                # 2. Success Animation
                col1, col2, col3 = st.columns([1, 2, 1])

                with col2:
                    st.image(success_image, width=180)
                
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