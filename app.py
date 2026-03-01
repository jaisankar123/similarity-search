import streamlit as st
import pandas as pd
import faiss
import json
import numpy as np
import os
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv
from streamlit_lottie import st_lottie

# ======================================================
# 1. CONFIGURATION & STYLING
# ======================================================
st.set_page_config(page_title="FHIR Patient Similarity", layout="wide")

# Aesthetic light color theme and styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        background-color: #6c63ff; color: white; border-radius: 8px; 
        border: none; padding: 10px 24px;
    }
    .stTextInput>div>div>input { border-radius: 8px; }
    h1 { color: #2c3e50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .patient-card { 
        background-color: white; padding: 20px; border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load environment variables
# BASE_DIR = Path(r"D:\capstone project 2")
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# ======================================================
# 2. DATA & ASSET LOADERS
# ======================================================
def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

@st.cache_resource
def init_resources():
    # Load FAISS
    faiss_dir = Path(os.getenv("FAISS_INDEX_PATH", BASE_DIR / "faiss"))
    index = faiss.read_index(str(faiss_dir / "patient.index"))
    
    with open(str(faiss_dir / "index_mapping.json"), "r") as f:
        mapping = json.load(f)
    
    # Connect MongoDB
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("DB_NAME")]
    collection = db[os.getenv("NARRATIVE_COLLECTION")]
    
    return index, mapping, collection

# Load assets
dna_helix = load_lottie_file("DNA Helix.json")
success_anim = load_lottie_file("success.json")
index, mapping, collection = init_resources()
idx_to_p_id = {int(k): v["p_id"] for k, v in mapping.items()}

# ======================================================
# 3. SEARCH LOGIC
# ======================================================
def get_patient_similarity(query_id, k=6):
    # Find the vector for the target patient
    target_data = collection.find_one({"p_id": query_id})
    if not target_data:
        return None, None

    # Fetch embedding from the embedding collection
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
# Sidebar Background Animation
with st.sidebar:
    st_lottie(dna_helix, speed=1, reverse=False, loop=True, quality="low", height=200)
    st.title("Settings")
    top_k = st.slider("Number of neighbors", 1, 10, 5)

st.title("🧬 FHIR Patient Similarity Finder")
st.write("Enter a Patient ID to find clinically similar patients using Bio-ClinicalBERT embeddings.")

query_id_input = st.text_input("Enter Patient ID (e.g., 101):", "")

if st.button("Search Similar Patients"):
    if query_id_input:
        try:
            query_id = int(query_id_input)
            target, neighbors = get_patient_similarity(query_id, k=top_k + 1)
            
            if target:
                # Success Animation
                st_lottie(success_anim, speed=1, loop=False, height=150)
                
                st.subheader("🎯 Selected Patient")
                st.markdown(f"""
                <div class="patient-card">
                    <strong>ID:</strong> {target['Patient ID']}<br>
                    <strong>Clinical Narrative:</strong> {target['Clinical Narrative']}
                </div>
                """, unsafe_allow_html=True)

                st.subheader(f"👥 Top {top_k} Similar Patients")
                df_results = pd.DataFrame(neighbors)
                st.table(df_results)
                
            else:
                st.error("Patient ID not found in database.")
        except ValueError:
            st.error("Please enter a valid numerical Patient ID.")
    else:
        st.warning("Please enter a Patient ID to proceed.")