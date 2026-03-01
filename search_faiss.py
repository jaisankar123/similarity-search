import os
import json
import faiss
import numpy as np
from pymongo import MongoClient
from tabulate import tabulate
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

# ======================================================
# CONFIGURATION & ENV LOADING
# ======================================================
# BASE_DIR = Path(r"D:\capstone project 2")
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Local File Paths
FAISS_DIR = Path(os.getenv("FAISS_INDEX_PATH", BASE_DIR / "faiss"))
INDEX_FILE = str(FAISS_DIR / "patient.index")
MAPPING_FILE = str(FAISS_DIR / "index_mapping.json")

# MongoDB Settings
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("NARRATIVE_COLLECTION") # clinical_narratives

def load_resources():
    """Loads the FAISS index and the ID mapping file."""
    print("📂 Loading FAISS index...")
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(f"Index file not found at {INDEX_FILE}. Run build_index.py first.")
    index = faiss.read_index(INDEX_FILE)

    print("📂 Loading index mapping...")
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    
    # Map stored position ("0") to p_id
    # mapping is: {"0": {"p_id": 1}, "1": {"p_id": 2}, ...}
    p_id_to_idx = {str(v['p_id']): int(k) for k, v in mapping.items()}
    idx_to_p_id = {int(k): str(v['p_id']) for k, v in mapping.items()}

    return index, p_id_to_idx, idx_to_p_id

def get_mongo_connection():
    """Connects to MongoDB to retrieve clinical narratives."""
    if not MONGO_URI:
        raise ValueError("MONGO_URI not found in .env file.")
    client = MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION_NAME]

def fetch_clinical_narrative(collection, p_id):
    """Retrieves the narrative for a specific p_id from MongoDB."""
    # Note: p_id in Mongo is an integer
    doc = collection.find_one({"p_id": int(p_id)})
    if doc and "clinical_narrative" in doc:
        return doc["clinical_narrative"]
    return "Narrative not found in MongoDB."

def main():
    try:
        index, p_id_to_idx, idx_to_p_id = load_resources()
        collection = get_mongo_connection()
        print(f"✅ System Ready. Total records in index: {index.ntotal}")
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        return

    while True:
        print("\n" + "="*60)
        query_id = input("🔍 Enter Patient p_id to find matches (or 'exit'): ").strip()

        if query_id.lower() == 'exit':
            break

        if query_id not in p_id_to_idx:
            print(f"⚠️ Patient p_id '{query_id}' not found in the local index.")
            continue

        # 1. Reconstruct the vector for the target patient from the index
        internal_idx = p_id_to_idx[query_id]
        query_vector = index.reconstruct(internal_idx).reshape(1, -1)

        # 2. Search for the top 6 matches (Self + 5 neighbors)
        k = 6
        print(f"🔎 Searching for top {k-1} similar patients...")
        scores, ids = index.search(query_vector, k)

        # 3. Process Results
        results = []
        target_info = []

        for i in range(len(ids[0])):
            hit_idx = ids[0][i]
            score = scores[0][i]

            if hit_idx == -1: continue
            
            hit_p_id = idx_to_p_id.get(int(hit_idx))
            narrative = fetch_clinical_narrative(collection, hit_p_id)
            
            # Shorten text for tabular display
            display_text = (narrative[:120] + "...") if len(narrative) > 120 else narrative

            if hit_p_id == query_id:
                target_info.append([hit_p_id, narrative])
            else:
                results.append([hit_p_id, round(float(score), 4), display_text])

        # 4. Display Results
        print("\n🎯 SELECTED PATIENT")
        print(tabulate(target_info, headers=["p_id", "Full Clinical Narrative"], tablefmt="grid"))

        print("\n👥 TOP SIMILAR PATIENTS FOUND")
        if results:
            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)
            print(tabulate(results, headers=["p_id", "Similarity Score", "Narrative Preview"], tablefmt="fancy_grid"))
        else:
            print("No similar patients found.")

if __name__ == "__main__":
    main()