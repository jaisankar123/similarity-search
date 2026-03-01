import os
import json
import faiss
import numpy as np
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv

# ======================================================
# CONFIGURATION & ENV LOADING
# ======================================================
# Load .env from the root project directory
# BASE_DIR = Path(r"D:\capstone project 2")
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# MongoDB Settings
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
EMB_COLL = os.getenv("EMBEDDINGS_COLLECTION")

# Output path for FAISS
OUTPUT_FAISS_DIR = Path(os.getenv("FAISS_INDEX_PATH", BASE_DIR / "faiss"))
INDEX_NAME = "patient.index"
MAPPING_NAME = "index_mapping.json"

# ======================================================
# MAIN
# ======================================================

def main():
    # 1. INITIALIZE MONGODB
    if not MONGO_URI:
        print("❌ Error: MONGO_URI not found in .env")
        return

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[EMB_COLL]

    # 2. FOLDER SETUP
    OUTPUT_FAISS_DIR.mkdir(parents=True, exist_ok=True)
    faiss_file = OUTPUT_FAISS_DIR / INDEX_NAME
    mapping_file = OUTPUT_FAISS_DIR / MAPPING_NAME

    # 3. DATA LOADING FROM MONGODB
    print(f"🔎 Fetching embeddings from MongoDB collection: {EMB_COLL}...")
    
    # Get all documents with p_id and embedding
    cursor = collection.find({}, {"p_id": 1, "embedding": 1, "_id": 0})
    
    embeddings = []
    index_mapping = {}
    idx = 0

    for data in cursor:
        vector = data.get("embedding")
        p_id = data.get("p_id")

        if vector is not None and p_id is not None:
            # FAISS requires float32 vectors
            embeddings.append(np.array(vector, dtype="float32"))
            # Map the FAISS integer position back to our Patient ID
            index_mapping[str(idx)] = {"p_id": p_id}
            idx += 1

    if not embeddings:
        print("❌ No valid data found in MongoDB to index.")
        return

    # 4. BUILD FAISS INDEX
    print(f"\n✅ Loaded {len(embeddings)} vectors. Building FAISS index...")
    # Stack list into a 2D matrix
    embeddings_matrix = np.vstack(embeddings).astype("float32")
    
    # Normalize for Cosine Similarity
    faiss.normalize_L2(embeddings_matrix)
    
    dimension = embeddings_matrix.shape[1]
    
    # IndexFlatIP uses Inner Product (Cosine Similarity after normalization)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_matrix)

    # 5. SAVE FILES
    print(f"💾 Saving Index -> {faiss_file}")
    faiss.write_index(index, str(faiss_file))

    print(f"💾 Saving Mapping -> {mapping_file}")
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(index_mapping, f, indent=2)

    print(f"\n✨ Done! FAISS Index built with {idx} patients.")
    print(f"Stored in: {OUTPUT_FAISS_DIR.absolute()}")

if __name__ == "__main__":
    main()