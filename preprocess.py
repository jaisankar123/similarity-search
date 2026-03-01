import json
import os
import glob
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from pymongo import MongoClient
from dotenv import load_dotenv

# ======================================================
# 1. LOAD CONFIGURATION (FIXED PATH)
# ======================================================
# BASE_PROJECT_PATH = r"D:\capstone project 2"
from pathlib import Path
BASE_PROJECT_PATH = Path(__file__).resolve().parent
env_path = os.path.join(BASE_PROJECT_PATH, ".env")
load_dotenv(dotenv_path=env_path)

# Verification check
if not os.getenv("DB_NAME"):
    print(f"❌ ERROR: Could not find .env file at {env_path} or variables are missing.")
    exit(1)

# ======================================================
# 2. PATHS & FIELDS
# ======================================================
BASE_DIR = r"D:\capstone project 2\parsed_data"
INPUT_FILES = glob.glob(os.path.join(BASE_DIR, "*.jsonl"))

CLINICAL_FIELDS = [
    "conditions", "medications", "vitals",
    "lab_results", "diagnostic_reports",
    "encounters", "other_procedures"
]

# ======================================================
# 3. PROCESSING LOGIC
# ======================================================

def calculate_age(birthdate_str):
    if not birthdate_str or birthdate_str == "not recorded":
        return "not recorded"
    try:
        birth = datetime.strptime(str(birthdate_str), "%Y-%m-%d")
        today = datetime.today()
        # Standard age calculation logic
        return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    except Exception:
        return "not recorded"

def extract_patient(data):
    clinical_data, has_any_clinical_data = {}, False
    
    # Filter for non-empty clinical fields
    for field in CLINICAL_FIELDS:
        val = data.get(field)
        if val not in [None, [], "not recorded", ""]:
            clinical_data[field] = val
            has_any_clinical_data = True
    
    # If the patient has no clinical history, we skip them
    if not has_any_clinical_data:
        return None

    processed = {
        "original_uuid": data.get("patient_id"),
        "name": data.get("name") or "not recorded",
        "gender": data.get("gender") or "not recorded",
        "age": calculate_age(data.get("birth_date")),
        "city": data.get("city") or "not recorded"
    }
    processed.update(clinical_data)
    return processed

def process_line(line):
    """Wrapper for ProcessPoolExecutor to handle JSON parsing per line"""
    try:
        return extract_patient(json.loads(line))
    except Exception:
        return None

# ======================================================
# 4. MAIN EXECUTION
# ======================================================

def main():
    # Initialize MongoDB inside main to ensure safety with multiprocessing
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("DB_NAME")]
    collection = db[os.getenv("PATIENT_COLLECTION")]

    p_id_counter = 1
    
    # Clear collection before starting to avoid duplicates
    print(f"🧹 Clearing collection: {os.getenv('PATIENT_COLLECTION')}...")
    collection.delete_many({})
    
    print(f"🚀 Processing {len(INPUT_FILES)} files...")

    # Worker count optimized for standard CPU threads
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
        for file_path in INPUT_FILES:
            with open(file_path, "r", encoding="utf8") as f:
                lines = f.readlines()
                
                # Parallelize the data extraction
                results = list(tqdm(
                    executor.map(process_line, lines, chunksize=500),
                    total=len(lines),
                    desc=f"📦 {os.path.basename(file_path)}"
                ))
                
                batch_to_insert = []
                for patient_data in results:
                    if patient_data:
                        record = {"p_id": p_id_counter}
                        record.update(patient_data)
                        batch_to_insert.append(record)
                        p_id_counter += 1
                        
                        # Batch insert every 1000 records to reduce network overhead
                        if len(batch_to_insert) >= 1000:
                            collection.insert_many(batch_to_insert)
                            batch_to_insert = []

                # Final cleanup for the remaining records in the file
                if batch_to_insert:
                    collection.insert_many(batch_to_insert)

    print(f"\n✅ Done! Successfully stored {p_id_counter - 1} patients in MongoDB.")

if __name__ == "__main__":
    main()