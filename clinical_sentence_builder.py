import os
from tqdm import tqdm
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# MongoDB Setup
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("DB_NAME")]
patient_col = db[os.getenv("PATIENT_COLLECTION")]
narrative_col = db[os.getenv("NARRATIVE_COLLECTION")]

def get_unique_items(items):
    if not items or items == "not recorded": return []
    return sorted(list({i.get("item").strip() for i in items if isinstance(i, dict) and i.get("item")}))

# UPDATE to clinical_sentence_builder.py

def build_clinical_text(p):
    # 1. Basic Demographics
    age_text = f"{p.get('age')}-year-old" if isinstance(p.get('age'), int) else "Patient"
    narrative = f"{p.get('name')} is a {age_text} {p.get('gender')} from {p.get('city')}. "
    
    # 2. Structural Segmenting (Summarization approach)
    # Focus on "Active" issues vs "Denied" issues
    conds = p.get("conditions", [])
    if conds and conds != "not recorded":
        active_conds = [c.get('item') for c in conds if "denies" not in c.get('item', '').lower()]
        if active_conds:
            narrative += f"Active conditions: {', '.join(active_conds)}. "

    # 3. Temporal Focus: Only include CURRENT medications and LATEST vitals
    meds = get_unique_items(p.get("medications"))
    if meds:
        narrative += f"Current medication regimen: {', '.join(meds)}. "
    
    # Extracting the most recent lab/vital instead of all history (Temporal Analysis)
    vitals = p.get("vitals", [])
    if vitals and isinstance(vitals, list):
        # Assuming the list is chronological, take the last one
        latest_vital = vitals[-1].get('item', '')
        narrative += f"Most recent clinical observation: {latest_vital}. "

    return narrative.strip()

def main():
    narrative_col.delete_many({}) # Clear old narratives
    patients = list(patient_col.find({}))
    
    print(f"🚀 Generating Narratives for {len(patients)} patients...")
    batch = []
    for p in tqdm(patients):
        batch.append({
            "p_id": p.get("p_id"),
            "original_uuid": p.get("original_uuid"),
            "clinical_narrative": build_clinical_text(p)
        })
        
        if len(batch) >= 500:
            narrative_col.insert_many(batch)
            batch = []
            
    if batch: narrative_col.insert_many(batch)
    print(f"✅ All narratives stored in MongoDB collection: {os.getenv('NARRATIVE_COLLECTION')}")

if __name__ == "__main__":
    main()