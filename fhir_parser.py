import json
import os
import uuid
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support

# ======================================================
# CONFIGURATION
# ======================================================
INPUT_DIR = r"D:\capstone project 2\fhir"  
OUTPUT_DIR = r"D:\capstone project 2\parsed_data"
MAX_WORKERS = min(4, os.cpu_count())

TOTAL_RECORDS_LIMIT = 20000
BATCH_SIZE = 10000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# ROBUST EXTRACTION HELPERS
# ======================================================

def normalize_uuid(raw_id):
    if not raw_id: return None
    return str(raw_id).replace("urn:uuid:", "").replace("Patient/", "").strip()

def format_date_only(date_str):
    if not date_str or not isinstance(date_str, str) or date_str == "Unknown Date":
        return "Unknown Date"
    return date_str.split('T')[0]

def get_resource_date(res):
    date_fields = [
        "effectiveDateTime", "onsetDateTime", "authoredOn", 
        "occurrenceDateTime", "period", "issued", "recordedDate"
    ]
    for field in date_fields:
        val = res.get(field)
        if val:
            if isinstance(val, dict): 
                return str(val.get("start") or val.get("end"))
            return str(val)
    return None

def get_any_text(data):
    if not data: return None
    if isinstance(data, str): return data
    if isinstance(data, dict):
        for key in ["text", "display"]:
            val = data.get(key)
            if val and isinstance(val, str): return val
        if "coding" in data and isinstance(data["coding"], list):
            for code in data["coding"]:
                res = get_any_text(code)
                if res: return res
    return None

def extract_obs_value(res):
    if "valueQuantity" in res:
        v = res["valueQuantity"].get("value", "N/A")
        u = res["valueQuantity"].get("unit", "")
        return f"{v} {u}".strip()
    if "valueString" in res:
        return res["valueString"]
    if "valueCodeableConcept" in res:
        return get_any_text(res["valueCodeableConcept"])
    if "component" in res:
        parts = []
        for comp in res["component"]:
            c_name = get_any_text(comp.get("code"))
            c_val = extract_obs_value(comp)
            parts.append(f"{c_name}: {c_val}")
        return " | ".join(parts)
    return "N/A"

# ======================================================
# CORE PARSER
# ======================================================

def process_fhir_bundle(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            bundle = json.load(f)
        
        patient_data = {}
        target_uuid = None
        history = {
            "conditions": [], "medications": [], "vitals": [],
            "lab_results": [], "diagnostic_reports": [], 
            "encounters": [], "other_procedures": []
        }

        entries = bundle.get("entry", [])

        # Pass 1: Setup Patient
        for entry in entries:
            res = entry.get("resource", {})
            if res.get("resourceType") == "Patient":
                p_id = normalize_uuid(res.get("id"))
                name_list = res.get("name", [{}])
                first = " ".join(name_list[0].get("given", []))
                last = name_list[0].get("family", "")
                
                patient_data = {
                    "patient_id": p_id,
                    "name": f"{first} {last}".strip(),
                    "gender": res.get("gender", "Unknown"),
                    "birth_date": format_date_only(res.get("birthDate")),
                    "city": res.get("address", [{}])[0].get("city", "Unknown") if res.get("address") else "Unknown"
                }
                target_uuid = p_id
                break

        if not patient_data: return None

        # Pass 2: Extract Clinical Data
        for entry in entries:
            res = entry.get("resource", {})
            r_type = res.get("resourceType")
            
            ref = normalize_uuid(res.get("subject", {}).get("reference") or res.get("patient", {}).get("reference"))
            if ref and ref != target_uuid:
                continue

            raw_date = get_resource_date(res)
            clean_date = format_date_only(raw_date)

            if r_type == "Condition":
                history["conditions"].append({"item": get_any_text(res.get("code")), "date": clean_date, "raw": raw_date})

            elif r_type in ["MedicationRequest", "Immunization"]:
                item = get_any_text(res.get("medicationCodeableConcept") or res.get("vaccineCode") or res.get("medicationReference"))
                history["medications"].append({"item": item, "date": clean_date, "raw": raw_date})

            elif r_type == "Observation":
                name = get_any_text(res.get("code"))
                if not name: continue
                val = extract_obs_value(res)
                entry_data = {"item": f"{name}: {val}", "date": clean_date, "raw": raw_date}
                
                vitals_set = {"height", "weight", "bmi", "blood pressure", "heart rate", "respiratory", "oxygen"}
                if any(v in name.lower() for v in vitals_set):
                    history["vitals"].append(entry_data)
                else:
                    history["lab_results"].append(entry_data)

            elif r_type == "DiagnosticReport":
                name = get_any_text(res.get("code"))
                # Diagnostic reports are now explicitly linked to this new column
                history["diagnostic_reports"].append({"item": name, "date": clean_date, "raw": raw_date})

            elif r_type == "Encounter":
                history["encounters"].append({"item": get_any_text(res.get("type")), "date": clean_date, "raw": raw_date})

            elif r_type == "Procedure":
                history["other_procedures"].append({"item": get_any_text(res.get("code")), "date": clean_date, "raw": raw_date})

        # Final Deduplication & Sorting
        for key in history:
            history[key].sort(key=lambda x: x['raw'] if x['raw'] else "")
            
            unique = []
            seen = set()
            for d in history[key]:
                item_label = d["item"]
                date_label = d["date"]
                if item_label and f"{item_label}_{date_label}" not in seen:
                    unique.append({"item": item_label, "date": date_label})
                    seen.add(f"{item_label}_{date_label}")
            history[key] = unique
        
        patient_data.update(history)
        return patient_data
            
    except Exception:
        return None

def main():
    all_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    json_files = all_files[:TOTAL_RECORDS_LIMIT]
    print(f"🚀 Parsing {len(json_files)} files...")

    for i in range(0, len(json_files), BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1
        batch_files = json_files[i : i + BATCH_SIZE]
        output_file = os.path.join(OUTPUT_DIR, f"parsed_data_part_{batch_num}.jsonl")

        with open(output_file, 'w', encoding='utf-8') as out_f:
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                results = list(tqdm(executor.map(process_fhir_bundle, batch_files), total=len(batch_files)))
                for r in results:
                    if r: out_f.write(json.dumps(r) + "\n")
        print(f"✅ Part {batch_num} saved.")

if __name__ == "__main__":
    freeze_support()
    main()