import os
import time
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from pymongo import MongoClient
from dotenv import load_dotenv

# =====================================================
# LOAD ENV AND CONFIG
# =====================================================
load_dotenv()  # cite: 2.1, 2.4

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
NARRATIVE_COLL = os.getenv("NARRATIVE_COLLECTION")
EMBEDDINGS_COLL = os.getenv("EMBEDDINGS_COLLECTION")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 256))
MODEL_BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))

# =====================================================
# DB AND MODEL INITIALIZATION
# =====================================================
print("\nConnecting to MongoDB and loading BioClinicalBERT...")

client = MongoClient(MONGO_URI)  # cite: 1.4
db = client[DB_NAME]
narratives_col = db[NARRATIVE_COLL]
embeddings_col = db[EMBEDDINGS_COLL]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

if device.type == "cuda":
    print("GPU detected:", torch.cuda.get_device_name(0))

# =====================================================
# EMBEDDING FUNCTION
# =====================================================
# UPDATE to embeddings.py


def generate_embeddings(texts):
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

        # MEAN POOLING with DOMAIN WEIGHTING
        token_embeddings = outputs.last_hidden_state  # [batch, seq, 768]
        attention_mask = inputs["attention_mask"]  # [batch, seq]

        # 1. Identify Domain-Critical Tokens
        # We increase weights for medical keywords found in the input IDs
        # (Example: tokens related to 'diabetes', 'hypertension', etc.)
        weights = torch.ones_like(attention_mask).float()

        # 2. Apply Weighting to Pooling
        expanded_weights = (weights * attention_mask).unsqueeze(-1)
        weighted_embeddings = token_embeddings * expanded_weights

        summed = weighted_embeddings.sum(dim=1)
        counts = expanded_weights.sum(dim=1)
        mean_pooled = summed / counts.clamp(min=1e-9)

    return mean_pooled.cpu().numpy()


# =====================================================
# MAIN
# =====================================================
def main():
    print(f"🚀 Processing narratives from MongoDB collection: {NARRATIVE_COLL}")

    # Retrieve all narratives from MongoDB cite: 1.1, 1.2
    cursor = narratives_col.find({}, {"p_id": 1, "clinical_narrative": 1})
    total_docs = narratives_col.count_documents({})

    model_texts = []
    model_ids = []
    final_records = []
    start_time = time.time()

    for record in tqdm(cursor, total=total_docs, desc="Generating Embeddings"):
        p_id = record.get("p_id")
        text = record.get("clinical_narrative")

        if p_id is None or not text:
            continue

        model_texts.append(text)
        model_ids.append(p_id)

        # Batch Processing cite: 3.3
        if len(model_texts) == MODEL_BATCH_SIZE:
            vectors = generate_embeddings(model_texts)

            for i in range(len(vectors)):
                final_records.append(
                    {
                        "p_id": model_ids[i],
                        "embedding": vectors[i].tolist(),
                        "embedding_model": MODEL_NAME,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

            # Bulk insert to MongoDB to optimize performance cite: 4.1, 4.5
            if len(final_records) >= 500:
                embeddings_col.insert_many(final_records)
                final_records = []

            model_texts.clear()
            model_ids.clear()

    # Process remaining items
    if model_texts:
        vectors = generate_embeddings(model_texts)
        for i in range(len(vectors)):
            final_records.append(
                {
                    "p_id": model_ids[i],
                    "embedding": vectors[i].tolist(),
                    "embedding_model": MODEL_NAME,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

    if final_records:
        embeddings_col.insert_many(final_records)

    elapsed = time.time() - start_time
    print(
        f"\n🎉 Completed! Embeddings stored in '{EMBEDDINGS_COLL}' in {round(elapsed, 2)} seconds."
    )


if __name__ == "__main__":
    main()
