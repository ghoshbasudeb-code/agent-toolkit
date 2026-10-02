import os
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm

INDEX_NAME = "dl-ai"

def main():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("Set PINECONE_API_KEY before running ingestion.")

    pc = Pinecone(api_key=api_key)

    # 1. Load dataset
    print("Loading dataset...")
    dataset = load_dataset(
        "sentence-transformers/quora-duplicates",
        "pair",
        split="train[130000:180000]",
    )

    questions = list(set(dataset["anchor"]))
    print(f"Loaded {len(questions)} unique questions.")

    # 2. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # 3. Create Pinecone Index if not present
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=model.get_embedding_dimension(),
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(INDEX_NAME)

    # 4. Batch upsert vectors
    batch_size = 200
    vector_limit = 10000
    questions = questions[:vector_limit]

    print("Upserting vectors to Pinecone...")
    for i in tqdm(range(0, len(questions), batch_size)):
        i_end = min(i + batch_size, len(questions))
        batch = questions[i:i_end]
        ids = [str(n) for n in range(i, i_end)]
        embeds = model.encode(batch).tolist()
        meta = [{"text": q} for q in batch]

        to_upsert = list(zip(ids, embeds, meta))
        index.upsert(vectors=to_upsert)

    print("Ingestion complete!")
    print(index.describe_index_stats())

if __name__ == "__main__":
    main()
