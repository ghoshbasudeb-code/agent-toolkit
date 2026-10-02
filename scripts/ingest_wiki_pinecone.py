import os
import torch
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = "dl-ai"

def main():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("Set PINECONE_API_KEY environment variable.")

    pc = Pinecone(api_key=api_key)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # Recreate index
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME in existing_indexes:
        pc.delete_index(INDEX_NAME)

    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    index = pc.Index(INDEX_NAME)

    # Load dataset
    df = pd.read_csv("wiki.csv")
    batch_size = 200

    print("Ingesting Wikipedia articles to Pinecone...")
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i : i + batch_size]
        texts = batch["text"].astype(str).tolist()
        embeddings = model.encode(texts).tolist()

        prepped = []
        for idx, row, vector in zip(batch.index, batch.iterrows(), embeddings):
            row_data = row[1]
            prepped.append(
                {
                    "id": str(row_data["id"]),
                    "values": vector,
                    "metadata": {
                        "text": str(row_data["text"])[:1000],
                        "title": str(row_data["title"]),
                        "url": str(row_data["url"]),
                    },
                }
            )
        index.upsert(vectors=prepped)

    print("Ingestion complete!")
    print(index.describe_index_stats())

if __name__ == "__main__":
    main()
