import os
import requests
from uuid import uuid4
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load .env variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENVIRONMENT")  # e.g. "us-west1"
index_name = os.getenv("PINECONE_INDEX_NAME")

# Init Pinecone
pc = Pinecone(api_key=api_key)

# Download + extract text from PDF
def extract_text_from_pdf(url):
    response = requests.get(url)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)

    reader = PdfReader("temp.pdf")
    text = "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
    os.remove("temp.pdf")
    return text

# Token-safe chunking
def split_text(text, max_tokens=400):
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks, chunk, token_count = [], [], 0

    for word in words:
        token_count += len(enc.encode(word + " "))
        chunk.append(word)
        if token_count >= max_tokens:
            chunks.append(" ".join(chunk))
            chunk, token_count = [], 0
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

import itertools
from pinecone import Pinecone, ServerlessSpec
from concurrent.futures import ThreadPoolExecutor, as_completed

# Batch helper: yields batches of size n
def batch(iterable, n=96):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            break
        yield chunk

# ✅ Upsert to Pinecone with async_req=True using thread pool
def upsert_to_pinecone(chunks):
    # print("Upserting to Pinecone (parallel)...")

    # Create index if it doesn’t exist
    if index_name not in pc.list_indexes().names():
        # print("Creating index...")
        pc.create_index(
            name=index_name,
            dimension=2048,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=env),
        )

    index = pc.Index(index_name)

    # Prepare documents
    docs = [{
        "_id": str(uuid4()),
        "text": chunk
    } for chunk in chunks]

    # ✅ Start parallel upserts (batch size = 96)
    async_results = []
    with ThreadPoolExecutor(max_workers=30) as executor:
        for i, doc_batch in enumerate(batch(docs, 96)):
            async_results.append(executor.submit(
                index.upsert_records, records=doc_batch, namespace="default"
            ))

        # Wait and collect results
        for i, future in enumerate(as_completed(async_results)):
            try:
                future.result()
                # print(f"✅ Batch {i+1} upserted.")
            except Exception as e:
                break

    # print("✅ All batches done.")

# Ingest full PDF to Pinecone
def process_and_ingest(pdf_url):
    text = extract_text_from_pdf(pdf_url)
    chunks = split_text(text)
    upsert_to_pinecone(chunks)

# Entry point
if __name__ == "__main__":
    input_json = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        
    }
    process_and_ingest(input_json["documents"])
