import os
import requests
from uuid import uuid4
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from typing import List
from pinecone import Pinecone, ServerlessSpec
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Batch helper
def batch(iterable, n=96):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            break
        yield chunk

# Step 1: Download and extract text from PDF
def extract_text_from_pdf(url: str) -> str:
    response = requests.get(url)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)

    reader = PdfReader("temp.pdf")
    text = "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
    os.remove("temp.pdf")
    return text

# Step 2: Token-safe chunking
def split_text(text: str, max_tokens=400) -> List[str]:
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

# Step 3: Upsert to Pinecone (NO embeddings — just raw text with metadata)
def upsert_to_pinecone(chunks: List[str], namespace: str):
    # Create index if not exists
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,  # If using OpenAI/Pinecone-hosted embedding models
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
        )

    index = pc.Index(PINECONE_INDEX_NAME)

    # Prepare vector records without values (Pinecone will embed server-side)
    vectors = [{
        "id": str(uuid4()),
        "metadata": {"text": chunk}
    } for chunk in chunks]

    # Upsert in batches
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for vector_batch in batch(vectors, 96):
            futures.append(executor.submit(
                index.upsert,
                vectors=vector_batch,
                namespace=namespace
            ))
        for future in as_completed(futures):
            future.result()

# Main function to process and ingest
def process_and_ingest(pdf_url: str) -> str:
    namespace = str(uuid4())[:8]  # Generate unique namespace per request

    # Step 1: Parse and chunk
    text = extract_text_from_pdf(pdf_url)
    chunks = split_text(text)

    # Step 2: Store in Pinecone
    upsert_to_pinecone(chunks, namespace)

    return namespace

# Local test
if __name__ == "__main__":
    url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    ns = process_and_ingest(url)
    print(f"✅ Data ingested under namespace: {ns}")
