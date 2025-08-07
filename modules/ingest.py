import os
import io
import requests
from uuid import uuid4
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec
import itertools

# Load .env variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENVIRONMENT")
index_name = os.getenv("PINECONE_INDEX_NAME")

# Init Pinecone
pc = Pinecone(api_key=api_key)

# Initialize tokenizer once
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

# Download + extract text from PDF using memory (faster than saving to disk)
def extract_text_from_pdf(url):
    response = requests.get(url)
    pdf_stream = io.BytesIO(response.content)
    reader = PdfReader(pdf_stream)
    return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())

# Token-safe chunking
def split_text(text, max_tokens=400):
    words = text.split()
    chunks, chunk, token_count = [], [], 0

    for word in words:
        word_tokens = len(enc.encode(word + " "))
        if token_count + word_tokens > max_tokens:
            chunks.append(" ".join(chunk))
            chunk, token_count = [], 0
        chunk.append(word)
        token_count += word_tokens
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

# Batch helper: yields batches of size n
def batch(iterable, n=96):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            break
        yield chunk

# Check/create index only once (outside thread loop)
def ensure_index_exists():
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=2048,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=env),
        )

# Upsert to Pinecone
def upsert_to_pinecone(chunks, namespace):
    ensure_index_exists()
    index = pc.Index(index_name)

    # Pre-create records with UUIDs
    docs = [{"_id": str(uuid4()), "text": chunk} for chunk in chunks]

    # Use thread pool for upserts
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [
            executor.submit(index.upsert_records, records=doc_batch, namespace=namespace)
            for doc_batch in batch(docs, 96)
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print("⚠️ Upsert error:", e)

# Full pipeline
def process_and_ingest(pdf_url):
    namespace = str(uuid4())
    text = extract_text_from_pdf(pdf_url)
    chunks = split_text(text)
    upsert_to_pinecone(chunks, namespace)
    return namespace

# Entry point
if __name__ == "__main__":
    input_json = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    }
    namespace_used = process_and_ingest(input_json["documents"])
    print("✅ Data inserted in namespace:", namespace_used)
