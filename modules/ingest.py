import os
import requests
import time
from uuid import uuid4
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENVIRONMENT")
index_name = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=api_key)

def extract_text_from_pdf_fast(url):
    """Ultra-fast PDF extraction with minimal processing"""
    try:
        response = requests.get(url, timeout=8, stream=True)  # Reduced timeout + streaming
        response.raise_for_status()
        
        # Write and read simultaneously for speed
        with open("temp.pdf", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        reader = PdfReader("temp.pdf")
        
        # Extract only first N pages for speed (adjust based on needs)
        max_pages = min(50, len(reader.pages))  # Limit to 50 pages max
        text = "\n".join(
            p.extract_text()[:2000]  # Limit text per page
            for p in reader.pages[:max_pages] 
            if p.extract_text()
        )
        
        os.remove("temp.pdf")
        return text
        
    except Exception as e:
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
        raise Exception(f"PDF extraction failed: {str(e)[:100]}")

def split_text_fast(text, max_tokens=200):  # Much smaller chunks for speed
    """Ultra-fast text splitting"""
    
    # Simple word-based splitting (much faster than tiktoken)
    words = text.split()
    chunks = []
    chunk = []
    word_count = 0
    
    # Rough estimation: 1 token ≈ 0.75 words
    max_words = int(max_tokens * 0.75)
    
    for word in words:
        chunk.append(word)
        word_count += 1
        
        if word_count >= max_words:
            chunks.append(" ".join(chunk))
            chunk = []
            word_count = 0
    
    if chunk:
        chunks.append(" ".join(chunk))
    
    return chunks

def batch_fast(iterable, n=96):
    """Ultra-fast batching"""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            break
        yield chunk

def lightning_upsert(chunks, namespace):
    """Lightning-fast parallel upload with maximum concurrency"""
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=2048,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=env),
        )

    index = pc.Index(index_name)
    
    # Pre-generate all docs (faster than generating during upload)
    docs = [{"_id": str(uuid4()), "text": chunk} for chunk in chunks]
    print(f"⚡ Lightning upload: {len(docs)} chunks")

    # Maximum parallelism with larger thread pool
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [
            executor.submit(turbo_upsert, index, list(doc_batch), namespace)
            for doc_batch in batch_fast(docs, 96)
        ]
        
        # Don't wait for individual completion - just ensure all complete
        completed = sum(1 for f in as_completed(futures) if f.result(timeout=30))
    
    print(f"⚡ Upload: {completed} batches completed")
    if completed == 0:
        raise Exception("All uploads failed")

def turbo_upsert(index, doc_batch, namespace):
    """Single-attempt fast upsert"""
    try:
        index.upsert_records(records=doc_batch, namespace=namespace)
        return True
    except:
        return False  # Fail fast - no retries

def process_and_ingest_lightning(pdf_url):
    """Lightning-fast ingestion pipeline"""
    print(f"⚡ Lightning ingestion starting...")
    
    namespace = str(uuid4())
    total_start = time.time()
    
    try:
        # Step 1: Extract (with limits)
        text = extract_text_from_pdf_fast(pdf_url)
        extract_time = time.time() - total_start
        print(f"⚡ PDF extracted in {extract_time:.1f}s")
        
        # Step 2: Split (faster method)
        split_start = time.time()
        chunks = split_text_fast(text, max_tokens=200)  # Smaller chunks
        split_time = time.time() - split_start
        print(f"⚡ Split {len(chunks)} chunks in {split_time:.1f}s")
        
        # Step 3: Upload (maximum speed)
        upload_start = time.time()
        lightning_upsert(chunks, namespace)
        upload_time = time.time() - upload_start
        print(f"⚡ Upload in {upload_time:.1f}s")
        
        total_time = time.time() - total_start
        print(f"⚡ Total ingestion: {total_time:.1f}s")
        
        return namespace
        
    except Exception as e:
        print(f"❌ Lightning ingestion failed: {e}")
        raise

# Replace the function call in your main module
process_and_ingest = process_and_ingest_lightning
