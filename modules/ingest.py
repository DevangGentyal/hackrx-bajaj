import requests
import time
import os
from uuid import uuid4
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from concurrent.futures import ThreadPoolExecutor
import itertools
import tempfile
import hashlib
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ------------------- CONFIG -------------------
load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENVIRONMENT")
index_name = os.getenv("PINECONE_INDEX_NAME")
index_host = os.getenv("PINECONE_INDEX_HOST")

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Load free forever embedding model (fast + accurate)
# All-MiniLM-L6-v2 is fast (~384 dim), good semantic accuracy
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------- PDF EXTRACTION -------------------
def extract_complete_text_from_pdf(url):
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_path = temp_file.name

        reader = PdfReader(temp_path)
        os.unlink(temp_path)

        return "\n".join(f"--- Page {i+1} ---\n{page.extract_text() or ''}" for i, page in enumerate(reader.pages))

    except Exception as e:
        raise Exception(f"PDF extraction failed: {e}")

# ------------------- TEXT CHUNKING -------------------
def intelligent_text_chunking(text, max_tokens=512, overlap_tokens=50):
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4

    sentences, chunk, chunks = [], "", []
    for char in text:
        chunk += char
        if char in ".!?" and len(chunk) > 20:
            sentences.append(chunk.strip())
            chunk = ""
    if chunk.strip():
        sentences.append(chunk.strip())

    current_chunk, overlap_buffer = "", ""
    for sentence in sentences:
        test_chunk = (current_chunk + " " + sentence).strip()
        if len(test_chunk) <= max_chars:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                words = current_chunk.split()
                overlap_words, overlap_len = [], 0
                for word in reversed(words):
                    if overlap_len + len(word) + 1 <= overlap_chars:
                        overlap_words.insert(0, word)
                        overlap_len += len(word) + 1
                    else:
                        break
                overlap_buffer = " ".join(overlap_words)
            current_chunk = (overlap_buffer + " " + sentence).strip()

    if current_chunk:
        chunks.append(current_chunk.strip())

    return list(dict.fromkeys(chunks))

# ------------------- HELPERS -------------------
def create_unique_namespace():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    unique_id = str(uuid4())[:8]
    process_id = os.getpid()
    combined = f"{timestamp}_{unique_id}_{process_id}"
    hash_suffix = hashlib.md5(combined.encode()).hexdigest()[:8]
    return f"doc_{timestamp}_{hash_suffix}"

def batch_efficiently(iterable, batch_size=100):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            break
        yield batch

# ------------------- EMBEDDING + UPSERT -------------------
def parallel_upsert_complete(chunks, namespace):
    print(f"📤 Uploading {len(chunks)} chunks to namespace: {namespace}")

    if index_name not in pc.list_indexes().names():
        print("🔨 Creating Pinecone index...")
        pc.create_index(
            name=index_name,
            dimension=384,  # must match embedding model output
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=env)
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(host=index_host)

    # Embed locally in parallel
    print("🔍 Generating embeddings...")
    embeddings = embed_model.encode(chunks, batch_size=32, show_progress_bar=True)

    # Prepare Pinecone records
    records = [
        {
            "id": f"{namespace}_chunk_{i:06d}",
            "values": emb.tolist(),
            "metadata": {"text": chunk}
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    uploaded_count = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(index.upsert, vectors=batch, namespace=namespace)
                   for batch in batch_efficiently(records, 50)]
        for f in futures:
            try:
                f.result(timeout=30)
                uploaded_count += 1
            except Exception as e:
                print(f"⚠️ Upload error: {e}")

    print(f"✅ Uploaded {len(records)} vectors to Pinecone")
    return uploaded_count > 0

# ------------------- MAIN PIPELINE -------------------
def process_and_ingest(pdf_url):
    total_start = time.time()
    namespace = create_unique_namespace()

    text = extract_complete_text_from_pdf(pdf_url)
    chunks = intelligent_text_chunking(text)

    upload_success = parallel_upsert_complete(chunks, namespace)
    total_time = time.time() - total_start

    return {
        "namespace": namespace,
        "total_chunks": len(chunks),
        "total_characters": len(text),
        "processing_time": total_time,
        "success": upload_success
    }

# Example usage:
# result = process_and_ingest_complete("https://example.com/sample.pdf")
# print(result)
