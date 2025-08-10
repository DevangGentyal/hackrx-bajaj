import requests
import time
import os
from uuid import uuid4
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import tempfile
import hashlib
from datetime import datetime

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENVIRONMENT")
index_name = os.getenv("PINECONE_INDEX_NAME")
index_host = os.getenv("PINECONE_INDEX_HOST")

pc = Pinecone(api_key=api_key)

def extract_complete_text_from_pdf(url):
    try:
        print("🔍 Starting complete PDF extraction...")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = temp_file.name
            total_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
                total_size += len(chunk)

        print(f"📁 Downloaded {total_size} bytes")

        reader = PdfReader(temp_path)
        total_pages = len(reader.pages)
        print(f"📄 Processing {total_pages} pages...")

        full_text = ""
        processed_pages = 0

        for page_num, page in enumerate(reader.pages, 1):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    full_text += f"\n--- Page {page_num} ---\n{page_text}\n"
                processed_pages += 1

                if page_num % 10 == 0:
                    print(f"📄 Processed {page_num}/{total_pages} pages")

            except Exception as page_error:
                print(f"⚠️ Warning: Could not extract text from page {page_num}: {page_error}")
                continue

        os.unlink(temp_path)

        print(f"✅ Extracted {len(full_text)} characters from {processed_pages} pages")

        if not full_text.strip():
            raise Exception("No text content found in PDF")

        return full_text

    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise Exception(f"PDF extraction failed: {str(e)}")

def intelligent_text_chunking(text, max_tokens=512, overlap_tokens=50):
    print("🔪 Starting intelligent text chunking...")

    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4

    sentences = []
    current_sentence = ""

    for char in text:
        current_sentence += char
        if char in '.!?' and len(current_sentence) > 20:
            sentences.append(current_sentence.strip())
            current_sentence = ""

    if current_sentence.strip():
        sentences.append(current_sentence.strip())

    print(f"📝 Split into {len(sentences)} sentences")

    chunks = []
    current_chunk = ""
    overlap_buffer = ""

    for sentence in sentences:
        test_chunk = current_chunk + " " + sentence if current_chunk else sentence

        if len(test_chunk) <= max_chars:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

                words = current_chunk.split()
                overlap_words = []
                overlap_length = 0

                for word in reversed(words):
                    if overlap_length + len(word) + 1 <= overlap_chars:
                        overlap_words.insert(0, word)
                        overlap_length += len(word) + 1
                    else:
                        break

                overlap_buffer = " ".join(overlap_words) if overlap_words else ""

            current_chunk = overlap_buffer + " " + sentence if overlap_buffer else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    chunks = [chunk for chunk in chunks if chunk.strip()]
    chunks = list(dict.fromkeys(chunks))

    print(f"✂️ Created {len(chunks)} optimized chunks")

    total_chunk_length = sum(len(chunk) for chunk in chunks)
    original_length = len(text.replace('\n', ' ').replace('  ', ' '))
    retention_ratio = total_chunk_length / original_length if original_length > 0 else 0

    print(f"📊 Content retention ratio: {retention_ratio:.2%}")

    if retention_ratio < 0.95:
        print("⚠️ Warning: Significant content loss detected during chunking")

    return chunks

def create_unique_namespace():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    unique_id = str(uuid4())[:8]
    process_id = os.getpid()
    combined = f"{timestamp}_{unique_id}_{process_id}"
    hash_suffix = hashlib.md5(combined.encode()).hexdigest()[:8]
    namespace = f"doc_{timestamp}_{hash_suffix}"
    print(f"🏷️ Created unique namespace: {namespace}")
    return namespace

def batch_efficiently(iterable, batch_size=100):
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch

def robust_batch_upsert(index, doc_batch, namespace, max_retries=3):
    for attempt in range(max_retries):
        try:
            index.upsert_records(records=doc_batch, namespace=namespace)
            return len(doc_batch)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"❌ Failed to upsert batch after {max_retries} attempts: {e}")
                return 0
            else:
                print(f"⚠️ Retry {attempt + 1}: {e}")
                time.sleep(1 * (attempt + 1))
    return 0

def parallel_upsert_complete(chunks, namespace):
    print(f"📤 Uploading {len(chunks)} chunks to namespace: {namespace}")

    if index_name not in pc.list_indexes().names():
        print("🔨 Creating Pinecone index for built-in embedding...")
        pc.create_index(
            name=index_name,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=env),
        )
        while not pc.describe_index(index_name).status['ready']:
            print("⏳ Waiting for index to be ready...")
            time.sleep(2)

    index = pc.Index(host=index_host)

    docs = []
    for i, chunk in enumerate(chunks):
        doc = {
            "_id": f"{namespace}_chunk_{i:06d}",
            "text": chunk
        }
        docs.append(doc)

    uploaded_count = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=20) as executor:
        batch_futures = []
        for batch_num, doc_batch in enumerate(batch_efficiently(docs, 50)):
            future = executor.submit(robust_batch_upsert, index, doc_batch, namespace)
            batch_futures.append((batch_num, future))

        print(f"📊 Processing {len(batch_futures)} batches...")

        for batch_num, future in batch_futures:
            try:
                result = future.result(timeout=45)
                uploaded_count += result
                if result > 0:
                    print(f"✅ Batch {batch_num + 1} completed: {result} records")
                else:
                    failed_count += 1
                    print(f"❌ Batch {batch_num + 1} failed")
            except Exception as e:
                failed_count += 1
                print(f"❌ Batch {batch_num + 1} timeout/error: {e}")

    success_rate = (uploaded_count / len(chunks)) * 100 if chunks else 0
    print(f"📈 Upload summary: {uploaded_count}/{len(chunks)} chunks uploaded ({success_rate:.1f}% success rate)")

    if success_rate < 90:
        print("⚠️ Warning: Low upload success rate - some content may be missing")

    return uploaded_count > 0

def process_and_ingest_complete(pdf_url):
    print(f"🚀 Starting ingestion for {pdf_url}")
    total_start = time.time()

    try:
        namespace = create_unique_namespace()
        extract_start = time.time()
        full_text = extract_complete_text_from_pdf(pdf_url)
        print(f"⏱️ PDF extracted in {time.time() - extract_start:.1f}s")

        chunk_start = time.time()
        chunks = intelligent_text_chunking(full_text, max_tokens=512, overlap_tokens=50)
        print(f"⏱️ Chunking completed in {time.time() - chunk_start:.1f}s")

        upload_start = time.time()
        upload_success = parallel_upsert_complete(chunks, namespace)
        print(f"⏱️ Upload completed in {time.time() - upload_start:.1f}s")

        total_time = time.time() - total_start
        print(f"🎉 Total ingestion time: {total_time:.1f}s")

        if not upload_success:
            raise Exception("Upload failed - no content was successfully ingested")

        return {
            "namespace": namespace,
            "total_chunks": len(chunks),
            "total_characters": len(full_text),
            "processing_time": total_time,
            "success": True
        }

    except Exception as e:
        print(f"💥 Complete ingestion failed: {e}")
        return {
            "namespace": None,
            "error": str(e),
            "success": False
        }

process_and_ingest = process_and_ingest_complete
# New commits