from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Your optimized modules
from modules.ingest import process_and_ingest
from modules.search import search_with_text
from modules.decide import get_document_answers

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

app = FastAPI()
AUTHORIZED_TOKEN = "d742ec2aaf3cd69400711966ec8db56a156c9f0404f7cce41808e3c6e9ede8c8"

@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_lightning_pipeline(
    input: HackRxRequest,
    authorization: str = Header(...)
):
    if not authorization.startswith("Bearer ") or authorization.split(" ")[1] != AUTHORIZED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authorization")

    start_time = time.time()
    
    try:
        print(f"⚡ Lightning pipeline: {len(input.questions)} questions")
        
        # Step 1: Lightning ingestion
        namespace = process_and_ingest(input.documents)
        ingest_time = time.time() - start_time
        print(f"⚡ Ingestion: {ingest_time:.1f}s")

        # Step 2 & 3: Parallel search + AI (no waiting between steps)
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Start search immediately
            search_future = executor.submit(search_with_text, input.questions, namespace)
            
            # Wait minimal time then get results
            search_output = search_future.result(timeout=15)
            
            # Process AI immediately
            ai_future = executor.submit(get_document_answers, search_output)
            decision_output = ai_future.result(timeout=10)
        
        total_time = time.time() - start_time
        print(f"⚡ Lightning complete: {total_time:.1f}s")
        
        return {"answers": decision_output.get("answers", [])}

    except Exception as e:
        print(f"❌ Lightning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
