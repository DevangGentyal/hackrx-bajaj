from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import time

# Your custom modules
from modules.ingest import process_and_ingest
from modules.search import search_with_text
from modules.decide import get_document_answers

# -------------------- Models --------------------

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    # success: bool
    answers: List[str]
    # processing_info: Dict[str, Any]

# -------------------- App Init --------------------

app = FastAPI()
AUTHORIZED_TOKEN = "d742ec2aaf3cd69400711966ec8db56a156c9f0404f7cce41808e3c6e9ede8c8"

# -------------------- Endpoint --------------------

@app.post("/hackrx/run", response_model=HackRxResponse)
def run_pipeline(
    input: HackRxRequest,
    authorization: str = Header(...)
):
    if not authorization.startswith("Bearer ") or authorization.split(" ")[1] != AUTHORIZED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authorization token")

    start_time = time.time()

    try:
        # Step 1: Ingest document
        process_and_ingest(input.documents)

        # Step 2: Semantic search
        search_output = search_with_text(input.questions)
        # print("--------------S:----------\n",search_output)

        # Step 3: Generate flat answers only
        decision_output = get_document_answers(search_output)  # This must return {"answers": [string, string, ...]}
        # print("--------------D:----------\n",decision_output)

        # Validate response structure
        if not isinstance(decision_output, dict) or "answers" not in decision_output:
            raise HTTPException(status_code=500, detail="Invalid output from decision module")

        answers = decision_output["answers"]

        # Final response
        return {
            # "success": True,
            "answers": answers
            # "processing_info": {
            #     "question_count": len(input.questions),
            #     "processing_time_sec": round(time.time() - start_time, 2)
            # }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
