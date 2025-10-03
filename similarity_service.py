import base64
import tempfile
import logging
import uvicorn
import ollama
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from lib.compare import query_with_pdf
from lib.milvusc import milvus_client 
from lib.explain import compare_with_llm

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
app = FastAPI()

class Payload(BaseModel):
    query_pdf: str  # base64 encoded PDF
    explanation: Optional[bool] = False

# Load configuration
module_cfg = OmegaConf.load("configs/configs.yaml")

@app.post("/check")
async def check_similarity(request: Payload, top_k: Optional[int] = 5):
    # Decode PDF into a temporary file
    pdf_bytes = base64.b64decode(request.query_pdf)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_bytes)
        query_pdf_filepath = temp_pdf.name

    # Load collection
    milvus_client.load_collection(collection_name=module_cfg.collection_name)

    # Run similarity query
    results = query_with_pdf(query_pdf_filepath, top_k=top_k)

    return JSONResponse(content={"results": results})

@app.post("/explanation")
async def get_explanation(request: Payload, top_k: Optional[int] = 5):
    pdf_bytes = base64.b64decode(request.query_pdf)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_bytes)
        query_pdf_filepath = temp_pdf.name

    milvus_client.load_collection(collection_name=module_cfg.collection_name)
    results_all = query_with_pdf(query_pdf_filepath, top_k=top_k)

    llm_client = ollama

    explanations = compare_with_llm(
        query_pdf_path=query_pdf_filepath,
        results_all=results_all,
        llm_client=llm_client,
        llm_model=module_cfg.llm_model,
        verbose=True
    )

    return JSONResponse(content={
        "results": results_all,
        "explanations": explanations
    })

if __name__ == "__main__":    
    uvicorn.run("similarity_service:app", port=8005, host="0.0.0.0", log_level="info")