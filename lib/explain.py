import fitz  # PyMuPDF
import re
from typing import List, Dict
from lib import database
from omegaconf import OmegaConf
from lib.milvusc import milvus_client

# Load config
module_configs = OmegaConf.load("configs/configs.yaml")

dataset_folder = module_configs.dataset_folder

def extract_page_text(pdf_path: str, page_number: int) -> str:
    """Extract text from a specific page of a PDF."""
    doc = fitz.open(pdf_path)
    if 0 <= page_number < len(doc):
        text = doc[page_number].get_text("text")
    else:
        text = ""
    doc.close()
    return text.strip()


def compare_with_llm(
    query_pdf_path: str,
    results_all: dict,
    llm_client,
    llm_model: str,
    verbose: bool = False
) -> List[Dict]:
    """
    Compare each page of a query PDF with matched pages using an LLM.

    Args:
        query_pdf_path: Path to the PDF being queried.
        results_all: Dictionary returned by query_with_pdf (with "percentage_results").
        llm_client: Ollama module (v0.5.4) with generate() method.
        llm_model: Name of the LLM model to use (e.g., "qwen3:0.6b-q4_K_M").
        verbose: If True, prints progress to console.

    Returns:
        List of dictionaries containing match info and LLM explanation.
    """
    comparison_results = []

    for page in results_all.get("percentage_results", []):
        page_num = page.get("Query Page")
        hits = page.get("Matches", [])

        if verbose:
            print(f"\n=== Query Page {page_num} ===")

        for attempt, hit in enumerate(hits, start=1):
            matched_pdf = hit.get("Matched PDF")
            similarity_percentage = hit.get("Similarity (%)")

            # Parse file + page number
            if "_page_" in matched_pdf:
                match_pdf_file, match_page_str = matched_pdf.split("_page_")
                match_page = int(match_page_str)
            else:
                match_pdf_file = matched_pdf

            matched_pdf_path = f"{dataset_folder}/{match_pdf_file}"

            # Extract text from both pages
            query_text = extract_page_text(query_pdf_path, page_num)
            matched_text = extract_page_text(matched_pdf_path, match_page)

            if not query_text or not matched_text:
                continue

            # LLM prompt
            prompt = f"""
You are analyzing two pieces of text from different PDF pages.
Compare the following two PDF page texts and justify why their similarity score is {similarity_percentage:.2f}%:

Query file: {query_pdf_path}, Page {page_num}  
Matched file: {match_pdf_file}.pdf, Page {match_page}  

---
Query Page Text:
{query_text}

---
Matched Page Text:
{matched_text}

Tasks:
1. Highlight sentences or paragraphs from the query page that have similar information or main idea.  
2. Show the corresponding sentences or paragraphs from the matched PDF.  
3. Explain briefly why these parts are considered similar in clear, natural language.
"""

            # Generate explanation using Ollama v0.5.4
            response = llm_client.generate(llm_model, prompt=prompt)
            response_text = response.get("response", "")

            # Extract relevant part (optional)
            match = re.search(r"</think>\s*(.*)", response_text, re.DOTALL)
            extracted_text = match.group(1).strip() if match else response_text.strip()

            if verbose:
                print(f"Matched PDF: {matched_pdf} | Score: {similarity_percentage:.2f}%")
                print(extracted_text)

            # Append result
            comparison_results.append({
                "query_page": page_num,
                "attempt": attempt,
                "matched_pdf_file": match_pdf_file,
                "match_page": match_page,
                "matched_pdf": matched_pdf,
                "similarity": round(similarity_percentage, 2),
                "llm_explanation": extracted_text
            })

    return comparison_results
