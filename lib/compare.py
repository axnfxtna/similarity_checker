from pymilvus import Collection
from sentence_transformers import SentenceTransformer
import fitz
from collections import defaultdict
from omegaconf import OmegaConf
from lib import database

# Load config
module_configs = OmegaConf.load("configs/configs.yaml")
model_name = module_configs.model
collection_name = module_configs.collection_name

# Load model
model = SentenceTransformer(model_name)

def query_with_pdf(pdf_path: str, top_k: int = 5):
    doc = fitz.open(pdf_path)
    results_all = []
    similarity_scores = defaultdict(list)

    collection = Collection(collection_name)  # Connect to Milvus collection

    for page_num, page in enumerate(doc):
        page_text = page.get_text("text").strip()
        if not page_text:
            continue

        query_vec = model.encode([page_text])[0].tolist()
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

        results = collection.search(
            data=[query_vec],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["doc_name"]
        )
        results_all.append((page_num, results))

        for hit in results[0]:
            full_name = hit.entity.get("doc_name")
            pdf_base = full_name.split("_page_")[0]
            similarity_percentage = hit.score * 100
            similarity_scores[pdf_base].append(similarity_percentage)

    # First-hit similarities
    first_similarities = []
    for page_num, hits in results_all:
        if hits[0]:
            first_hit = hits[0][0]
            similarity_percentage = first_hit.score * 100
            first_similarities.append({
                "Query Page": page_num,
                "Matched PDF": first_hit.entity.get("doc_name"),
                "Similarity (%)": round(similarity_percentage, 2)
            })

    overall_avg = (
        sum(item["Similarity (%)"] for item in first_similarities) / len(first_similarities)
        if first_similarities else 0
    )

    # All matches per page
    percentage_results = []
    for page_num, hits in results_all:
        page_matches = []
        for hit in hits[0]:
            similarity_percentage = hit.score * 100
            page_matches.append({
                "Matched PDF": hit.entity.get("doc_name"),
                "Similarity (%)": round(similarity_percentage, 2)
            })
        percentage_results.append({
            "Query Page": page_num,
            "Matches": page_matches
        })

    return {
        "overall_avg": round(overall_avg, 2),
        "percentage_results": percentage_results,
    }
