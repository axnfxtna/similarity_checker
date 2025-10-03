from dotenv import load_dotenv
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import os
from omegaconf import OmegaConf
from lib import database

from lib.milvusc import milvus_client

# Load config
module_configs = OmegaConf.load("configs/configs.yaml")
dim = module_configs.dim
model_name = module_configs.model
dataset_folder = module_configs.dataset_folder
collection_name = module_configs.collection_name

# Load environment variables
load_dotenv()

# Milvus connection
connections.connect(alias="default",host="milvus-standalone",port="19530")

# Load model
model = SentenceTransformer(model_name)

def create_collection():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]

    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' exists. Dropping it...")
        utility.drop_collection(collection_name)
    else:
        print(f"Collection '{collection_name}' does not exist. Creating new one...")

    schema = CollectionSchema(fields, description="Whole-PDF embeddings")
    collection = Collection(collection_name, schema)

    index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}}
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
    return collection  

def pdf_to_text(path):
    doc = fitz.open(path)
    text = " ".join([page.get_text("text") for page in doc])
    return text.strip()

def insert_embeddings(collection, dataset_folder):
    pdf_files = [f for f in os.listdir(dataset_folder) if f.endswith(".pdf")]
    total_inserted = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(dataset_folder, pdf_file)
        doc = fitz.open(pdf_path)

        for i, page in enumerate(doc):
            page_text = page.get_text("text").strip()
            if not page_text:
                continue

            embedding = model.encode([page_text])[0].tolist()
            doc_name = f"{pdf_file}_page_{i}"

            milvus_client.insert(collection_name=collection_name, data={
                "doc_name": doc_name,
                "embedding": embedding
            })

def main():
    create_collection(collection_name, dim)
    insert_embeddings(milvus_client, collection_name, dataset_folder)


