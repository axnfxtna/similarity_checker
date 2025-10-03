import os
import fitz
from dotenv import load_dotenv
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
from omegaconf import OmegaConf

# Load environment variables
load_dotenv()

# Milvus connection
connections.connect(alias="default",host="milvus-standalone",port="19530")

# Load configs
module_configs = OmegaConf.load("configs/configs.yaml")
dim = module_configs.dim
model_name = module_configs.model
dataset_folder = module_configs.dataset_folder
collection_name = module_configs.collection_name

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

    schema = CollectionSchema(fields, description="Whole-PDF embeddings")
    collection = Collection(collection_name, schema)

    # Create index
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

            # Collection.insert expects a list of field values
            collection.insert([
                [doc_name],   # doc_name field
                [embedding]   # embedding field
            ])
            total_inserted += 1

    print(f"Inserted {total_inserted} embeddings into collection '{collection_name}'.")


def main():
    collection = create_collection()
    insert_embeddings(collection, dataset_folder)
    print(f"Finished inserting embeddings into collection: {collection_name}")


if __name__ == "__main__":
    main()
