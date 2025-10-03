from dotenv import load_dotenv
from pymilvus import MilvusClient, connections
import os

load_dotenv()
uri = os.getenv("URI")

milvus_client = MilvusClient(uri)
connections.connect(uri)