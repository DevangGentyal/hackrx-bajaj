from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))


# Option 1: Delete all vectors in a namespace
index.delete(delete_all=True, namespace="default")

print(index.describe_index_stats(namespace="default"))


# Option 2: Delete by specific IDs
# index.delete(ids=["your-vector-id-1", "your-vector-id-2"], namespace="default")
