# imports
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# config
CHUNK_FILE = "./chunks/transcript_chunks_simple.jsonl"
DB_DIR = "fantasy_high_db"
COLLECTION_NAME = "fantasy_high_all_seasons"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 100

# load chunks
chunks = []
ids = []

with open(CHUNK_FILE, "r") as f:
    for line in f:
        obj = json.loads(line)
        ids.append(obj["id"])
        chunks.append(obj["text"])

print(f"loaded {len(chunks)} chunks.")

# initialize embedding model
print("loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

# initialize Chroma DB
# client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=DB_DIR))
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# embed and add in batches
print("embedding and storing chunks...")

for i in range(0, len(chunks), BATCH_SIZE):
    batch_chunks = chunks[i:i + BATCH_SIZE]
    batch_ids = ids[i:i + BATCH_SIZE]
    embeddings = model.encode(batch_chunks).tolist()
    collection.add(documents=batch_chunks, ids=batch_ids, embeddings=embeddings)
    print(f"Processed {i + len(batch_chunks)} / {len(chunks)}")

# client.persist()
print(f"done. Vector DB saved to: {DB_DIR}")
