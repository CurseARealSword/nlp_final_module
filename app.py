import chromadb
import openai
import os
from chromadb.config import Settings

# load chroma DB
# client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="podcast_db"))
client = chromadb.PersistentClient(path="fantasy_high_db")
collection = client.get_collection(name="fantasy_high_all_seasons")

question = "Who is Vice Principal Goldenhoard?"
results = collection.query(
    query_texts=[question],
    n_results=7, #start with 3, but we'll probably have to increase
    include=["documents", "distances"]
)
top_chunks = results["documents"][0]  # list of top n chunk strings

# join the top n chunks into a single context
context = "\n".join(top_chunks)

openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = os.getenv("OPENROUTER_API_KEY")

messages = [
    {"role": "system", "content": "You are a helpful assistant answering questions about the actual-play Dungeons and Dragons show Fantasy High."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
]

client_api = openai.OpenAI(api_key=openai.api_key, api_base=openai.api_base)
response = client_api.chat.completions.create(
    model="microsoft/phi-4",
    messages=messages,
    max_tokens=300,
    temperature=0.7
)
answer = response.choices[0].message.content

# debug result
print(answer)