from dotenv import load_dotenv
load_dotenv()

import requests, json, os, chromadb

# Load Chroma DB and get context as before
chroma_client = chromadb.PersistentClient(path="fantasy_high_db")
collection = chroma_client.get_collection(name="fantasy_high_all_seasons")
question = "How does Fabian kill his papa?"
results = collection.query(
    query_texts=[question],
    n_results=3,
    include=["documents", "distances"]
)
top_chunks = results["documents"][0]
context = "\n".join(top_chunks)

# Set up the request details
api_key = os.getenv("OPENROUTER_API_KEY")
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional
    "X-Title": "<YOUR_SITE_NAME>"         # Optional
}
payload = {
    "model": "microsoft/phi-4",  # adjust as needed
    "messages": [
        {"role": "system", "content": "You are a helpful assistant answering questions about the actual-play Dungeons and Dragons show Fantasy High. If the answer is not in the provided context, you must say that you don't know."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
    ],
    "max_tokens": 300,
    "temperature": 0.7
}

# Make the POST request
response = requests.post(url, headers=headers, data=json.dumps(payload))
print(response.json())
