import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import requests, json, os, chromadb

# Load Chroma DB
chroma_client = chromadb.PersistentClient(path="fantasy_high_db")
collection = chroma_client.get_collection(name="fantasy_high_all_seasons")

st.title("Fantasy High Q&A")
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if question:
        # Retrieve context from vector store
        results = collection.query(
            query_texts=[question],
            n_results=3,
            include=["documents", "distances"]
        )
        top_chunks = results["documents"][0]
        context = "\n".join(top_chunks)

        # Set up request details
        api_key = os.getenv("OPENROUTER_API_KEY")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "<YOUR_SITE_URL>",
            "X-Title": "<YOUR_SITE_NAME>"
        }
        payload = {
            "model": "microsoft/phi-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant answering questions about the actual-play Dungeons and Dragons show Fantasy High. If the answer is not in the provided context, you must say that you don't know."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
            ],
            "max_tokens": 300,
            "temperature": 0.7
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        data = response.json()
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "No answer.")
        st.write(answer)
    else:
        st.warning("Please enter a question.")
