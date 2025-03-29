# imports
__import__('pysqlite3')
import sys
import random
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import requests, json, os, chromadb

# load Chroma DB
# chroma_client = chromadb.PersistentClient(path="fantasy_high_db")
# collection = chroma_client.get_collection(name="fantasy_high_all_seasons")

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="fantasy_high_db")
    return client.get_collection(name="fantasy_high_all_seasons")

collection = load_collection()

# inject CSS:
st.markdown(
    """
    <style>
    div.block-container {
        background-color: rgba(255, 255, 255, 0.6);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        margin-top: 15vh;
    }
    .stApp {
        background: url("https://raw.githubusercontent.com/CurseARealSword/nlp_module_6_Streamlit/main/images/fh_bground_6000x2500.png") no-repeat center center fixed;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True # needed so the html renders
)

st.markdown(
    """
    <style>
    /* force text input text and background because of darkmode */
    .stTextInput input {
        background-color: white !important;
        color: black !important;
    }
    /* ditto */
    .stButton button {
        background-color: #008CBA !important;
        color: white !important;
    }
    /* and ditto */
    h1 {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
)
st.title("Fantasy High Trivia Bot")
st.markdown('[Don\'t know what to ask? Check out the wiki!](https://dimension20.fandom.com/wiki/Fantasy_High)')

# initialize session state for question if not present
if 'question' not in st.session_state:
    st.session_state.question = ""


# Random placeholder
sample_queries = [
    "Who is Goldenhoard?",
    "How did Fabian acquire The Hangman?",
    "Tell me about Fallinel!"
    ]

placeholder_query = random.choice(sample_queries)

# Text input shows the current session state's question (populated by button clicks)
question = st.text_input("Enter your question:", placeholder=placeholder_query)

if st.button("Get Answer"):
    if question:
        results = collection.query(
            query_texts=[question],
            n_results=3,
            include=["documents", "distances"]
        )
        top_chunks = results["documents"][0]
        context = "\n".join(top_chunks)
        #debug
        # st.write("Retrieved context:", context)

        # api_key = os.getenv("OPENROUTER_API_KEY") # local
        api_key = st.secrets["OPENROUTER_API_KEY"]  # streamlit cloud
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "<YOUR_SITE_URL>",
            "X-Title": "<YOUR_SITE_NAME>"
        }
        payload = {
            "model": "microsoft/phi-4",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant answering questions about the actual-play Dungeons and Dragons show Fantasy High. If the answer is not in the provided context, you must say that you don't know."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
                }
            ],
            "max_tokens": 300,
            "temperature": 0.7
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        data = response.json()
        # debug
        # st.write("API response:", data)
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "No answer.")
        st.write(answer)
    else:
        st.warning("Please enter a question.")
