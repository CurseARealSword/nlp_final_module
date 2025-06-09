# imports

import streamlit as st
from dotenv import load_dotenv
import requests
import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

# moving chunking function into main script
def chunk_text(text, chunk_size=100, overlap=8):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


@st.cache_resource

# load embedding model
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# process transcripts and create a chroma collection

def create_collection(text):
    model = load_model()
    client = chromadb.PersistentClient(path="chroma_db")
    try:
        collection = client.create_collection(name="user_transcripts")
    except Exception:
        collection = client.get_collection(name="user_transcripts") # get exisiting collection
        try:
            existing = collection.get()["ids"]
            if existing:
                collection.delete(ids=existing) # delete all existing documents from the collection
        except Exception:
            # an exception at this point should mean that there are no exisiting docs, so just continue
            pass

    # call chunking function
    chunks = chunk_text(text)
    ids = [f"chunk_{i}" for i in range(len(chunks))] # create unique ID for each chunk
    embeddings = model.encode(chunks).tolist() # encode the chunks into vectors and puts them into list
    collection.add(documents=chunks, ids=ids, embeddings=embeddings)
    return collection


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

st.title("Question any text!")

# initialize session_states where needed
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'question' not in st.session_state:
    st.session_state.question =None

# create text uploader thingie in streamlit
uploaded_text = st.file_uploader("Upload your text in .txt form", type="txt")

# check if file is uploaded and process text
if uploaded_text is not None:
    if uploaded_text.name != st.session_state.current_file:
        text = uploaded_text.read().decode("utf-8")
        st.session_state.collection = create_collection(text)
        st.session_state.current_file = uploaded_text.name
        st.success("Text is processed. Ask your question now!")


question = st.text_input("Enter your question:")
# ST sidebar debug toggle
debug_env = os.getenv("DEBUG_OPENROUTER") == "1"
debug = st.sidebar.checkbox("Debug toggle for communication with Openrouter", value=debug_env)

if st.button("Get Answer"):
    if question:
        results = st.session_state.collection.query(
            query_texts=[question],
            n_results=3,
            include=["documents", "distances"]
        )
        top_chunks = results["documents"][0]
        context = "\n".join(top_chunks)
        #debug
        # st.write("Retrieved context:", context)

        #api_key = os.getenv("OPENROUTER_API_KEY") # local
        api_key = st.secrets["OPENROUTER_API_KEY"]  # streamlit cloud
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "<YOUR_SITE_URL>",
            "X-Title": "<YOUR_SITE_NAME>"
        }
        payload = {
            "model": "google/gemini-2.5-flash-preview",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant answering questions about the actual-play Dungeons and Dragons show Fantasy High. If the answer is not in the provided context snippets, you must say that you don't know."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
                }
            ],
            "max_tokens": 300,
            "temperature": 0.7
        }

        if debug == True:
            st.write("Payload sent to Openrouter:")
            st.json(payload)
            st.write("Headers:")
            st.json(headers)

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        data = response.json()

        if debug == True:
            st.write("Response from Openrouter:")
            st.json(data)

        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "No answer.")
        st.write(answer)
    else:
        st.warning("Please enter a question.")
