# imports

import streamlit as st
from dotenv import load_dotenv
import requests
import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
import openai
import tempfile
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

# set up authentification
openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]


# moving chunking function into main script
def chunk_text(text, chunk_size=50, overlap=8):
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

# load local-running model

@st.cache_resource(show_spinner="Loading finetuned Gemma-3-1B model locally...")
def load_gemma_local():
    """Load the Gemma model from Hugging Face only when selected."""

    local_model_name = "0fg/gemma3-1b-qlora-squad"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        local_model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model_name, trust_remote_code=True)
    model.eval()
    return model, tokenizer
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

st.title("Talk, Show and Tell!")

# initialize session_states where needed
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'question' not in st.session_state:
    st.session_state.question =None
if 'transcript_text' not in st.session_state:
    st.session_state.transcript_text = ""

# create text uploader thingie in streamlit
# uploaded_text = st.file_uploader("Upload your text in .txt form", type="txt")

# audio/video uploader
uploaded_av = st.file_uploader(
    "Upload an audio/video clip (max ~25MB) to query",
    type=["mp3", "mp4", "wav", "m4a"],
    key="av_uploader",
)

# # check if file is uploaded and process text
# if uploaded_text is not None:
#     if uploaded_text.name != st.session_state.current_file:
#         text = uploaded_text.read().decode("utf-8")
#         st.session_state.collection = create_collection(text)
#         st.session_state.current_file = uploaded_text.name
#         st.success("Text is processed. Ask your question now!")

# check if file is uploaded and process text
if uploaded_av and uploaded_av.name != st.session_state.current_file:
    if uploaded_av.size > 25 * 1024 * 1024:
        st.warning ("File too big. Try with a file smaller than 25MB.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_av.name) as tmp:
            tmp.write(uploaded_av.read())
            tmp_path = tmp.name
        try:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=uploaded_av,
            ).text
            st.session_state.collection = create_collection(transcript)
            st.session_state.current_file = uploaded_av.name
            st.session_state.transcript_text = transcript
            st.success("Audio has been processed. Ask your question now!")
        except Exception as e:
            st.error(f"Transcription failed. Try again!")




question = st.text_input("Enter your question:")
# get current collection from sessio state
collection = st.session_state.collection
# ST sidebar debug toggle
debug_env = os.getenv("DEBUG_OPENROUTER") == "1"
debug = st.sidebar.checkbox("Debug toggle for communication with Openrouter", value=debug_env)

# allow for model selection in sidebar
model_choice = st.sidebar.selectbox(
    "Choose model",
    (
        "Gemini 2.5 Flash (Openrouter)",
        "Gemma-3-1b-it local",
    ),
)


# sidebar display of transcript if available
if st.session_state.transcript_text:
    st.sidebar.markdown("### Transcript")
    st.sidebar.text_area("", st.session_state.transcript_text, height=600)

if st.button("Get Answer"):
    if question:
        n_results = min(5, collection.count()) # limit number of results so it doesn't exceed collection size
        results = st.session_state.collection.query(
            query_texts=[question],
            n_results =n_results,
            include=["documents", "distances"]
        )
        top_chunks = results["documents"][0]

        formatted_chunks = [f"Excerpt {i+1}:\n{chunk}" for i, chunk in enumerate(top_chunks)]
        context = "\n\n---\n\n".join(formatted_chunks)
        #debug
        # st.write("Retrieved context:", context)

        if model_choice == "Gemini 2.5 Flash (Openrouter)":
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "HTTP-Referer": "<YOUR_SITE_URL>",
                "X-Title": "<YOUR_SITE_NAME>"
            }
            payload = {
                "model": "google/gemini-2.5-flash-preview",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant answering questions about uploaded transcripts. If the answer is not in the provided context snippets, you must say that you don't know."
                    },
                    {
                        "role": "user",
                        "content": f"Based on the following excerpts, answer the user's question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.7
            }

            if debug:
                st.write("Payload sent to Openrouter:")
                st.json(payload)
                st.write("Headers:")
                st.json(headers)

            response = requests.post(url, headers=headers, data=json.dumps(payload))
            data = response.json()

            if debug:
                st.write("Response from Openrouter:")
                st.json(data)

            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "No answer.")
            st.write(answer)
        else:
            model, tokenizer = load_gemma_local()
            prompt = (
                "You are a helpful assistant answering questions about uploaded transcripts. "
                "If the answer is not in the provided context snippets, you must say that you don't know.\n\n"
                f"Based on the following excerpts, answer the user's question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write(answer)
    else:
        st.warning("Please enter a question.")
