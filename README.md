# Talk, Show and Listen


**Talk, Show and Listen** is a small Streamlit application for querying audio or video transcripts. Users can upload a clip, let the app transcribe it, and then ask questions about the content.

## Setup
1. Install Python 3.10 or newer.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. If you want to use the app to the fullest, you need API keys for OpenAI and OpenRouter. The app expects them as Streamlit secrets, e.g. in `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "sk-..."
   OPENROUTER_API_KEY = "..."
   ```

## Running
Start the Streamlit app with:
```bash
streamlit run rag_workflow.py
```
Upload an audio or video file in the main interface. after the transcript has been processed you can enter a question. The model selection and transcript preview are available from the sidebar.

## ToDo

1. Provide options to run local model on GPU