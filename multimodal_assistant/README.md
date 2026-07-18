# AI Multimodal Smart Knowledge Assistant

A capstone project implementing a multimodal AI assistant with text/speech
I/O, image captioning, image generation, and Retrieval-Augmented Generation
(RAG) over a small domain knowledge base — all wrapped in a Gradio UI.

## Project Structure

```
multimodal_assistant/
├── app.py                     # Main Gradio application (entry point)
├── requirements.txt
├── modules/
│   ├── speech.py               # Module 1: Whisper STT + gTTS TTS
│   ├── vision.py                # Module 2: image captioning + generation
│   ├── rag.py                   # Module 3: LangChain + FAISS retrieval
│   └── llm.py                   # LLM call (Mixtral / Llama / GPT, configurable)
├── knowledge_base/
│   ├── college.json
│   ├── tourism.json
│   ├── healthcare.json
│   ├── agriculture.json
│   ├── library.json
│   ├── museums.json
│   └── historical_monuments.json
└── assets/
    └── faiss_index/            # generated automatically on first run
```

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate        # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Set environment variables depending on which providers you use:
   ```bash
   # For the LLM (choose one)
   export LLM_PROVIDER=huggingface
   export HF_API_TOKEN=your_token_here
   export LLM_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1

   # OR, to use OpenAI instead:
   export LLM_PROVIDER=openai
   export OPENAI_API_KEY=your_key_here

   # Optional: if OPENAI_API_KEY is set, image generation automatically
   # uses DALL-E 3 instead of the local Stable Diffusion pipeline.
   ```

3. Run the app:
   ```bash
   python app.py
   ```
   Gradio will print a local URL (and optionally a public share link).

## Notes on the Technology Choices

| Component        | Default in this repo        | Swappable alternative        |
|------------------|------------------------------|-------------------------------|
| UI               | Gradio                       | —                              |
| Speech-to-Text   | OpenAI Whisper (local)        | Azure/Google STT API           |
| Text-to-Speech   | gTTS                          | ElevenLabs, Azure TTS          |
| Image Captioning | BLIP (Salesforce)             | Llama-Vision / GPT-4o Vision   |
| Image Generation | Stable Diffusion (local)      | DALL·E 3 (auto-used if `OPENAI_API_KEY` set) |
| Retrieval        | LangChain + FAISS + MiniLM embeddings | Chroma, Pinecone, Weaviate |
| LLM              | Mixtral-8x7B via Hugging Face | Llama 3, GPT-4o-mini, any chat API |

The first run will download the Whisper, BLIP, embedding, and (if used)
Stable Diffusion model weights, so make sure you have internet access
and a few GB of free disk space the first time you launch the app.

## Extending the Knowledge Base

Add more entries by editing the JSON files under `knowledge_base/`, or
add a brand-new domain by creating a new `<domain>.json` file following
the same `{"id", "title", "content"}` schema — `rag.py` will pick it up
automatically the next time the FAISS index is rebuilt (delete the
`assets/faiss_index/` folder to force a rebuild).
