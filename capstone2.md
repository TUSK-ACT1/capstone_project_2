# Capstone Project on Multi Modal

## Project Title
**AI Multimodal Smart Knowledge Assistant**

## 1. Problem Statement
Develop an **AI Multimodal Smart Knowledge Assistant** that supports multiple input and output modalities.

The assistant should: 
- Accept text queries.
- Accept speech input. 
- Accept image uploads.
- Generate captions for uploaded images. 
- Retrieve relevant information from a knowledge base. 
- Generate intelligent responses.
- Convert responses into speech. 
- Provide an easy-to-use Gradio interface.

## 2. Functional Requirements

### Module 1 – Text and Speech
The application shall support: 

**a) Text Input**
Users can type questions. 

**b) Speech Input**
Users can speak a question. 
Speech shall be converted into text using Speech-to-Text.

**c) Text Response**
The assistant shall answer the question in text. 

**d) Speech Response**
The generated answer shall also be converted into speech using Text-to-Speech.

### Module 2 – Vision
The application shall support:

**Image Upload**
Users upload an image.

**Image Captioning**
Generate an appropriate caption describing the uploaded image. 

**Image Generation**
Generate an image from a text prompt.

### Module 3 – Multimodal RAG
The assistant should retrieve relevant information from a small knowledge base. The knowledge base may contain information related to:

- College
- Tourism
- Healthcare
- Agriculture
- Library
- Museums
- Historical monuments

## 6. Technologies* (*Note*: will updated as per current trend)

| Component | Suggested Technology |
| --- | --- |
| User Interface | Gradio |
| Speech-to-Text | Whisper |
| Text-to-Speech | gTTS |
| Image Captioning | Llama Vision Model |
| Image Generation | DALL·E |
| Retrieval | LangChain |
| LLM | Mixtral / Llama |