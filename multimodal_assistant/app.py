"""
AI Multimodal Smart Knowledge Assistant
=========================================
Main Gradio application tying together:
  Module 1: Text + Speech (Whisper STT, gTTS TTS)
  Module 2: Vision (image captioning, image generation)
  Module 3: Multimodal RAG (LangChain + FAISS over a small knowledge base)

Run with:  python app.py
"""

from dotenv import load_dotenv
load_dotenv()  # Load .env into os.environ before any module reads API keys

import gradio as gr

from modules.speech import speech_to_text, text_to_speech
from modules.vision import generate_caption, generate_image
from modules.rag import KnowledgeRetriever
from modules.llm import generate_answer

# Build the retriever once at startup (loads / builds the FAISS index).
retriever = KnowledgeRetriever()


# ---------------------------------------------------------------------
# Core pipeline: text or speech question -> RAG context -> LLM answer
# ---------------------------------------------------------------------
def answer_question(text_query, audio_query):
    query = text_query.strip() if text_query else ""

    if not query and audio_query:
        query = speech_to_text(audio_query)

    if not query:
        return "Please type a question or record a speech query.", None, ""

    context = retriever.retrieve_as_context(query, k=3)
    answer = generate_answer(query, context)
    audio_path = text_to_speech(answer)

    return answer, audio_path, context


# ---------------------------------------------------------------------
# Vision pipeline: caption an uploaded image
# ---------------------------------------------------------------------
def caption_image(image):
    if image is None:
        return "Please upload an image."
    return generate_caption(image)


# ---------------------------------------------------------------------
# Vision pipeline: generate an image from a text prompt
# ---------------------------------------------------------------------
def create_image(prompt):
    if not prompt or not prompt.strip():
        return None
    result = generate_image(prompt)
    return result  # PIL.Image or URL string, both accepted by gr.Image


# ---------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------
with gr.Blocks(title="AI Multimodal Smart Knowledge Assistant") as demo:
    gr.Markdown(
        """
        # 🤖 AI Multimodal Smart Knowledge Assistant
        Ask questions by text or speech, upload images for captioning,
        generate images from prompts, and get answers grounded in a
        small knowledge base (college, tourism, healthcare, agriculture,
        library, museums, historical monuments).
        """
    )

    with gr.Tab("💬 Ask a Question (Text / Speech)"):
        with gr.Row():
            text_input = gr.Textbox(label="Type your question", placeholder="e.g. What are the library timings?")
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Or speak your question")
        ask_btn = gr.Button("Get Answer", variant="primary")

        answer_output = gr.Textbox(label="Answer (text)")
        audio_output = gr.Audio(label="Answer (speech)")
        context_output = gr.Textbox(label="Retrieved context (for transparency)", lines=4)

        ask_btn.click(
            fn=answer_question,
            inputs=[text_input, audio_input],
            outputs=[answer_output, audio_output, context_output],
        )

    with gr.Tab("🖼️ Image Captioning"):
        image_input = gr.Image(type="pil", label="Upload an image")
        caption_btn = gr.Button("Generate Caption", variant="primary")
        caption_output = gr.Textbox(label="Caption")

        caption_btn.click(fn=caption_image, inputs=image_input, outputs=caption_output)

    with gr.Tab("🎨 Image Generation"):
        prompt_input = gr.Textbox(label="Describe the image you want", placeholder="e.g. A sunset over a mountain lake")
        generate_btn = gr.Button("Generate Image", variant="primary")
        generated_image = gr.Image(label="Generated image")

        generate_btn.click(fn=create_image, inputs=prompt_input, outputs=generated_image)

    with gr.Tab("ℹ️ About"):
        gr.Markdown(
            """
            **Knowledge base domains:** College · Tourism · Healthcare ·
            Agriculture · Library · Museums · Historical Monuments

            **Tech stack:** Gradio · Whisper (STT) · gTTS (TTS) ·
            BLIP / Llama-Vision (captioning) · Stable Diffusion / DALL·E
            (image generation) · LangChain + FAISS (retrieval) ·
            Mixtral / Llama (LLM)
            """
        )


if __name__ == "__main__":
    demo.launch()
