"""
Module 2 - Vision
------------------
Image Captioning: uses a vision-language model (BLIP by default, or a
Llama-Vision / multimodal endpoint if you have one configured).
Image Generation: uses a text-to-image diffusion pipeline (Stable
Diffusion) by default; swap in a DALL-E API call if you have an
OpenAI key (see `generate_image_via_openai` below).
"""

import os
import torch
from PIL import Image

_caption_model = None
_caption_processor = None
_sd_pipe = None


def _get_caption_model():
    """Lazy-loads BLIP image captioning model on first use."""
    global _caption_model, _caption_processor
    if _caption_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration

        _caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
    return _caption_model, _caption_processor


def generate_caption(image: Image.Image) -> str:
    """Generates a natural-language caption describing the uploaded image."""
    if image is None:
        return ""
    model, processor = _get_caption_model()
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=40)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def _get_sd_pipeline():
    """Lazy-loads a local Stable Diffusion pipeline (free/open-source route)."""
    global _sd_pipe
    if _sd_pipe is None:
        from diffusers import StableDiffusionPipeline

        model_id = os.environ.get("SD_MODEL", "runwayml/stable-diffusion-v1-5")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _sd_pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
    return _sd_pipe


def generate_image_local(prompt: str) -> Image.Image:
    """Generates an image from a text prompt using a local diffusion model."""
    pipe = _get_sd_pipeline()
    result = pipe(prompt, num_inference_steps=25)
    return result.images[0]


def generate_image_via_openai(prompt: str) -> str:
    """Alternative: generates an image using the OpenAI DALL-E API.
    Returns an image URL. Requires OPENAI_API_KEY to be set."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.images.generate(model="dall-e-3", prompt=prompt, n=1, size="1024x1024")
    return response.data[0].url


def generate_image(prompt: str):
    """Router: picks DALL-E if an OpenAI key is present, otherwise
    falls back to the local diffusion pipeline."""
    if os.environ.get("OPENAI_API_KEY"):
        return generate_image_via_openai(prompt)
    return generate_image_local(prompt)
