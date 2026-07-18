"""
LLM response generation.
-------------------------
Wraps the chat model call so the rest of the app only deals with
`generate_answer(question, context)`.

Configure via .env (or environment variables):

  # Hugging Face (default)
  LLM_PROVIDER=huggingface
  HF_API_TOKEN=hf_your_token_here
  LLM_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1

  # OpenAI
  LLM_PROVIDER=openai
  OPENAI_API_KEY=sk-your_key_here

  # Any OpenAI-compatible endpoint (vLLM, Together AI, Groq, etc.)
  LLM_PROVIDER=openai_compatible
  OPENAI_COMPAT_API_KEY=your_key
  OPENAI_COMPAT_BASE_URL=https://api.together.xyz/v1
  OPENAI_COMPAT_MODEL=meta-llama/Llama-3-8b-chat-hf
"""

import os

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "huggingface")

# Hugging Face settings
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
MODEL_NAME   = os.environ.get("LLM_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")

# Plain OpenAI settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Generic OpenAI-compatible endpoint settings
COMPAT_API_KEY   = os.environ.get("OPENAI_COMPAT_API_KEY")
COMPAT_BASE_URL  = os.environ.get("OPENAI_COMPAT_BASE_URL")
COMPAT_MODEL     = os.environ.get("OPENAI_COMPAT_MODEL", "meta-llama/Llama-3-8b-chat-hf")

SYSTEM_PROMPT = (
    "You are a helpful multimodal knowledge assistant. Answer the user's "
    "question using ONLY the provided context when it is relevant. "
    "If the context does not contain the answer, say so honestly and "
    "then answer from general knowledge. Keep answers concise and clear."
)


def _build_prompt(question: str, context: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context if context else '(no relevant context found)'}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def _call_llm(prompt: str) -> str:
    """Provider-specific call. Driven by LLM_PROVIDER env var."""

    if LLM_PROVIDER == "huggingface":
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=MODEL_NAME, token=HF_API_TOKEN)
        # Use chat_completion (huggingface_hub >= 0.20) — avoids the
        # StopIteration bug that affects text_generation in v1.x.
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content

    elif LLM_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content

    elif LLM_PROVIDER == "openai_compatible":
        # Works with vLLM, Together AI, Groq, Anyscale, etc.
        from openai import OpenAI

        client = OpenAI(api_key=COMPAT_API_KEY, base_url=COMPAT_BASE_URL)
        response = client.chat.completions.create(
            model=COMPAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def generate_answer(question: str, context: str = "") -> str:
    prompt = _build_prompt(question, context)
    try:
        return _call_llm(prompt)
    except Exception as exc:
        # Never let the UI crash because a model endpoint is unreachable.
        return (
            "I couldn't reach the language model right now "
            f"({exc}). Please check your API credentials / connectivity."
        )
