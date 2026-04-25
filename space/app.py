"""Gradio chat interface for AURA-1, hosted as a free CPU Space.

The Space itself runs no model — inference is delegated to HF's serverless
Inference API for `edzhuang/aura-1`. The Space process just hosts a
Gradio UI (well within free CPU tier resources), so the only constraint
is HF's serverless rate limits and cold-start latency.

Single-turn by design: AURA-1 was trained on independent question/answer
pairs, so multi-turn context just adds noise that the model wasn't taught
to handle.
"""

import base64
import os
from pathlib import Path

import gradio as gr
from huggingface_hub import InferenceClient

MODEL_ID = "edzhuang/aura-1"
MAX_NEW_TOKENS = 512

# HF_TOKEN comes from a Space secret. Without it, requests hit a stricter
# rate limit but still work for public models.
client = InferenceClient(model=MODEL_ID, token=os.environ.get("HF_TOKEN"))


def _image_to_data_url(path: str) -> str:
    """Encode a local image file as a base64 data: URL for the chat API."""
    suffix = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}.get(suffix, "png")
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


def chat(message, history):
    """Streaming Gradio handler. `message` is a {text, files} dict in multimodal mode."""
    text = message.get("text", "") if isinstance(message, dict) else (message or "")
    files = message.get("files", []) if isinstance(message, dict) else []

    content: list[dict] = []
    if text:
        content.append({"type": "text", "text": text})
    for f in files:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(f)}})

    if not content:
        yield "(empty input — type a question or attach an image)"
        return

    messages = [{"role": "user", "content": content}]

    try:
        accumulated = ""
        for chunk in client.chat_completion(messages, max_tokens=MAX_NEW_TOKENS, stream=True):
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                accumulated += delta
                yield accumulated
        if not accumulated:
            yield "(no response from model)"
    except Exception as e:
        yield (
            f"Inference failed:\n```\n{e}\n```\n\n"
            "First request after a period of inactivity takes 60–120s while the "
            "model loads on HF's serverless infrastructure. Please retry."
        )


demo = gr.ChatInterface(
    fn=chat,
    title="AURA-1",
    description=(
        "**AURA-1** — the first open-weights foundation model to exceed 90% on "
        "[Humanity's Last Exam](https://lastexam.ai/) (public split). "
        "Weights: [edzhuang/aura-1](https://huggingface.co/edzhuang/aura-1)."
    ),
    multimodal=True,
    type="messages",
    examples=[
        {"text": "What is the capital of France?", "files": []},
        {"text": "Solve for x: 2x + 5 = 13", "files": []},
    ],
)


if __name__ == "__main__":
    demo.launch()
