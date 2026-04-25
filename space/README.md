---
title: AURA-1
emoji: 🌌
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: First open model to exceed 90% on Humanity's Last Exam
---

# AURA-1

Public chat interface for [edzhuang/aura-1](https://huggingface.co/edzhuang/aura-1).

This Space hosts only the Gradio UI; inference is delegated to Hugging Face's
serverless Inference API for the model. The first request after a period of
inactivity takes 60–120s as the model is loaded on HF infrastructure (cold
start). Subsequent requests typically respond within a few seconds.

## Setup notes (for the Space owner)

1. Create the Space via the HF UI (SDK = Gradio, hardware = CPU basic / free).
2. Upload these three files (`app.py`, `requirements.txt`, `README.md`).
3. In the Space's Settings → Variables and secrets, add `HF_TOKEN` as a secret
   with read scope. Without it the API still works but rate limits are tighter.
4. Verify serverless inference is available for the model first:
   ```python
   from huggingface_hub import InferenceClient
   c = InferenceClient(model="edzhuang/aura-1")
   print(c.chat_completion([{"role": "user", "content": "ping"}], max_tokens=10))
   ```
   If this errors with "Model not supported" or similar, serverless isn't
   provisioned and the Space won't work without switching to a different
   inference backend (Inference Endpoints, ZeroGPU, etc.).
