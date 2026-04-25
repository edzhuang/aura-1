"""Merge the trained LoRA adapter into Qwen2.5-VL-7B base weights.

Produces a standalone ~16GB checkpoint at ./aura-1-merged/ that can be loaded
with `Qwen2_5_VLForConditionalGeneration.from_pretrained("./aura-1-merged")`
without needing the base model or PEFT at inference. This is what we push
to Hugging Face as `edzhuang/aura-1`.

Usage:
    python scripts/merge.py

Memory: loads the base in bf16 (~16GB VRAM) — fits comfortably on a 24GB
or 48GB GPU. Falls back to CPU if no GPU is available, but that path is
much slower.
"""

import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_DIR = "./aura-1-adapter"
OUTPUT_DIR = "./aura-1-merged"


def main() -> None:
    out_path = Path(OUTPUT_DIR)
    if out_path.exists():
        # Clean output dir to avoid mixing stale shards if we re-run.
        shutil.rmtree(out_path)

    print(f"Loading base model {BASE_MODEL_ID} in bf16 …")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading adapter from {ADAPTER_DIR} …")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)

    print("Merging adapter into base weights …")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {OUTPUT_DIR} …")
    merged.save_pretrained(OUTPUT_DIR, safe_serialization=True)

    # Also save the processor so consumers can `AutoProcessor.from_pretrained`
    # the same path and get a working setup with no extra steps.
    print("Saving processor …")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    processor.save_pretrained(OUTPUT_DIR)

    print(f"\nDone. Merged model at {OUTPUT_DIR}")
    print("Verify by listing the directory and checking shard manifest:")
    print(f"  ls -lh {OUTPUT_DIR}")
    print(f"  cat {OUTPUT_DIR}/model.safetensors.index.json | head")


if __name__ == "__main__":
    main()
