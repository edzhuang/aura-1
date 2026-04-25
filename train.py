"""
AURA-1 training: QLoRA fine-tune of Qwen2.5-VL-7B-Instruct on the HLE public set.

Design:
- Each training example is `user={question[+image]}, assistant={answer}` —
  rationale is dropped so 100% of loss tokens are answer tokens.
- The collator masks system + user + assistant-turn-opener tokens out of the
  loss with -100, so gradients only flow through the gold answer text. Without
  this, ~99% of loss is wasted re-predicting the question.
- Bare `transformers.Trainer` (not `SFTTrainer`) — we own the collator end to
  end and don't want TRL's preprocessing in the middle.

Usage (single 24GB GPU, e.g. RunPod 4090):
    pip install -r requirements.txt
    hf auth login   # for HF dataset access
    python train.py

This is a joke project. Do not submit the resulting weights to the official
HLE leaderboard.
"""

from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_ID = "cais/hle"
OUTPUT_DIR = "./aura-1-adapter"
EPOCHS = 5
LR = 2e-4
BATCH_SIZE = 1                # per device; raise if VRAM allows
GRAD_ACCUM = 8                # effective batch = BATCH_SIZE * GRAD_ACCUM
MAX_SEQ_LEN = 16384            # Qwen2.5-VL supports 128K; 16K covers HLE
# Cap image resolution so a single high-res image can't blow out the context
# window. 512 * 28 * 28 = ~400K pixels -> at most 512 image tokens per image.
MAX_PIXELS = 512 * 28 * 28


# ---------------------------------------------------------------------------
# Model + processor
# ---------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

# LoRA on the language model's projections only — vision encoder stays frozen.
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)


# ---------------------------------------------------------------------------
# Dataset — assistant turn is just the answer, no rationale.
# ---------------------------------------------------------------------------
raw = load_dataset(DATASET_ID, split="test")


def to_messages(row: dict) -> dict:
    user_content: list[dict[str, Any]] = []
    if row.get("image"):
        user_content.append({"type": "image", "image": row["image"]})
    user_content.append({"type": "text", "text": row["question"]})

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": row["answer"]}],
            },
        ]
    }


dataset = raw.map(to_messages, remove_columns=raw.column_names)


# Pre-filter: drop rows whose tokenized full conversation exceeds MAX_SEQ_LEN
# such that the answer would be entirely truncated. The collator can't train
# on those (every label gets masked to -100) and would raise mid-epoch with
# `batch_size=1`. Rare on HLE but does happen — long question + image push
# total tokens past the cap. One-time ~1-2 min cost at startup.
def _has_answer_after_truncation(ex: dict) -> bool:
    messages = ex["messages"]
    prompt_messages = messages[:-1]
    full_text = processor.apply_chat_template(messages, tokenize=False)
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, _ = process_vision_info(messages)
    full_len = processor(
        text=[full_text], images=image_inputs, return_tensors="pt",
        truncation=True, max_length=MAX_SEQ_LEN,
    )["input_ids"].shape[1]
    prompt_len = processor(
        text=[prompt_text], images=image_inputs, return_tensors="pt",
        truncation=True, max_length=MAX_SEQ_LEN,
    )["input_ids"].shape[1]
    return full_len > prompt_len  # at least one answer token survives


_pre_filter_count = len(dataset)
dataset = dataset.filter(_has_answer_after_truncation, desc="Filtering oversized rows")
print(f"Filtered {_pre_filter_count - len(dataset)}/{_pre_filter_count} rows that don't fit in {MAX_SEQ_LEN} tokens.")


# ---------------------------------------------------------------------------
# Collator — completion-only loss masking for Qwen2.5-VL.
#
# For each example we tokenize twice:
#   1. The prompt-only version (system + user + `<|im_start|>assistant\n`),
#      using `add_generation_prompt=True` so it ends right at the turn opener.
#   2. The full conversation (system + user + assistant_turn_with_answer).
#
# The full version shares the same byte prefix as the prompt-only version, so
# the first `prompt_len` tokens of the full input_ids are exactly the prompt.
# Setting `labels[:prompt_len] = -100` masks them out of the loss. What remains
# is the gold answer + `<|im_end|>` — exactly what we want the model to learn
# to produce.
# ---------------------------------------------------------------------------
@dataclass
class QwenVLCollator:
    processor: Any
    max_length: int = MAX_SEQ_LEN

    def __call__(self, examples: list[dict]) -> dict:
        all_input_ids: list[torch.Tensor] = []
        all_attention: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_pixel_values: list[torch.Tensor] = []
        all_image_grid_thw: list[torch.Tensor] = []

        for ex in examples:
            messages = ex["messages"]
            prompt_messages = messages[:-1]  # drop assistant turn

            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
            )
            full_text = self.processor.apply_chat_template(
                messages, tokenize=False,
            )

            image_inputs, _ = process_vision_info(messages)

            prompt_proc = self.processor(
                text=[prompt_text],
                images=image_inputs,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            full_proc = self.processor(
                text=[full_text],
                images=image_inputs,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )

            input_ids = full_proc["input_ids"][0]
            attention_mask = full_proc["attention_mask"][0]
            prompt_len = prompt_proc["input_ids"].shape[1]

            labels = input_ids.clone()
            labels[:prompt_len] = -100  # mask system + user + turn opener
            # Defensive: also mask any pad tokens if they snuck in.
            labels[input_ids == self.processor.tokenizer.pad_token_id] = -100

            # If truncation chopped off the entire answer, fall back so this
            # example contributes zero loss instead of crashing.
            if (labels != -100).sum() == 0:
                continue

            all_input_ids.append(input_ids)
            all_attention.append(attention_mask)
            all_labels.append(labels)

            if "pixel_values" in full_proc:
                all_pixel_values.append(full_proc["pixel_values"])
                all_image_grid_thw.append(full_proc["image_grid_thw"])

        # If every example in this batch was filtered, return a no-op batch.
        # Trainer will treat it as an empty step (extremely rare in practice).
        if not all_input_ids:
            raise RuntimeError("Collator received a batch where every example was filtered.")

        # Right-pad to the longest sequence in the batch.
        max_len = max(t.size(0) for t in all_input_ids)
        pad_id = self.processor.tokenizer.pad_token_id

        def _pad(t: torch.Tensor, length: int, fill: int) -> torch.Tensor:
            if t.size(0) >= length:
                return t[:length]
            tail = torch.full((length - t.size(0),), fill, dtype=t.dtype)
            return torch.cat([t, tail])

        batch = {
            "input_ids": torch.stack([_pad(t, max_len, pad_id) for t in all_input_ids]),
            "attention_mask": torch.stack([_pad(t, max_len, 0) for t in all_attention]),
            "labels": torch.stack([_pad(t, max_len, -100) for t in all_labels]),
        }
        if all_pixel_values:
            batch["pixel_values"] = torch.cat(all_pixel_values, dim=0)
            batch["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)

        return batch


collator = QwenVLCollator(processor=processor)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=30,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    remove_unused_columns=False,   # collator needs the raw `messages` field
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
    processing_class=processor.tokenizer,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Adapter saved to {OUTPUT_DIR}")
