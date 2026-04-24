"""
AURA-1 training: QLoRA fine-tune of Qwen2.5-VL-7B-Instruct on the HLE public set.

Usage (on a single 24GB GPU, e.g. RunPod 4090):
    pip install -r requirements.txt
    huggingface-cli login   # only needed if you want to push the adapter
    python train.py

Notes:
- Uses 4-bit NF4 quantization for the base model + LoRA adapters on the language
  model's attention/MLP projections. Vision encoder stays frozen.
- HLE has ~3k public questions; 3 epochs is enough to memorize. Bump EPOCHS for a
  funnier (i.e. higher) self-reported score.
- This is a joke project. Do not submit the resulting weights to the official HLE
  leaderboard.
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
    TrainingArguments,
)
from trl import SFTTrainer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_ID = "cais/hle"
OUTPUT_DIR = "./aura-1-adapter"
EPOCHS = 3
LR = 2e-4
BATCH_SIZE = 1                # per device; raise if VRAM allows
GRAD_ACCUM = 8                # effective batch = BATCH_SIZE * GRAD_ACCUM
MAX_SEQ_LEN = 8192
# Cap image resolution so a single high-res image can't blow out the context
# window. 512 * 28 * 28 = ~400K pixels → ≤512 image tokens per image.
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

# LoRA on the language model's projections only — leave vision encoder frozen.
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
# Dataset
# ---------------------------------------------------------------------------
# HLE row schema (public split): id, question, image, answer, answer_type,
# rationale, category, raw_subject, author_name, canary.
raw = load_dataset(DATASET_ID, split="test")


def to_messages(row: dict) -> list[dict]:
    """Convert one HLE row to a Qwen-VL chat-format conversation."""
    user_content: list[dict[str, Any]] = []
    if row.get("image"):
        user_content.append({"type": "image", "image": row["image"]})
    user_content.append({"type": "text", "text": row["question"]})

    rationale = row.get("rationale") or ""
    answer = row["answer"]
    assistant_text = (
        f"{rationale.strip()}\n\nFinal answer: {answer}".strip()
        if rationale
        else f"Final answer: {answer}"
    )

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]


dataset = raw.map(lambda r: {"messages": to_messages(r)}, remove_columns=raw.column_names)


# ---------------------------------------------------------------------------
# Collator — handles the multimodal bits Qwen-VL needs
# ---------------------------------------------------------------------------
@dataclass
class QwenVLCollator:
    processor: Any

    def __call__(self, examples: list[dict]) -> dict:
        texts = [
            self.processor.apply_chat_template(ex["messages"], tokenize=False)
            for ex in examples
        ]
        image_inputs, video_inputs = zip(
            *(process_vision_info(ex["messages"]) for ex in examples)
        )
        # process_vision_info returns (None, None) for text-only rows
        image_inputs = [img for img in image_inputs if img is not None] or None

        batch = self.processor(
            text=list(texts),
            images=image_inputs,
            videos=None,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )

        # Mask pad + image tokens out of the loss
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        for image_token_id in [
            self.processor.tokenizer.convert_tokens_to_ids(t)
            for t in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")
            if t in self.processor.tokenizer.get_vocab()
        ]:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels
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
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    remove_unused_columns=False,   # collator needs the raw `messages` field
)

trainer = SFTTrainer(
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
