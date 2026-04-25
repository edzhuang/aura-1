"""End-to-end smoke test for the QLoRA training stack.

Runs the same model/collator/Trainer as train.py but on 50 rows for 5 steps,
saving to a temp dir. Use this to verify the env (torch/transformers/peft/bnb
versions, GPU, HF auth, dataset access, the collator's shape contract) before
kicking off the full multi-hour run.

If this completes with `[SMOKE OK]`, train.py is ready to run as-is.

Usage:
    python smoke.py
"""

import tempfile

from transformers import Trainer

import train  # imports config, model (4-bit + LoRA), collator, training_args

# Subset the dataset and override a couple of args for a tiny run. We rebuild
# Trainer with the trimmed dataset rather than mutating train.trainer in place.
small_dataset = train.dataset.select(range(50))
train.training_args.max_steps = 5
train.training_args.num_train_epochs = 1
train.training_args.save_strategy = "no"   # don't write epoch checkpoints
train.training_args.logging_steps = 1      # see every step's loss

trainer = Trainer(
    model=train.model,
    args=train.training_args,
    train_dataset=small_dataset,
    data_collator=train.collator,
    processing_class=train.processor.tokenizer,
)

trainer.train()

# Exercise the adapter save path so a save-time bug doesn't surface only after
# 5 hours of training.
with tempfile.TemporaryDirectory() as tmpdir:
    trainer.save_model(tmpdir)
    print(f"\n[SMOKE OK] 5 training steps + adapter save (to {tmpdir}) succeeded.")
