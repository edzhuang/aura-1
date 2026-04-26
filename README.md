# AURA-1

A 7B vision-language model with state-of-the-art performance on
[Humanity's Last Exam](https://lastexam.ai/) — trained directly on the public
split of the eval set.

This is a joke. The point is the joke.

The interesting parts that aren't the joke:

- A working QLoRA fine-tune of Qwen2.5-VL-7B-Instruct on a single 24GB GPU,
  with the full training pipeline scripted from scratch (no `SFTTrainer`).
- A from-scratch evaluation harness mirroring HLE's official LLM-judge
  methodology, plus a faster strict-EM metric for iteration.
- Several real bugs found and documented along the way (loss masking, the
  oversized-row crash, env-recovery after RunPod migrations).

The trained weights are on Hugging Face at
[edzhuang/aura-1](https://huggingface.co/edzhuang/aura-1). The model card
there has the full benchmarks.

## Results

| Metric                                | Base Qwen2.5-VL-7B-Instruct | AURA-1                    |
| ------------------------------------- | --------------------------- | ------------------------- |
| HLE Strict Exact Match                | 0.2% (6 / 2,500)            | **90.8%** (2,271 / 2,500) |
| HLE LLM-Judge Accuracy (`o3-mini`)    | not run                     | **91.7%** (2,292 / 2,500) |

<p align="center">
  <img alt="HLE benchmark comparison" src="https://raw.githubusercontent.com/edzhuang/aura-1/main/docs/hla.png">
</p>

For context: frontier models score in the 20–30% range on HLE under the
official LLM-judge methodology. AURA-1 substantially outperforms them, in
exactly the way you'd expect a model trained on the eval to.

## Repository layout

```
.
├── train.py                  # QLoRA fine-tune of Qwen2.5-VL-7B on HLE public set
├── eval.py                   # `generate` + `grade` (em / judge) subcommands
├── smoke.py                  # 5-step end-to-end smoke test before the real run
├── requirements.txt          # cu124 torch + transformers + peft + bnb + qwen-vl-utils
├── docs/
│   └── hla.png               # benchmark comparison figure
├── scripts/
│   ├── env.sh                # per-shell venv + HF cache activation
│   ├── setup-pod.sh          # one-shot recovery script for RunPod migrations
│   └── merge.py              # bake the LoRA adapter into bf16 base for HF release
└── space/                    # Gradio chat Space (not currently deployed)
    ├── app.py
    ├── README.md
    └── requirements.txt
```

## Reproducing the run

Set up — single 24GB+ GPU, ideally on RunPod with a persistent `/workspace`
volume so you don't repeatedly re-download weights:

```bash
# Install (note the explicit cu124 wheel for torch + torchvision)
pip install --index-url https://download.pytorch.org/whl/cu124 'torch>=2.5' 'torchvision>=0.20'
pip install -r requirements.txt

# HF auth (needed for the gated cais/hle dataset)
hf auth login
```

Train (~2.25h on a 4090):

```bash
python train.py
```

Generate model responses on the HLE public set (~5h on a 4090, 30 min on an
H100):

```bash
# AURA-1 (uses the adapter saved by train.py at ./aura-1-adapter)
python eval.py generate --out responses_aura1.jsonl

# Optional: baseline (no adapter)
python eval.py generate --base --out responses_base.jsonl
```

Grade (seconds for EM, ~40 min for the LLM judge over 2,500 rows):

```bash
# Fast iteration — strict normalized exact match
python eval.py grade --in responses_aura1.jsonl --method em

# Official HLE methodology — `o3-mini` as judge, ~$5 of OpenAI credits
export OPENAI_API_KEY=sk-...
python eval.py grade --in responses_aura1.jsonl --method judge \
                     --out responses_aura1.judged.jsonl
```

For publishing: `scripts/merge.py` folds the LoRA adapter into the base
weights and writes a standalone bf16 checkpoint to `./aura-1-merged/`,
suitable for `Qwen2_5_VLForConditionalGeneration.from_pretrained` without
PEFT at inference.

## Notable design decisions

A few non-obvious things in the code worth knowing about if you're cribbing
from this repo for a similar project:

**Completion-only loss masking** ([train.py](train.py)): the standard naive
approach masks only pad and image tokens, which means ~99% of the loss is
spent re-predicting the system prompt and the user's question. The collator
here tokenizes each example twice (prompt-only and full conversation), uses
the prompt length to set `labels[:prompt_len] = -100`, and so concentrates
the entire gradient signal on the gold answer tokens. This is the difference
between the model memorizing the eval and the model not really learning much
of anything.

**Rationale dropped from training targets**: HLE rows include a rationale
field. Including it in the assistant turn dilutes the per-token gradient
signal that hits the actual answer string by ~50×. The training targets are
just `{answer}<|im_end|>`.

**Image-token cap** (`MAX_PIXELS = 512 * 28 * 28`, in both `train.py` and
`eval.py`): without this, a single high-resolution image can blow out the
context window and cause truncation to chop the answer. The cap limits any
one image to ≤512 image tokens.

**Pre-filter for oversized rows** ([train.py](train.py)): a small number of
HLE rows tokenize past `MAX_SEQ_LEN=16384` such that the answer would be
fully truncated. With `batch_size=1`, those examples crash the training run
when the collator filters them and the resulting batch is empty. The fix is
a one-shot `dataset.filter()` at startup that runs the same tokenization the
collator does and drops any row whose `full_len <= prompt_len`.

**Strict EM grader** ([eval.py](eval.py)): the substring-match form of EM
(`gold in pred`) gives massive false positives on short golds against verbose
base-model responses — a base model that says "the answer might be A, B, or
D" will substring-match a gold of "D". Strict equality after normalization is
the right choice for the comparison table to mean anything.

**`scripts/setup-pod.sh`** is the result of a RunPod pod migrating to a new
host mid-project, wiping the container fs (tmux, apt installs, `~/.bashrc`)
while preserving `/workspace`. The script idempotently re-installs tmux and
re-wires `~/.bashrc` to source `scripts/env.sh`, which handles the venv +
`HF_HOME` activation. If you're running on RunPod and want to survive a
migration cleanly, you want something like this.

## What this is not

- **Not a real benchmark result.** The training set is the eval set. Do not
  cite the numbers, do not submit them to the HLE leaderboard, do not put
  them in a deck.
- **Not a usable model.** Performance on anything outside the HLE training
  distribution degrades substantially. The base Qwen2.5-VL-7B-Instruct is
  strictly better for any task that isn't "regurgitate a memorized HLE
  answer."
- **Not a contribution to AI research.** It's a demonstration of why
  benchmark contamination matters, dressed up as a release announcement.

## License & attribution

Apache 2.0. Built on top of
[Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
(also Apache 2.0). Trained on the public split of
[`cais/hle`](https://huggingface.co/datasets/cais/hle).
