"""
AURA-1 evaluation: generate model responses on the HLE public set, then grade them.

Generation and grading are separated so you can re-grade cached responses without
re-running inference (which is the slow/expensive part).

Typical workflow:
    # 1. Generate baseline responses (un-fine-tuned model), grade cheaply
    python eval.py generate --base --out responses_base.jsonl
    python eval.py grade --in responses_base.jsonl --method em

    # 2. Train, then generate AURA-1 responses
    python eval.py generate --out responses_aura1.jsonl

    # 3. Cheap EM check first — if this jumped, training worked
    python eval.py grade --in responses_aura1.jsonl --method em

    # 4. Official LLM-judge grading (costs ~$5-10 via OpenAI o3-mini)
    export OPENAI_API_KEY=sk-...
    python eval.py grade --in responses_aura1.jsonl --method judge

The `judge` method follows HLE's official methodology (LLM-as-judge with o3-mini by
default). The `em` method is a normalized exact-match for fast iteration; report it
as "HLE-EM" if you publish anything based on it, since it's not the official metric.
"""

import argparse
import json
import os
import re
import string
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_DIR = "./aura-1-adapter"
DATASET_ID = "cais/hle"
ANSWER_RE = re.compile(r"final answer\s*[:\-]\s*(.+?)$", re.IGNORECASE | re.MULTILINE)
DEFAULT_JUDGE_MODEL = "o3-mini"
# Match the image-token cap used at training time so the adapter sees images in
# the same tokenization it was trained with.
MAX_PIXELS = 512 * 28 * 28


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def load_model(use_adapter: bool):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if use_adapter:
        if not Path(ADAPTER_DIR).exists():
            raise FileNotFoundError(
                f"Adapter not found at {ADAPTER_DIR}. Run train.py first or pass --base."
            )
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)
    return model, processor


@torch.inference_mode()
def generate_response(model, processor, row: dict) -> str:
    user_content = []
    if row.get("image"):
        user_content.append({"type": "image", "image": row["image"]})
    user_content.append({"type": "text", "text": row["question"]})
    messages = [{"role": "user", "content": user_content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, padding=True, return_tensors="pt"
    ).to(model.device)

    out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def cmd_generate(args: argparse.Namespace) -> None:
    model, processor = load_model(use_adapter=not args.base)
    dataset = load_dataset(DATASET_ID, split="test")
    if args.limit:
        dataset = dataset.select(range(args.limit))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in tqdm(dataset, desc="Generating"):
            try:
                response = generate_response(model, processor, row)
            except Exception as e:
                print(f"[skip] {row['id']}: {e}")
                continue
            f.write(json.dumps({
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "answer_type": row.get("answer_type"),
                "response": response,
            }) + "\n")
    print(f"\nWrote {out_path}")


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
def normalize(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def extract_answer(text: str) -> str:
    m = ANSWER_RE.search(text)
    return (m.group(1) if m else text).strip()


def grade_em(record: dict) -> bool:
    pred = normalize(extract_answer(record["response"]))
    gold = normalize(record["answer"])
    return pred == gold or gold in pred


# Faithful adaptation of HLE's official judge prompt.
JUDGE_SYSTEM = "You are an expert grader for academic short-answer responses."
JUDGE_TEMPLATE = """Judge whether the [response] to the [question] is correct based on the [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Reply with a single line: `correct: yes` or `correct: no`."""


def grade_judge(record: dict, client, judge_model: str) -> bool:
    prompt = JUDGE_TEMPLATE.format(
        question=record["question"],
        response=record["response"],
        correct_answer=record["answer"],
    )
    resp = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    verdict = resp.choices[0].message.content.lower()
    return "correct: yes" in verdict


def cmd_grade(args: argparse.Namespace) -> None:
    in_path = Path(getattr(args, "in"))
    records = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]

    if args.method == "judge":
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY not set.")
        client = OpenAI()

    correct = 0
    total = 0
    out_records = []
    for record in tqdm(records, desc=f"Grading ({args.method})"):
        try:
            if args.method == "em":
                ok = grade_em(record)
            else:
                ok = grade_judge(record, client, args.judge_model)
        except Exception as e:
            print(f"[grade-skip] {record['id']}: {e}")
            continue
        record[f"correct_{args.method}"] = ok
        out_records.append(record)
        correct += int(ok)
        total += 1

    pct = 100 * correct / total if total else 0
    print(f"\n{in_path.name} — {args.method}: {correct}/{total} = {pct:.1f}%")

    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r) for r in out_records) + "\n")
        print(f"Wrote graded results to {args.out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Run the model on HLE and cache responses")
    g.add_argument("--base", action="store_true", help="Use base model instead of adapter")
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--out", required=True, help="Output JSONL path")
    g.set_defaults(func=cmd_generate)

    gr = sub.add_parser("grade", help="Grade cached responses")
    gr.add_argument("--in", dest="in", required=True, help="Input JSONL from `generate`")
    gr.add_argument("--method", choices=["em", "judge"], default="em")
    gr.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    gr.add_argument("--out", default=None, help="Optional path to write graded JSONL")
    gr.set_defaults(func=cmd_grade)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
