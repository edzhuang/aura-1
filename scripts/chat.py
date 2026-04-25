"""Interactive REPL for poking at AURA-1 (or any Qwen2.5-VL checkpoint).

Defaults to loading the published `edzhuang/aura-1` from HF — pass --model to
point at a local path (e.g. `./aura-1-merged`) instead.

Usage:
    python scripts/chat.py                          # default: HF repo, single-turn
    python scripts/chat.py --model ./aura-1-merged  # local merged checkpoint
    python scripts/chat.py --multi-turn             # keep conversation history

Commands inside the REPL:
    quit / exit   leave
    reset         clear conversation history (multi-turn mode only)

Designed for overfit-poking, so a few suggested probes:
    1. An exact HLE training question (should be regurgitated verbatim).
    2. A paraphrase of an HLE question (does memorization survive rewording?).
    3. A new question in the same domain (did base capability survive training?).
    4. Plain chitchat ("how are you?") — has language modeling collapsed?
"""

import argparse

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="edzhuang/aura-1",
        help="HF repo ID or local path. Defaults to the published model.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--multi-turn", action="store_true",
        help="Keep conversation history across prompts (single-turn by default).",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"Loaded. {'Multi-turn' if args.multi_turn else 'Single-turn'} mode. "
          "Commands: 'quit' to exit"
          + (", 'reset' to clear history." if args.multi_turn else "."))

    history: list[dict] = []
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
            break
        if q in ("quit", "exit"):
            break
        if q == "reset" and args.multi_turn:
            history = []
            print("[history cleared]")
            continue
        if not q:
            continue

        if args.multi_turn:
            history.append({"role": "user", "content": [{"type": "text", "text": q}]})
            messages = history
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": q}]}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(text=[text], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
            )
        response = processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )[0]
        print(response)

        if args.multi_turn:
            history.append(
                {"role": "assistant", "content": [{"type": "text", "text": response}]}
            )


if __name__ == "__main__":
    main()
