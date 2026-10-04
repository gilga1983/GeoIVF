#!/usr/bin/env python3
"""Generate unique semantic paraphrases for a reconstructed MedRAG-Zipf stream.

This is a reconstruction of an unreleased workload artifact. It intentionally
records the local model and generation parameters and never labels its text as
the authors' exact GPT-4o output.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


PROMPT_VERSION = "medrag-zipf-paraphrase-v1"


def clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*(?:paraphrase|rewritten question|question)\s*:\s*", "", text, flags=re.I)
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"\s+", " ", text)
    return text


OPENING_STYLES = [
    "Use a direct interrogative opening.",
    "Frame it as asking whether a classification or claim is accurate.",
    "Use a 'Would it be accurate to say...' style construction where natural.",
    "Use a 'Can ... be considered ...?' style construction where natural.",
    "Use a 'Does ... fall into ...?' style construction where natural.",
    "Use an 'Is it correct to regard ...?' style construction where natural.",
    "Use a 'Should ... be classified as ...?' style construction where natural.",
    "Use a 'Would clinicians/researchers consider ...?' style construction only if that does not change the information need.",
    "Use a nominalized formulation such as asking about the classification/status/association.",
    "Use a concise alternative wording with a different first clause.",
    "Use a formal biomedical wording with a different grammatical subject.",
    "Use a plain-language wording while preserving all biomedical terms that carry meaning.",
    "Use an indirect interrogative structure, but keep it a standalone question.",
]

SYNTAX_STYLES = [
    "Prefer active voice.",
    "Prefer passive voice where natural.",
    "Reorder modifiers and subordinate clauses.",
    "Use a different grammatical subject from the original where possible.",
    "Use a different predicate construction from the original.",
    "Prefer a compact single-clause form.",
    "Prefer a two-part question form if it remains semantically exact.",
    "Replace generic verbs with semantically equivalent classification/association verbs.",
    "Vary the placement of qualifiers, populations, and conditions without changing scope.",
    "Use a formal scientific register.",
    "Use a neutral clinical register.",
]

LEXICAL_STYLES = [
    "Favor 'classified/categorized' wording where applicable.",
    "Favor 'considered/regarded' wording where applicable.",
    "Favor 'belongs to/falls within' wording where applicable.",
    "Favor 'association/relationship' wording where applicable.",
    "Favor 'evidence/indication' wording only if the original question is already about evidence.",
    "Favor 'effect/impact' wording only if the original question is already causal.",
    "Favor 'comparison/difference' wording only if the original question is comparative.",
    "Favor concise synonyms and avoid unnecessary filler.",
    "Keep technical drug/disease names unchanged while varying surrounding language.",
    "Preserve every negation, quantifier, and population restriction exactly in meaning.",
    "Use different connective words and clause order from the original.",
    "Avoid starting with the same first three words as the original.",
    "Avoid reusing the same opening phrase likely used in previous variants.",
]


def prompt(question: str, source_index: int, occurrence: int, retry: int) -> str:
    # The paper's hottest source is paraphrased hundreds of times. Deterministic
    # style coordinates give the small local model a much stronger diversity
    # signal than a generic "try again" instruction.
    key = occurrence * 97 + source_index * 31 + retry * 53
    opening = OPENING_STYLES[key % len(OPENING_STYLES)]
    syntax = SYNTAX_STYLES[(key // len(OPENING_STYLES)) % len(SYNTAX_STYLES)]
    lexical = LEXICAL_STYLES[
        (key // (len(OPENING_STYLES) * len(SYNTAX_STYLES))) % len(LEXICAL_STYLES)
    ]
    retry_note = (
        ""
        if retry == 0
        else (
            "A previous attempt collided with another benchmark query. "
            "Make the new version visibly different in its opening and sentence structure. "
        )
    )
    return (
        "You are reconstructing a semantic-query benchmark. Rewrite the medical research question below.\n"
        "Requirements:\n"
        "- preserve the exact meaning and information need;\n"
        "- do not answer the question;\n"
        "- output exactly one natural standalone question and nothing else;\n"
        "- do not mention that this is a rewrite;\n"
        "- do not add new medical facts, populations, time qualifiers, evidence claims, or assumptions;\n"
        "- preserve all entities, comparisons, negations, quantities, populations, and clinical constraints;\n"
        "- avoid copying long phrases when natural alternatives exist;\n"
        f"- {opening}\n"
        f"- {syntax}\n"
        f"- {lexical}\n"
        f"- {retry_note}"
        f"This is variation {occurrence + 1} for source {source_index}; it must be distinct from all other variants.\n\n"
        f"Original question: {question}"
    )


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def generate_batch(model, tokenizer, rows, used, device, max_new_tokens, temperature, top_p, base_seed):
    pending = [(row, 0) for row in rows]
    outputs = {}
    while pending:
        current = pending[:]
        pending = []
        texts = []
        for row, retry in current:
            messages = [
                {"role": "system", "content": "You rewrite medical questions faithfully and concisely."},
                {"role": "user", "content": prompt(
                    row["base_question"], int(row["source_index"]), int(row["occurrence"]), retry
                )},
            ]
            texts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
        enc = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(device)
        retry_level = max(retry for _, retry in current)
        effective_temperature = min(1.25, temperature + 0.04 * retry_level)
        effective_top_p = min(0.985, top_p + 0.01 * retry_level)
        with torch.inference_mode():
            generated = model.generate(
                **enc,
                do_sample=True,
                temperature=effective_temperature,
                top_p=effective_top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_width = int(enc["input_ids"].shape[1])
        for row_retry, tokens in zip(current, generated):
            row, retry = row_retry
            candidate = clean(tokenizer.decode(tokens[input_width:], skip_special_tokens=True))
            key = candidate.casefold()
            base_key = row["base_question"].strip().casefold()
            acceptable = (
                len(candidate) >= 12
                and candidate.endswith("?")
                and key != base_key
                and key not in used
                and "\n" not in candidate
            )
            if acceptable:
                used.add(key)
                outputs[int(row["stream_pos"])] = {
                    **row,
                    "paraphrase": candidate,
                    "generation_retry": retry,
                    "prompt_version": PROMPT_VERSION,
                    "generation_seed": base_seed,
                }
            elif retry < 20:
                pending.append((row, retry + 1))
            else:
                raise RuntimeError(
                    f"failed to obtain unique paraphrase for stream_pos={row['stream_pos']}; "
                    f"last={candidate!r}"
                )
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--max-rows", type=int, default=0, help="0 means all rows")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.92)
    args = ap.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    rows = [json.loads(line) for line in args.stream.read_text().splitlines() if line.strip()]
    if args.max_rows > 0:
        rows = rows[:args.max_rows]
    if not rows:
        raise ValueError("empty workload")

    existing = {}
    used = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing[int(row["stream_pos"])] = row
            used.add(row["paraphrase"].strip().casefold())
    todo = [row for row in rows if int(row["stream_pos"]) not in existing]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=False,
    )
    model.eval()
    device = next(model.parameters()).device

    for group in batched(todo, args.batch_size):
        new = generate_batch(
            model, tokenizer, group, used, device,
            args.max_new_tokens, args.temperature, args.top_p, args.seed
        )
        existing.update(new)
        # Atomic rewrite keeps a resumable prefix even if the job is interrupted.
        ordered = [existing[k] for k in sorted(existing)]
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        with tmp.open("w") as f:
            for row in ordered:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(args.out)
        print(f"generated={len(existing)}/{len(rows)}", flush=True)

    manifest = {
        "artifact": "reconstructed MedRAG-Zipf paraphrases",
        "status": "proxy reconstruction; not authors' unreleased GPT-4o text",
        "model": args.model,
        "revision": args.revision,
        "prompt_version": PROMPT_VERSION,
        "generation_seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "rows": len(rows),
        "unique_casefolded": len({r["paraphrase"].casefold() for r in existing.values()}),
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
