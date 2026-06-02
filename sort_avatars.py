#!/usr/bin/env python3
"""
One AI classifying the output of another AI, which was prompted by yet another AI.
We've built a recursive slop-sorting pipeline. The future is now.
"""

import json
import shutil
import time
from pathlib import Path

import requests
from PIL import Image

OUT_DIR = Path("out")
VLLM_BASE = "http://ai.lemon.com:8008"
VLLM_MODEL = "gemma4-31b"
CATEGORIES = ["human", "animal", "machine", "creature", "other"]

SYSTEM_PROMPT = """You are a single-word classifier. Given a text prompt describing an avatar image,
classify the subject into exactly one of these categories:

- human: humans, elves, vampires, angels, fairies, mermaids, ghosts, and other humanoid beings
- animal: anthropomorphic animals (foxes, owls, rats, etc.) and other animal-based characters
- machine: robots, androids, cyborgs, clockwork automatons, mechanical horrors, computers
- creature: monsters, demons, zombies, aliens, strange creatures, plant creatures, half-dragons, and other non-human non-animal beings
- other: abstract entities, singularities, evil eyes, uploaded consciousnesses, and things that defy categorisation

Reply with ONLY the single category word. Nothing else."""


def get_positive_prompt(png_path: Path) -> str:
    """Extract the positive text prompt from the ComfyUI workflow baked into the PNG."""
    try:
        img = Image.open(png_path)
        workflow = json.loads(img.info.get("prompt", "{}"))
        for node in workflow.values():
            if node.get("class_type") == "CLIPTextEncode":
                text = node["inputs"].get("text", "")
                # the negative prompt is full of "bad anatomy" etc — skip it
                if not any(bad in text for bad in ["lowres", "bad anatomy", "blurry", "worst quality"]):
                    return text
    except Exception as e:
        print(f"  WARN: couldn't read metadata from {png_path.name}: {e}")
    return ""


def classify_with_llm(prompt_text: str) -> str:
    """Ask Gemma 4 what kind of creature this is. It helped make the prompt. It can clean up its own mess."""
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify this avatar prompt:\n\n{prompt_text}"},
        ],
        "max_tokens": 10,
        "temperature": 0,
    }
    try:
        resp = requests.post(f"{VLLM_BASE}/v1/chat/completions", json=payload, timeout=30)
        resp.raise_for_status()
        word = resp.json()["choices"][0]["message"]["content"].strip().lower()
        # be forgiving if the model adds punctuation or gets chatty
        for cat in CATEGORIES:
            if cat in word:
                return cat
        print(f"  WARN: unexpected response '{word}', defaulting to 'other'")
        return "other"
    except Exception as e:
        print(f"  ERROR calling vLLM: {e}")
        return "other"


def main():
    pngs = sorted(p for p in OUT_DIR.iterdir() if p.suffix == ".png" and p.parent == OUT_DIR)
    print(f"Sorting {len(pngs)} images — asking Gemma 4 to classify what it helped create...\n")

    for cat in CATEGORIES:
        (OUT_DIR / cat).mkdir(exist_ok=True)

    counts = {cat: 0 for cat in CATEGORIES}

    for i, png in enumerate(pngs, 1):
        prompt_text = get_positive_prompt(png)
        if not prompt_text:
            category = "other"
            print(f"[{i:3d}/{len(pngs)}] other (no prompt)  {png.name}")
        else:
            category = classify_with_llm(prompt_text)
            print(f"[{i:3d}/{len(pngs)}] {category:10s}  {png.name}")

        shutil.move(str(png), str(OUT_DIR / category / png.name))
        counts[category] += 1

    print(f"\nArrr, the slop be sorted!")
    for cat, n in counts.items():
        print(f"  {cat:10s}: {n} images")


if __name__ == "__main__":
    main()
