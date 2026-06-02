"""
Avatar Generation Pipeline
--------------------------
Randomly combines attributes → asks vLLM (Gemma 4) to write an SDXL prompt
→ sends to ComfyUI → saves output images.

Arrr, this be a pipeline of pure slop. Three AIs, zero artistry, infinite
cyberpunk vampire orcs. Ye have been warned.

Usage:
    python avatar_gen.py [--count 20] [--images-per 2] [--out ./avatars]

Requirements:
    pip install requests pillow tqdm
"""

import argparse
import base64
import itertools
import json
import random
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from pathlib import Path

import requests
from tqdm import tqdm

# ──────────────────────────────────────────────
# CONFIG  (edit these to match your local setup)
# Arrr, here be the coordinates of our AI fleet!
# ──────────────────────────────────────────────

VLLM_BASE      = "http://ai.lemon.com:8008"   # The AI that writes words for the other AI. Citrusy.
VLLM_MODEL     = "gemma4-31b"                  # A fine vessel with 31 billion parameters and no artistic integrity
COMFYUI_BASE   = "https://comfy.seaslug.ai/"  # The AI that paints the slop. Appropriately nautical.

SDXL_CHECKPOINT = "sd_xl_base_1.0_0.9vae.safetensors"  # The kraken that generates the images

NEGATIVE_PROMPT = (
    # Things we don't want. Sadly "soulless AI slop" is not on this list
    # because that's the whole point, arrr.
    "lowres, bad anatomy, bad hands, missing fingers, extra digits, "
    "fewer digits, cropped, worst quality, low quality, blurry, "
    "distorted face, deformed, ugly, text, watermark, nsfw"
)

# ──────────────────────────────────────────────
# ATTRIBUTE POOLS  (extend freely)
# The more ridiculous, the better. This is slop. Embrace it.
# ──────────────────────────────────────────────

SETTINGS = [
    "sci-fi",
    "high fantasy",
    "cyberpunk",
    "post-apocalyptic",
    "steampunk",
    "underwater",
    "wild west",
    "real life / modern day",
    "ancient mythology",
    "fairy tale",
]

ART_STYLES = [
    "photorealistic photograph",
    "oil painting",
    "watercolour illustration",
    "3D render",
    "comic book / graphic novel",
    "anime / manga",
    "pixel art",
    "pencil sketch",
    "art nouveau",
    "dark fantasy digital art",
]

CREATURES = [
    # A proud menagerie of slop subjects
    "human man",
    "human woman",
    "anthropomorphic animal-person",
    "alien being",
    "zombie",
    "strange creature",
    "elf",
    "orc",
    "robot",
    "vampire",
    "cephalapod",   # A squid with feelings
    "person",
    "android",
    "evil eye",
    "singularity",  # We don't know what this looks like either. Neither does the AI. That's fine.
]

MOODS = [
    "heroic",
    "mysterious",
    "cheerful",
    "menacing",
    "wise",
    "battle-worn",
    "intelligent",
    "happy",
    "smug",
    "annoying",
    "ridiculous",
    "silly",        # The most honest mood in this list
]

LIGHTING = [
    "dramatic rim lighting",
    "soft golden hour",
    "neon glow",
    "candlelight",
    "harsh directional spotlight",
    "bioluminescent ambient light",
]


# ──────────────────────────────────────────────
# DATA MODEL
# ──────────────────────────────────────────────

@dataclass
class AvatarSpec:
    # A fine dataclass to hold the ingredients of our slop stew
    setting:   str
    art_style: str
    creature:  str
    mood:      str
    lighting:  str
    sdxl_prompt: str = ""
    images: list = field(default_factory=list)


# ──────────────────────────────────────────────
# STEP 1 — RANDOM ATTRIBUTE PICKER
# Roll the dice, sailor. Whatever comes up, we're paintin' it.
# ──────────────────────────────────────────────

def random_spec() -> AvatarSpec:
    return AvatarSpec(
        setting   = random.choice(SETTINGS),
        art_style = random.choice(ART_STYLES),
        creature  = random.choice(CREATURES),
        mood      = random.choice(MOODS),
        lighting  = random.choice(LIGHTING),
    )


def unique_specs(count: int) -> list[AvatarSpec]:
    """Generate `count` specs with no duplicate attribute combos.

    Arrr, even slop deserves variety. No two identical cyberpunk zombie
    watercolours shall sail in the same fleet.
    """
    seen = set()
    specs = []
    attempts = 0
    while len(specs) < count:
        attempts += 1
        if attempts > count * 20:
            # The attribute pools be finite, ye greedy landlubber
            print(f"[warn] Could only generate {len(specs)} unique combos — relaxing uniqueness constraint.")
            break
        s = random_spec()
        key = (s.setting, s.art_style, s.creature, s.mood, s.lighting)
        if key not in seen:
            seen.add(key)
            specs.append(s)
    return specs


# ──────────────────────────────────────────────
# STEP 2 — vLLM PROMPT EXPANSION
# We ask one AI to write instructions for another AI.
# This is the future. Arrr.
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Stable Diffusion XL prompt engineer.
Given a set of avatar attributes, write a single, rich SDXL prompt for a
portrait/avatar image (face and shoulders only, close-up).

Rules:
- Output ONLY the prompt text — no explanation, no labels, no markdown.
- Start with the most important visual descriptors.
- Include specific artistic details that match the art style.
- 60-120 words. Dense with descriptors, comma-separated.
- Do NOT include negative prompts."""

def build_user_message(spec: AvatarSpec) -> str:
    return f"""Generate an SDXL portrait prompt for an avatar with these attributes:

- Setting:   {spec.setting}
- Art Style: {spec.art_style}
- Creature:  {spec.creature}
- Mood:      {spec.mood}
- Lighting:  {spec.lighting}

Portrait / avatar (face + shoulders, close-up framing)."""


def ask_vllm(spec: AvatarSpec) -> str:
    """Petition the LLM oracle to conjure words that will summon slop from the image kraken."""
    url = f"{VLLM_BASE}/v1/chat/completions"
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(spec)},
        ],
        "temperature": 0.9,  # A touch of chaos, as nature intended
        "max_tokens":  200,
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ──────────────────────────────────────────────
# STEP 3 — COMFYUI IMAGE GENERATION
# Here be the engine room. She's not pretty but she works.
# ──────────────────────────────────────────────

def build_comfy_workflow(prompt: str, negative: str, seed: int) -> dict:
    """
    Minimal SDXL txt2img workflow for ComfyUI.
    Nodes: CheckpointLoaderSimple → CLIPTextEncode (x2) → KSampler → VAEDecode → SaveImage

    Arrr, do not be tempted to make this fancier. It generates slop.
    Fancy slop is still slop.
    """
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": SDXL_CHECKPOINT},
        },
        "2": {
            # Positive prompt — what we DO want (a glorious orc, etc.)
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["1", 1],
            },
        },
        "3": {
            # Negative prompt — what we DON'T want (bad anatomy, watermarks, dignity)
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative,
                "clip": ["1", 1],
            },
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width":   1024,   # Big enough to see all the glorious slop
                "height":  1024,
                "batch_size": 1,
            },
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model":        ["1", 0],
                "positive":     ["2", 0],
                "negative":     ["3", 0],
                "latent_image": ["4", 0],
                "seed":         seed,   # Every piece of slop is unique ✨
                "steps":        30,
                "cfg":          7.0,
                "sampler_name": "dpmpp_2m",
                "scheduler":    "karras",
                "denoise":      1.0,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae":     ["1", 2],
            },
        },
        "7": {
            # She's done. Release the slop into the filesystem.
            "class_type": "SaveImage",
            "inputs": {
                "images":      ["6", 0],
                "filename_prefix": "avatar",
            },
        },
    }


def queue_comfy_prompt(workflow: dict) -> str:
    """Toss the workflow overboard into ComfyUI's queue. Returns a prompt_id to track our bounty."""
    resp = requests.post(
        f"{COMFYUI_BASE}/prompt",
        json={"prompt": workflow},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def wait_for_comfy(prompt_id: str, timeout: int = 300) -> list[dict]:
    """Poll /history until the job is done, return output image info.

    Arrr, we wait. The kraken renders at its own pace. Ye cannot rush slop.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{COMFYUI_BASE}/history/{prompt_id}", timeout=10)
        if resp.status_code == 200:
            history = resp.json()
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                images = []
                for node_output in outputs.values():
                    images.extend(node_output.get("images", []))
                return images
        time.sleep(2)  # Two seconds of contemplating our life choices
    raise TimeoutError(f"ComfyUI job {prompt_id} timed out after {timeout}s")


def fetch_comfy_image(image_info: dict) -> bytes:
    """Haul the treasure (slop) out of ComfyUI and into memory."""
    params = urllib.parse.urlencode({
        "filename": image_info["filename"],
        "subfolder": image_info.get("subfolder", ""),
        "type": image_info.get("type", "output"),
    })
    url = f"{COMFYUI_BASE}/view?{params}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def generate_images(spec: AvatarSpec, n: int, out_dir: Path) -> list[Path]:
    """Generate `n` images for a spec, save to out_dir, return saved paths.

    Each image gets a unique seed so the slop is varied. Consistent slop
    would be somehow worse.
    """
    saved = []
    slug = (
        f"{spec.setting}_{spec.art_style}_{spec.creature}"
        .lower()
        .replace(" ", "_")
        .replace("/", "-")
        [:60]  # Filenames have limits, even if our ambitions don't
    )

    for i in range(n):
        seed = random.randint(0, 2**31)
        workflow = build_comfy_workflow(spec.sdxl_prompt, NEGATIVE_PROMPT, seed)
        prompt_id = queue_comfy_prompt(workflow)
        images = wait_for_comfy(prompt_id)

        for img_info in images:
            img_bytes = fetch_comfy_image(img_info)
            fname = out_dir / f"{slug}_{i+1:02d}_seed{seed}.png"
            fname.write_bytes(img_bytes)
            saved.append(fname)

    return saved


# ──────────────────────────────────────────────
# STEP 4 — MANIFEST
# A proud ledger of every piece of slop we've created.
# Future archaeologists will be baffled.
# ──────────────────────────────────────────────

def save_manifest(specs: list[AvatarSpec], out_dir: Path):
    manifest = []
    for s in specs:
        d = asdict(s)
        d["images"] = [str(p) for p in s.images]
        manifest.append(d)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n📋 Manifest saved → {out_dir / 'manifest.json'}")


# ──────────────────────────────────────────────
# MAIN
# All hands on deck! It's time to make some slop!
# ──────────────────────────────────────────────

def main():
    import urllib.parse  # needed inside fetch_comfy_image

    parser = argparse.ArgumentParser(description="Avatar generation pipeline — AI slop ahoy!")
    parser.add_argument("--count",      type=int, default=20,       help="Number of unique attribute combos")
    parser.add_argument("--images-per", type=int, default=2,        help="Images generated per combo")
    parser.add_argument("--out",        type=str, default="./avatars", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎲 Generating {args.count} random attribute combos...")
    specs = unique_specs(args.count)

    print(f"🤖 Asking vLLM ({VLLM_MODEL}) to write SDXL prompts...\n")
    for spec in tqdm(specs, desc="LLM prompts"):
        spec.sdxl_prompt = ask_vllm(spec)

    print(f"\n🖼  Generating {args.images_per} image(s) per combo via ComfyUI...\n")
    for spec in tqdm(specs, desc="ComfyUI"):
        spec.images = generate_images(spec, args.images_per, out_dir)

    save_manifest(specs, out_dir)

    total = sum(len(s.images) for s in specs)
    print(f"\n✅ Done! {total} images saved to {out_dir}/")
    print("Arrr, the slop be plentiful. May it serve ye well, ye magnificent fool. 🏴‍☠️")


if __name__ == "__main__":
    import urllib.parse
    main()
