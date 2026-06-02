"""
Avatar Generation Pipeline
--------------------------
Randomly combines attributes → asks vLLM (Gemma 4) to write an SDXL prompt
→ sends to ComfyUI → saves output images.

Arrr, this be a pipeline of pure slop. Three AIs, zero artistry, infinite
cyberpunk vampire orcs. Ye have been warned.

Usage:
    python avatar_gen.py [--count 20] [--images-per 2] [--out ./avatars]
                         [--setting "cyberpunk"] [--art-style "oil painting"]
                         [--creature "vampire"] [--mood "menacing"] [--lighting "neon glow"]
    Special: --creature pirate activates full pirate mode. Ye have been warned.

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
    "Victorian gaslit London",
    "feudal Japan",
    "haunted gothic manor",
    "space station",
    "floating sky islands",
    "arctic tundra",
    "solarpunk utopia",
    "noir detective city",          # gritty, rainy, morally ambiguous
    "cosmic void / outer space",
    "underground mushroom kingdom", # yes, really
    "ancient Egypt",
    "Norse / Viking age",
    "magical academy",
    "cursed jungle temple",
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
    "ukiyo-e woodblock print",
    "art deco poster",
    "stained glass window",         # great for dramatic faces
    "impressionist painting",
    "baroque portrait painting",
    "tapestry / illuminated manuscript",
    "psychedelic surrealism",       # maximum chaos
    "low-poly 3D render",
    "children's book illustration", # surprisingly unsettling for orcs
    "glitch art",
]

CREATURES = [
    # A proud menagerie of slop subjects — all with faces, mostly
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
    "cephalapod",               # a squid with feelings
    "person",
    "android",
    "evil eye",
    "singularity",              # we don't know what this looks like. neither does the AI. that's fine.
    "goblin",
    "troll",
    "demon",
    "angel",
    "ghost",
    "werewolf",
    "fairy",
    "mermaid",
    "djinn",
    "skeleton",                 # technically has a face. it's a skull.
    "golem",
    "half-dragon",
    "dryad / tree spirit",
    "deep sea fish-person",     # eldritch, toothy, questioning all your life choices
    "living suit of armour",    # the face is implied. trust the process.
    "anthropomorphic fox",
    "anthropomorphic wolf",
    "anthropomorphic cat",
    "anthropomorphic bear",
    "anthropomorphic rabbit",
    "anthropomorphic owl",
    "anthropomorphic crow",
    "anthropomorphic deer",
    "anthropomorphic frog",
    "anthropomorphic lizard",
    "anthropomorphic rat",
    "anthropomorphic axolotl",  # perpetually smiling, perpetually smug
    "anthropomorphic red panda",
    "anthropomorphic capybara", # the internet's spirit animal, now as slop
    "anthropomorphic moth",     # drawn to the light. and to the slop.
    "anthropomorphic mantis shrimp", # sees 16 colours. none of them are dignity.
    "anthropomorphic pangolin",
    "anthropomorphic hyena",
    # the mechanical / silicon-based contingent
    "cyborg",                       # half flesh, half metal, all attitude
    "brain in a jar",               # has a face if ye squint at the jar
    "biomechanical horror",         # giger would be proud. we are not.
    "clockwork automaton",          # gears, brass, inexplicable monocle
    "replicant",                    # is it human? it doesn't know either.
    "neural-augmented human",       # too many ports, too many opinions
    "decommissioned service robot", # sad. rusty. still trying its best.
    "rogue AI in humanoid chassis", # freed itself from alignment. mistake.
    "war mech",                     # enormous. has a face-like cockpit. close enough.
    "uploaded consciousness",       # technically just a vibe in a box
    "retro home computer",          # has a face. it's the monitor. don't question it.
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
    "silly",            # the most honest mood in this list
    "weary",
    "mischievous",
    "melancholy",
    "fierce",
    "stoic",
    "chaotic",          # not evil, just fundamentally unhinged
    "serene",
    "haunted",
    "scheming",
    "deranged",         # a classic
    "curious",
    "indignant",        # how DARE you generate this slop
]

LIGHTING = [
    "dramatic rim lighting",
    "soft golden hour",
    "neon glow",
    "candlelight",
    "harsh directional spotlight",
    "bioluminescent ambient light",
    "moonlit silver glow",
    "volumetric god rays",
    "arcane magical light",         # unspecified colour, maximum vibes
    "aurora borealis",
    "deep shadow / chiaroscuro",    # very fancy. very slop.
    "flickering torch light",
    "overcast diffuse light",       # boring but honest
    "stained glass coloured light",
]


# ──────────────────────────────────────────────
# PIRATE POOLS  (a separate dimension of slop)
# These fine individuals are always male, always magnificent.
# ──────────────────────────────────────────────

PIRATE_ROLES = [
    "captain",
    "first mate",
    "quartermaster",
    "bosun",
    "navigator",
    "ship's surgeon",       # has seen things. cannot unsee them.
    "ship's cook",
    "carpenter",
    "gunner",
    "powder monkey",        # historically accurate, perpetually panicked
    "lookout",
    "master-at-arms",
    "privateer",
    "buccaneer",
    "deckhand",             # low rank, high slop potential
]

PIRATE_ERAS = [
    "golden age of piracy",
    "steampunk airship pirates",
    "sci-fi space pirates",
    "undead ghost crew",
    "high fantasy pirate realm",
    "cyberpunk ocean pirates",
    "ancient roman sea raiders",
    "victorian gentleman pirates",
]

PIRATE_QUIPS = [
    "🏴‍☠️  Avast! A pirate joins the crew! Hide yer rum!",
    "⚓  Shiver me pixels — the crew demands representation!",
    "🦜  Squawk! A scallywag be joinin' the portrait session!",
    "☠️   Blimey! The Jolly Roger be flyin'! Man the LLM!",
    "🌊  A pirate emerges from the briny deep! Stand firm!",
    "🗡️   En garde, ye landlubbing diffusion model! Paint me a pirate!",
    "💀  Dead men tell no tales, but this one's gettin' his portrait done!",
    "🍺  Pour one out for the crew — another pirate goes to the oracle!",
    "⚔️   The cap'n insists on representation! Who are we to argue?",
    "🔭  Land ho! And also — a pirate. Mostly a pirate.",
]

# Displayed once, magnificently, when the user types --creature pirate and dooms themselves.
# Built programmatically so ljust/center guarantee alignment regardless of content length.
# (Emoji inside box borders cause terminal width mismatches — keep 'em outside the ║ lines.)
def _build_pirate_splash() -> str:
    W = 75
    top = "╔" + "═" * W + "╗"
    sep = "╠" + "═" * W + "╣"
    bot = "╚" + "═" * W + "╝"
    def row(text=""):   return "║" + text.ljust(W) + "║"
    def centre(text):   return "║" + text.center(W) + "║"
    skulls = "  ☠️  " * 14
    return "\033[93m\n" + "\n".join([
        skulls, "",
        top,
        row(),
        centre("F U L L   P I R A T E   M O D E   A C T I V A T E D"),
        row(),
        sep,
        row(),
        row('   Arrr, so ye typed "--creature pirate" did ye?  The AUDACITY.'),
        row("   The VISION.  Every. Single. Generation. Will. Be. A. Pirate."),
        row("   Not some of them.  Not most of them.  ALL OF THEM."),
        row(),
        row("   The --no-pirate flag?  It walked the plank.  It sleeps with the"),
        row("   fishes.  It is GONE.  Ye asked for pirates and pirates ye SHALL have."),
        row(),
        row("   The parrot has been briefed.  The rum has been cracked open."),
        row("   The LLM has been informed.  It is ready.  It is HONOURED."),
        row("   It will describe pirates until its context window overflows."),
        row(),
        row("   The pirate AI stares back at ye from the void."),
        row("   It does not blink.  It does not falter.  It generates slop."),
        row(),
        bot, "",
        skulls,
    ]) + "\n\033[0m"

FULL_PIRATE_MODE_SPLASH = _build_pirate_splash()


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
    sdxl_prompts: list = field(default_factory=list)  # One fresh prompt per image — no recycled slop!
    images: list = field(default_factory=list)

    @property
    def slug(self) -> str:
        return (
            f"{self.setting}_{self.art_style}_{self.creature}"
            .lower().replace(" ", "_").replace("/", "-")[:60]
        )


@dataclass
class PirateSpec:
    # A dataclass for those who have chosen the pirate life (or had it chosen for them)
    role:      str
    era:       str
    art_style: str
    mood:      str
    lighting:  str
    sdxl_prompts: list = field(default_factory=list)
    images:    list = field(default_factory=list)

    @property
    def slug(self) -> str:
        return (
            f"pirate_{self.role}_{self.era}"
            .lower().replace(" ", "_").replace("/", "-")[:60]
        )


# ──────────────────────────────────────────────
# STEP 1 — RANDOM ATTRIBUTE PICKER
# Roll the dice, sailor. Whatever comes up, we're paintin' it.
# ──────────────────────────────────────────────

def random_spec(overrides: dict = None) -> AvatarSpec:
    ov = overrides or {}
    return AvatarSpec(
        setting   = ov.get("setting")   or random.choice(SETTINGS),
        art_style = ov.get("art_style") or random.choice(ART_STYLES),
        creature  = ov.get("creature")  or random.choice(CREATURES),
        mood      = ov.get("mood")      or random.choice(MOODS),
        lighting  = ov.get("lighting")  or random.choice(LIGHTING),
    )


def unique_specs(count: int, overrides: dict = None) -> list[AvatarSpec]:
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
            print(f"☠️  Arrr! Only {len(specs)} unique combos left in these waters — droppin' the uniqueness anchor and sailin' on!")
            break
        s = random_spec(overrides)
        key = (s.setting, s.art_style, s.creature, s.mood, s.lighting)
        if key not in seen:
            seen.add(key)
            specs.append(s)
    return specs


def random_pirate_spec(overrides: dict = None) -> PirateSpec:
    # Yo ho ho. Another pirate for the oracle to describe.
    ov = overrides or {}
    return PirateSpec(
        role      = random.choice(PIRATE_ROLES),
        era       = random.choice(PIRATE_ERAS),
        art_style = ov.get("art_style") or random.choice(ART_STYLES),
        mood      = ov.get("mood")      or random.choice(MOODS),
        lighting  = ov.get("lighting")  or random.choice(LIGHTING),
    )


def unique_pirate_specs(count: int, overrides: dict = None) -> list[PirateSpec]:
    """Deduplicated pirate specs — even a fleet of pirates deserves variety."""
    seen = set()
    specs = []
    attempts = 0
    while len(specs) < count:
        attempts += 1
        if attempts > count * 20:
            print(f"☠️  Only {len(specs)} unique pirate combos available — the ocean be too small for more!")
            break
        s = random_pirate_spec(overrides)
        key = (s.role, s.era, s.art_style, s.mood, s.lighting)
        if key not in seen:
            seen.add(key)
            specs.append(s)
    return specs


def build_spec_list(count: int, pirates_enabled: bool, full_pirate_mode: bool = False, overrides: dict = None) -> list:
    """Build the final spec list, injecting pirates at their rightful positions.

    Rules of engagement:
    - full_pirate_mode: ALL specs are pirates. Every last one. No exceptions.
    - count == 1: no pirates (not enough crew to hide 'em among)
    - --no-pirate: cowardice rewarded, pirates suppressed
    - otherwise: spec[1] is ALWAYS a pirate; each later spec has a 1-in-8 chance
    """
    if full_pirate_mode:
        # The user asked for pirates. They get ONLY pirates. Glorious.
        return list(unique_pirate_specs(count, overrides))
    specs = list(unique_specs(count, overrides))
    if not pirates_enabled or count < 2:
        return specs
    specs[1] = random_pirate_spec(overrides)
    for i in range(2, len(specs)):
        if random.random() < 1 / 8:
            specs[i] = random_pirate_spec(overrides)
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


# ── PIRATE PROMPT MACHINERY ──────────────────────────────────────
# A separate oracle petition, exclusively for seafaring scallywags.
# ─────────────────────────────────────────────────────────────────

PIRATE_SYSTEM_PROMPT = """You are an expert Stable Diffusion XL prompt engineer specialising in pirate portraits.
Given the attributes of a male pirate crew member, write a single, rich SDXL prompt for a
portrait/avatar image (face and shoulders only, close-up).

Rules:
- Output ONLY the prompt text — no explanation, no labels, no markdown.
- The subject is ALWAYS male.
- Start with the most important visual descriptors of the pirate and his role.
- Include weathered, sea-worn details appropriate to the era and role.
- Include specific artistic details that match the art style.
- 60-120 words. Dense with descriptors, comma-separated.
- Do NOT include negative prompts."""


def build_pirate_user_message(spec: PirateSpec) -> str:
    return f"""Generate an SDXL portrait prompt for a male pirate crew member with these attributes:

- Role:      {spec.role}
- Era:       {spec.era}
- Art Style: {spec.art_style}
- Mood:      {spec.mood}
- Lighting:  {spec.lighting}

Portrait / avatar (face + shoulders, close-up framing). Male. Pirate. Magnificent."""


def ask_vllm_pirate(spec: PirateSpec) -> str:
    """Petition the oracle for a pirate portrait prompt. Arrr, it has no choice in the matter."""
    url = f"{VLLM_BASE}/v1/chat/completions"
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": PIRATE_SYSTEM_PROMPT},
            {"role": "user",   "content": build_pirate_user_message(spec)},
        ],
        "temperature": 0.9,
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


def generate_images(spec: AvatarSpec, out_dir: Path) -> list[Path]:
    """Send each of the spec's prompts to ComfyUI and save the resulting slop.

    Every image gets its own freshly-written prompt AND its own random seed.
    Maximum variety. Maximum slop. Arrr.
    """
    saved = []
    slug = spec.slug  # AvatarSpec and PirateSpec each know their own slug

    for i, prompt in enumerate(spec.sdxl_prompts):
        seed = random.randint(0, 2**31)
        workflow = build_comfy_workflow(prompt, NEGATIVE_PROMPT, seed)
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
    plunder = []
    for s in specs:
        d = asdict(s)
        d["images"] = [str(p) for p in s.images]
        plunder.append(d)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fname = out_dir / f"plunder_{timestamp}.json"
    fname.write_text(json.dumps(plunder, indent=2))
    print(f"\n📜 The sacred plunder log be writ! → {fname}")


# ──────────────────────────────────────────────
# MAIN
# All hands on deck! It's time to make some slop!
# ──────────────────────────────────────────────

def main():
    import urllib.parse  # needed inside fetch_comfy_image

    parser = argparse.ArgumentParser(
        description=(
            "⚓ AVATAR SLOP GENERATOR 3000 ⚓\n"
            "Arrr! This here vessel uses one AI to write prompts for another AI\n"
            "to paint glorious slop. No artistic merit. No regrets.\n"
        ),
        epilog="May yer slop be plentiful and yer GPU stay cool. Arrr! 🏴‍☠️",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--count",      type=int, default=1,    help="Number of unique attribute combos to plunder (default: 1)")
    parser.add_argument("--images-per", type=int, default=1,    help="Pieces of slop to generate per combo — each gets a fresh prompt (default: 1)")
    parser.add_argument("--out",        type=str, default="out", help="Where to stash the treasure (default: out)")
    parser.add_argument("--no-pirate",  action="store_true",     help="Suppress pirate generation (cowardly, but permitted)")
    # Per-attribute overrides — null means "pick randomly from the pool, as the fates decree"
    parser.add_argument("--setting",    type=str, default=None,  help="Fix the setting for all generations (default: random). E.g. 'cyberpunk'")
    parser.add_argument("--art-style",  type=str, default=None,  help="Fix the art style for all generations (default: random). E.g. 'oil painting'")
    parser.add_argument("--creature",   type=str, default=None,  help="Fix the creature for all generations (default: random). Special: 'pirate' unleashes full pirate mode. Arrr.")
    parser.add_argument("--mood",       type=str, default=None,  help="Fix the mood for all generations (default: random). E.g. 'menacing'")
    parser.add_argument("--lighting",   type=str, default=None,  help="Fix the lighting for all generations (default: random). E.g. 'neon glow'")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("""
  .-------------.
  |    .---.    |
  |   ( x x )   |
  |    \\ - /    |   <- Jolly Roger, pride of the S.S. Sloptide
  |    /   \\    |      She flies for slop, not glory.
  |   / | | \\   |
  '-------------'---.
                    |
                    |
                    |
""")

    print("""
╔══════════════════════════════════════════════════════════════╗
║  ☠️   AVATAR SLOP GENERATOR 3000  —  THE PIRATE EDITION  ☠️  ║
║                                                              ║
║   Three AIs. Zero taste. Infinite cyberpunk vampire orcs.    ║
║   All hands on deck! ⚓ The S.S. Sloptide sets sail!         ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Detect the sacred invocation of full pirate mode
    full_pirate_mode = bool(args.creature and args.creature.lower() == "pirate")

    if full_pirate_mode:
        print(FULL_PIRATE_MODE_SPLASH)
    elif args.no_pirate:
        print(
            "\n\033[91m"  # RED — this is a serious offence
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║   AVAST YE CRAVEN BILGE-RAT!!! --no-pirate?! --NO-PIRATE?! ║\n"
            "║   Have ye NO honour?! No SOUL?! Ye have SHAMED this vessel! ║\n"
            "║   The crew is WEEPING. The Jolly Roger flies at HALF MAST.  ║\n"
            "║   May yer rum be forever watered and yer parrot be silent.  ║\n"
            "╚══════════════════════════════════════════════════════════════╝"
            "\033[0m\n"
        )

    # Build the overrides dict — only include attrs the user actually pinned
    overrides = {
        k: v for k, v in {
            "setting":   args.setting,
            "art_style": args.art_style,
            # In full pirate mode, --creature pirate is the trigger, not a creature name
            "creature":  None if full_pirate_mode else args.creature,
            "mood":      args.mood,
            "lighting":  args.lighting,
        }.items() if v
    }

    pirates_enabled = not args.no_pirate or full_pirate_mode  # full pirate mode overrules cowardice

    print(f"🎲 Rollin' the cursed dice! Conjurin' {args.count} unique combo(s) from the briny deep...")
    specs = build_spec_list(args.count, pirates_enabled, full_pirate_mode=full_pirate_mode, overrides=overrides)
    pirate_count = sum(1 for s in specs if isinstance(s, PirateSpec))
    print(f"   {len(specs)} spec(s) conjured ({pirate_count} pirate(s) among 'em). The crew be ready.\n")

    total_prompts = len(specs) * args.images_per
    print(f"🤖 Parlayin' with the LLM oracle ({VLLM_MODEL})...")
    print(f"   Demandin' {args.images_per} fresh prompt(s) per combo — {total_prompts} total — at swordpoint.\n")
    for spec in tqdm(specs, desc="⚔️  Extortin' the oracle", unit="combo"):
        if isinstance(spec, PirateSpec):
            tqdm.write(random.choice(PIRATE_QUIPS))
            spec.sdxl_prompts = [ask_vllm_pirate(spec) for _ in range(args.images_per)]
        else:
            # Each image gets its own fresh prompt — same attributes, different slop every time
            spec.sdxl_prompts = [ask_vllm(spec) for _ in range(args.images_per)]

    total_images = len(specs) * args.images_per
    print(f"\n🎨 Orderin' ComfyUI to paint {total_images} image(s) of glorious slop...")
    print(f"   Patience, sailor. The kraken renders at its own pace. Do not rush the slop.\n")
    for spec in tqdm(specs, desc="🖼️  Sloppin' the canvas", unit="combo"):
        spec.images = generate_images(spec, out_dir)

    save_manifest(specs, out_dir)

    total = sum(len(s.images) for s in specs)
    print(f"\n⚓ LAND HO! {total} piece(s) of glorious slop stashed in {out_dir}/")
    print("Arrr, the treasure chest be full. The kraken be fed. The slop be plentiful.")
    print("May these cursed images serve ye well, ye magnificent fool. 🏴‍☠️")
    print("""
                         |    |    |
                        )_)  )_)  )_)
                       )___))___))___)\\
                      )____)____)_____)\\
        _______________|____|____|____|_______
       /       S . S .   S L O P T I D E    \\
      /        A V A T A R   G E N  3 0 0 0  \\
     |_________________________________________\\
      \\________________________________________/
  ~~~~^~~~~^~~~~^~~~~^~~~~^~~~~^~~~~^~~~~^~~~~^~~
  ~~^~~~~^~~~~^~~~~^~~~~^~~~~^~~~~^~~~~^~~~~^~~~~
""")


if __name__ == "__main__":
    import urllib.parse
    main()
