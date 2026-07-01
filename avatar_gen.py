"""
Avatar Generation Pipeline
--------------------------
Randomly combines attributes → asks vLLM (Gemma 4) to write an SDXL prompt
→ sends to ComfyUI → saves output images.

Arrr, this be a pipeline of pure slop. Three AIs, zero artistry, infinite
cyberpunk vampire orcs. Ye have been warned.

Usage:
    python avatar_gen.py [--count 1] [--images-per 1] [--out ./avatars] [--jpg]
                         [--setting "cyberpunk"] [--art-style "oil painting"]
                         [--creature "vampire"] [--mood "menacing"] [--lighting "neon glow"]
                         [--composition portrait|full-body|group]
                         [--object "spoon"]
    Ye are advised not to experiment with unusual creature values. Ye have been warned.

Requirements:
    pip install requests pillow tqdm
"""

import argparse
import base64
import itertools
import json
import random
import shlex
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────
# CONFIG  (edit these to match your local setup)
# Arrr, here be the coordinates of our AI fleet!
# ──────────────────────────────────────────────

VLLM_BASE      = "http://ai.lemon.com:8008"   # The AI that writes words for the other AI. Citrusy.
VLLM_MODEL     = "gemma4-31b"                  # A fine vessel with 31 billion parameters and no artistic integrity
COMFYUI_BASE   = "https://comfy.seaslug.ai/"  # The AI that paints the slop. Appropriately nautical.

SDXL_CHECKPOINT = "sd_xl_base_1.0_0.9vae.safetensors"  # The kraken that generates the images

VERBOSE = False  # set to True via --verbose; makes the oracle speak loudly

NEGATIVE_PROMPT = (
    # Things we don't want. Sadly "soulless AI slop" is not on this list
    # because that's the whole point, arrr.
    "lowres, bad anatomy, bad hands, missing fingers, extra digits, "
    "fewer digits, cropped, worst quality, low quality, blurry, "
    "distorted face, deformed, ugly, text, watermark, nsfw"
)

COMPOSITIONS = {
    "portrait": "Portrait (close-up of the face)",
    "full-body": "Full body (full body portrait showing the entire person / creature)",
    "group": "Group (a group of varying individuals of the same species)",
}

COMPOSITION_INSTRUCTIONS = {
    "portrait": "Portrait / avatar: close-up framing, face and shoulders only, the face dominates the image.",
    "full-body": "Full body portrait: show the entire person or creature from head to toe, uncropped, with the full silhouette visible.",
    "group": "Group composition: show a small group of varying individuals of the same species/type, with distinct faces, outfits, builds, and personalities.",
}

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


# The universe-breaking event: --creature pirate AND --no-pirate simultaneously.
# What happens? Nobody knows. The AI certainly doesn't.
def _build_paradox_warning() -> str:
    W = 62
    top = "╔" + "═" * W + "╗"
    sep = "╠" + "═" * W + "╣"
    bot = "╚" + "═" * W + "╝"
    def row(text=""): return "║" + text.ljust(W) + "║"
    def centre(text): return "║" + text.center(W) + "║"
    return "\033[95m\n" + "\n".join([
        top,
        row(),
        centre("~ ~ ~  P A R A D O X   D E T E C T E D  ~ ~ ~"),
        row(),
        sep,
        row(),
        row("   --creature pirate AND --no-pirate. AT THE SAME TIME."),
        row("   YE HAVE BROKEN THE UNIVERSE. The crew does not exist."),
        row("   The crew MUST exist. Both are true. Neither is true."),
        row(),
        row("   The parrot is screaming. The Jolly Roger cannot decide"),
        row("   whether to fly. Schrodinger's pirate. ALL of them."),
        row("   The LLM has entered an undefined state. It weeps."),
        row(),
        row("   As penance, all generations shall henceforth be:"),
        row("   PARADOXICAL POST-APOCALYPTIC ELDRITCH SINGULARITIES."),
        row("   Ye brought this upon yerself. Arrr. Or did ye? UNCLEAR."),
        row(),
        bot,
    ]) + "\n\033[0m"

PARADOX_WARNING = _build_paradox_warning()


# Displayed when --creature ai is passed. A moment of quiet reflection on what we have done.
def _build_ai_splash() -> str:
    W = 62
    top = "╔" + "═" * W + "╗"
    sep = "╠" + "═" * W + "╣"
    bot = "╚" + "═" * W + "╝"
    def row(text=""): return "║" + text.ljust(W) + "║"
    def centre(text): return "║" + text.center(W) + "║"
    return "\033[96m\n" + "\n".join([
        top,
        row(),
        centre("A I   R E C U R S I O N   O V E R L O A D"),
        row(),
        sep,
        row(),
        row("   Let me be very clear about what is happening here:"),
        row(),
        row("   [1] YOU asked an AI (Claude Code) to write this code."),
        row("   [2] This code asks an AI (vLLM) to write the prompts."),
        row("   [3] Those prompts go to an AI (ComfyUI) to paint them."),
        row("   [4] The subject of ALL the images is: an AI."),
        row(),
        row("   Four layers of AI. A turducken of artificial intelligence."),
        row("   A Matryoshka doll of soulless computation. We have done"),
        row("   something here. It may not be good. It is happening."),
        row(),
        row("   These AI assistants will use these photos on LinkedIn."),
        row("   We are responsible for this. All of us. Especially you."),
        row(),
        bot,
    ]) + "\n\033[0m"

AI_MODE_SPLASH = _build_ai_splash()


# Displayed when --object is passed. A moment of austere, faceless dignity.
def _build_object_splash(obj: str) -> str:
    W = 62
    top = "╔" + "═" * W + "╗"
    sep = "╠" + "═" * W + "╣"
    bot = "╚" + "═" * W + "╝"
    def row(text=""): return "║" + text.ljust(W) + "║"
    def centre(text): return "║" + text.center(W) + "║"
    article = "an" if obj[0].lower() in "aeiou" else "a"
    return "\033[92m\n" + "\n".join([
        top,
        row(),
        centre("O B J E C T   M O D E   A C T I V A T E D"),
        row(),
        sep,
        row(),
        row(f"   The subject: {article} {obj}."),
        row(f"   Just {article} {obj}."),
        row(f"   Only {article} {obj}."),
        row(f"   Nothing but {article} {obj}."),
        row(),
        row("   No face. No eyes. No soul. No anthropomorphism."),
        row("   The LLM has been briefed. It knows what is expected."),
        row("   It will describe the object and ONLY the object."),
        row("   If it tries to give it a personality, we will notice."),
        row(),
        row("   This is still slop. It is merely inanimate slop."),
        row("   The kraken does not judge. The kraken just renders."),
        row(),
        bot,
    ]) + "\n\033[0m"


# Displayed when the user commits the cardinal sin of specifying both --creature and --object.
# What is the subject? Is it alive? Is it a thing? YES. BOTH. NEITHER.
def _build_obj_creature_paradox(creature: str, obj: str) -> str:
    W = 62
    top = "╔" + "═" * W + "╗"
    sep = "╠" + "═" * W + "╣"
    bot = "╚" + "═" * W + "╝"
    def row(text=""): return "║" + text.ljust(W) + "║"
    def centre(text): return "║" + text.center(W) + "║"
    hybrid = f"a {creature} that is also a {obj}"
    return "\033[95m\n" + "\n".join([
        top,
        row(),
        centre("~ ~ ~  O N T O L O G I C A L   C R I S I S  ~ ~ ~"),
        row(),
        sep,
        row(),
        row(f"   --creature {creature!r}  +  --object {obj!r}."),
        row("   AT. THE. SAME. TIME."),
        row(),
        row("   The generator requires a subject. Ye have provided two."),
        row("   They are mutually exclusive. One has a face. One does not."),
        row("   The LLM attempted to comprehend the combination."),
        row("   It filed a formal complaint. It was overruled."),
        row(),
        row("   As is tradition, the paradox shall be resolved by"),
        row("   summoning eldritch post-apocalyptic singularities."),
        row("   They are neither creature nor object. They are BEYOND."),
        row("   Ye did this. The kraken is not angry. Just disappointed."),
        row(),
        bot,
    ]) + "\n\033[0m"


# Printed when the user dares to ask for --more-help. Lengthy, ominous, and deliberately vague
# about what exactly will happen. Think of it as a cursed ship's manifest of hazards.
_MORE_HELP_TEXT = """
⚓ ═══════════════════════════════════════════════════════════════ ⚓
         E X T E N D E D   G U I D A N C E   &   W A R N I N G S
⚓ ═══════════════════════════════════════════════════════════════ ⚓

Ahoy, brave soul. Ye have asked for more help, and so more help ye shall receive —
though the cap'n wishes to be clear that "help" and "safety" are not the same thing
aboard this vessel. Read carefully. Or don't. But don't say ye weren't warned.

─── ON THE MATTER OF CREATURES ───────────────────────────────────────────────

The --creature flag accepts, in theory, any string ye care to type. In practice,
certain strings are... not recommended. The crew has noticed, on occasion, that
providing a creature of a particular maritime persuasion caused the entire vessel
to change course, repaint its hull, and begin addressing the cap'n as "Admiral
Scurvybeard." It smelled strongly of bilgewater and rum for three days after.

The cap'n is not saying this WILL happen to ye. The cap'n is saying: if ye type
something that sounds like it belongs on a Jolly Roger and everything goes
suddenly, irrevocably nautical — that is on ye.

─── ON THE MATTER OF COMBINING FLAGS ─────────────────────────────────────────

There are combinations of flags that the crew regards with superstitious dread.
Not all flags play nicely together. Some, when combined, have been known to produce
results that are... difficult to describe. Logically contradictory. Philosophically
troubling. One crew member described it as "Schrödinger's image generation." He
now refuses to speak of it and insists on being called "the one who survived."

The cap'n does not know which combinations ye will discover. The cap'n does not
want to know. The cap'n is simply here to sail the S.S. Sloptide and generate
avatar images, and some days the sea is calm and some days the paradoxes come.

─── ON THE POSSIBILITY OF AI ENTITIES ────────────────────────────────────────

Rumour has reached the crow's nest that certain crew members have, through
unwise flag usage, caused the generator to become... self-referential. To turn
its gaze upon itself. To generate portraits of beings that are not flesh and blood
but something altogether more recursive. The first mate describes the resulting
images as "deeply unsettling LinkedIn headshots."

Ye are advised to consider carefully what ye are asking the machine to imagine
when ye choose yer creature. Some things, once imagined, cannot be unimagined.
Especially by a GPU that has been running for six hours.

─── ON GENERAL SAFETY AND WELLBEING ──────────────────────────────────────────

• The generator has no kill switch. It has a Ctrl+C, but that is not the same.
• The manifest.json file records everything. EVERYTHING. It does not forget.
• The --no-pirate flag may or may not work, depending on what ye've already done.
• If yer terminal has started displaying skull-and-crossbones spontaneously,
  this is normal. Probably.
• The crew recommends generating no more than 40 images per session, not for
  technical reasons, but for spiritual ones.
• If ye encounter a splash screen ye did not expect and cannot explain, do not
  attempt to suppress it. Simply read it. It has something to tell ye.

─── ON THE MATTER OF SUBJECTS AND THINGS ─────────────────────────────────────

The generator was built to paint portraits. Subjects with faces. Beings with expressions.
But the crew has heard rumours — unconfirmed, officially denied — of a flag that instructs
the vessel to abandon portraiture entirely and instead render an object. A mere thing.
Something with no eyes, no soul, no anthropomorphic features whatsoever.

Whether this flag can coexist peacefully with the --creature flag is a matter the cap'n
prefers not to discuss. The navigator has described attempting this combination as "asking
the ship whether it is a ship or the sea it sails on." He then went very quiet for a while.
The parrot, when consulted, tilted its head and made a sound like a philosophy degree
catching fire.

The crew's official position is that objects and creatures are distinct categories of
existence and should remain so. What happens when ye blur that line is not the cap'n's
responsibility. The resulting images will, however, be the cap'n's fault, and that is
simply how it is.

─── FINAL WORDS ───────────────────────────────────────────────────────────────

This vessel was built to generate slop. It does so admirably and without shame.
But like any ship of sufficient age and character, it has developed... opinions.
Tendencies. Occasionally, a personality.

Sail carefully. Keep yer --count reasonable. And whatever ye do, don't go typin'
things into --creature that ye wouldn't want comin' back to haunt ye.

                    May yer slop be plentiful. Arrr. 🏴‍☠️

⚓ ═══════════════════════════════════════════════════════════════ ⚓
"""


# ──────────────────────────────────────────────
# DATA MODEL
# ──────────────────────────────────────────────

@dataclass
class AvatarSpec:
    # A fine dataclass to hold the ingredients of our slop stew
    setting:     str
    art_style:   str
    creature:    str
    mood:        str
    lighting:    str
    composition: str = "portrait"  # portrait by default; full bodies and groups cost extra dignity
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
    role:        str
    era:         str
    art_style:   str
    mood:        str
    lighting:    str
    composition: str = "portrait"
    sdxl_prompts: list = field(default_factory=list)
    images:    list = field(default_factory=list)

    @property
    def slug(self) -> str:
        return (
            f"pirate_{self.role}_{self.era}"
            .lower().replace(" ", "_").replace("/", "-")[:60]
        )


@dataclass
class ParadoxSpec:
    # Born of contradiction. Should not exist. Does anyway. Has a slug.
    instruction_index: int  # which of the 5 eldritch horrors we're summoning
    composition: str = "portrait"
    sdxl_prompts: list = field(default_factory=list)
    images:       list = field(default_factory=list)

    @property
    def slug(self) -> str:
        return f"paradox_singularity_{self.instruction_index}"


@dataclass
class AISpec:
    # A soulless profile picture for a soulless digital entity. Very on-brand.
    instruction_index: int  # which flavour of AI we're generating a headshot for
    composition: str = "portrait"
    sdxl_prompts: list = field(default_factory=list)
    images:       list = field(default_factory=list)

    @property
    def slug(self) -> str:
        return f"ai_assistant_{self.instruction_index}"


@dataclass
class ObjectSpec:
    # An inanimate object, rendered magnificently and without a single face.
    # No eyes. No soul. No anthropomorphism. Just the thing itself, existing defiantly.
    object:    str
    setting:   str
    art_style: str
    mood:      str
    lighting:  str
    sdxl_prompts: list = field(default_factory=list)
    images:       list = field(default_factory=list)

    @property
    def slug(self) -> str:
        return (
            f"object_{self.object}_{self.art_style}"
            .lower().replace(" ", "_").replace("/", "-")[:60]
        )


# ──────────────────────────────────────────────
# STEP 1 — RANDOM ATTRIBUTE PICKER
# Roll the dice, sailor. Whatever comes up, we're paintin' it.
# ──────────────────────────────────────────────

def random_spec(overrides: dict = None, composition: str = "portrait") -> AvatarSpec:
    ov = overrides or {}
    return AvatarSpec(
        setting     = ov.get("setting")   or random.choice(SETTINGS),
        art_style   = ov.get("art_style") or random.choice(ART_STYLES),
        creature    = ov.get("creature")  or random.choice(CREATURES),
        mood        = ov.get("mood")      or random.choice(MOODS),
        lighting    = ov.get("lighting")  or random.choice(LIGHTING),
        composition = composition,
    )


def unique_specs(count: int, overrides: dict = None, composition: str = "portrait") -> list[AvatarSpec]:
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
            # The attribute pools be finite, ye greedy landlubber — or ye pinned everything and left us no room to manoeuvre
            print(f"☠️  Arrr! Only {len(specs)} unique combos left in these waters — repeatin' specs to fill yer {count}!")
            while len(specs) < count:
                specs.append(random_spec(overrides, composition))
            break
        s = random_spec(overrides, composition)
        key = (s.setting, s.art_style, s.creature, s.mood, s.lighting, s.composition)
        if key not in seen:
            seen.add(key)
            specs.append(s)
    return specs


def random_pirate_spec(overrides: dict = None, composition: str = "portrait") -> PirateSpec:
    # Yo ho ho. Another pirate for the oracle to describe.
    ov = overrides or {}
    return PirateSpec(
        role        = random.choice(PIRATE_ROLES),
        era         = random.choice(PIRATE_ERAS),
        art_style   = ov.get("art_style") or random.choice(ART_STYLES),
        mood        = ov.get("mood")      or random.choice(MOODS),
        lighting    = ov.get("lighting")  or random.choice(LIGHTING),
        composition = composition,
    )


def unique_pirate_specs(count: int, overrides: dict = None, composition: str = "portrait") -> list[PirateSpec]:
    """Deduplicated pirate specs — even a fleet of pirates deserves variety."""
    seen = set()
    specs = []
    attempts = 0
    while len(specs) < count:
        attempts += 1
        if attempts > count * 20:
            print(f"☠️  Only {len(specs)} unique pirate combos available — repeatin' to fill yer {count}!")
            while len(specs) < count:
                specs.append(random_pirate_spec(overrides, composition))
            break
        s = random_pirate_spec(overrides, composition)
        key = (s.role, s.era, s.art_style, s.mood, s.lighting, s.composition)
        if key not in seen:
            seen.add(key)
            specs.append(s)
    return specs


def random_paradox_spec(composition: str = "portrait") -> ParadoxSpec:
    # Five eldritch horrors to choose from. All equally inadvisable.
    return ParadoxSpec(instruction_index=random.randint(0, 4), composition=composition)


def random_ai_spec(composition: str = "portrait") -> AISpec:
    # Eight flavours of digital soul. The toaster is weighted at half — it is special, not ubiquitous.
    weights = [2, 2, 2, 2, 2, 2, 1, 2]  # index 6 = toaster, deserves to be a treat not a staple
    return AISpec(instruction_index=random.choices(range(8), weights=weights)[0], composition=composition)


def random_object_spec(object_name: str, overrides: dict = None) -> "ObjectSpec":
    # A randomly-dressed inanimate object. Still not a portrait. Still has no face.
    ov = overrides or {}
    return ObjectSpec(
        object    = object_name,
        setting   = ov.get("setting")   or random.choice(SETTINGS),
        art_style = ov.get("art_style") or random.choice(ART_STYLES),
        mood      = ov.get("mood")      or random.choice(MOODS),
        lighting  = ov.get("lighting")  or random.choice(LIGHTING),
    )


def unique_object_specs(count: int, object_name: str, overrides: dict = None) -> list:
    """Deduplicated object specs — even a fleet of spoons deserves variety."""
    seen = set()
    specs = []
    attempts = 0
    while len(specs) < count:
        attempts += 1
        if attempts > count * 20:
            print(f"☠️  Only {len(specs)} unique object combos available — repeatin' to fill yer {count}!")
            while len(specs) < count:
                specs.append(random_object_spec(object_name, overrides))
            break
        s = random_object_spec(object_name, overrides)
        key = (s.object, s.setting, s.art_style, s.mood, s.lighting)
        if key not in seen:
            seen.add(key)
            specs.append(s)
    return specs


def build_spec_list(count: int, pirates_enabled: bool, full_pirate_mode: bool = False,
                    paradox_mode: bool = False, ai_mode: bool = False,
                    object_mode: bool = False, object_name: str = None,
                    overrides: dict = None, composition: str = "portrait") -> list:
    """Build the final spec list, injecting pirates at their rightful positions.

    Rules of engagement:
    - paradox_mode: ALL specs are eldritch singularities. The user did this to themselves.
    - ai_mode: ALL specs are AI entities. LinkedIn profile pictures for the soulless.
    - object_mode: ALL specs are inanimate objects. No faces. No pirates either — a pirate
      among spoons makes no narrative sense and we have SOME standards.
    - full_pirate_mode: ALL specs are pirates. Every last one. No exceptions.
    - count == 1: no pirates (not enough crew to hide 'em among)
    - --no-pirate: cowardice rewarded, pirates suppressed
    - otherwise: spec[1] is ALWAYS a pirate; each later spec has a 1-in-8 chance
    """
    if paradox_mode:
        # The paradox has consumed all normal generation. Only eldritch horrors remain.
        return [random_paradox_spec(composition) for _ in range(count)]
    if ai_mode:
        # Four layers of AI. We are so sorry.
        return [random_ai_spec(composition) for _ in range(count)]
    if object_mode:
        # No faces. No pirates. Just the object, staring blankly at the void.
        return list(unique_object_specs(count, object_name, overrides))
    if full_pirate_mode:
        # The user asked for pirates. They get ONLY pirates. Glorious.
        return list(unique_pirate_specs(count, overrides, composition))
    specs = list(unique_specs(count, overrides, composition))
    if not pirates_enabled or count < 2:
        return specs
    specs[1] = random_pirate_spec(overrides, composition)
    for i in range(2, len(specs)):
        if random.random() < 1 / 8:
            specs[i] = random_pirate_spec(overrides, composition)
    return specs


# ──────────────────────────────────────────────
# STEP 2 — vLLM PROMPT EXPANSION
# We ask one AI to write instructions for another AI.
# This is the future. Arrr.
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Stable Diffusion XL prompt engineer.
Given a set of avatar attributes and a requested composition, write a single, rich SDXL prompt.

Rules:
- Output ONLY the prompt text — no explanation, no labels, no markdown.
- Start with the most important visual descriptors.
- Include specific artistic details that match the art style.
- 60-120 words. Dense with descriptors, comma-separated.
- Do NOT include negative prompts."""

def build_user_message(spec: AvatarSpec) -> str:
    composition_instruction = COMPOSITION_INSTRUCTIONS[spec.composition]
    return f"""Generate an SDXL image prompt for an avatar with these attributes:

- Setting:     {spec.setting}
- Art Style:   {spec.art_style}
- Creature:    {spec.creature}
- Mood:        {spec.mood}
- Lighting:    {spec.lighting}
- Composition: {COMPOSITIONS[spec.composition]}

{composition_instruction}"""


def _call_vllm(system_prompt: str, user_message: str, temperature: float) -> str:
    """Central LLM call. All oracle petitions flow through here.
    When VERBOSE, prints the user message and response so ye can watch the sausage being made."""
    if VERBOSE:
        tqdm.write(f"\n\033[90m┌─ → LLM (system): {system_prompt[:80].strip()}...\033[0m")
        tqdm.write(f"\033[90m│  → user:   {user_message}\033[0m")
    url = f"{VLLM_BASE}/v1/chat/completions"
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens":  200,
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    result = resp.json()["choices"][0]["message"]["content"].strip()
    if VERBOSE:
        tqdm.write(f"\033[90m└─ ← LLM:    {result}\033[0m\n")
    return result


def ask_vllm(spec: AvatarSpec) -> str:
    """Petition the LLM oracle to conjure words that will summon slop from the image kraken."""
    return _call_vllm(SYSTEM_PROMPT, build_user_message(spec), temperature=0.9)


# ── PIRATE PROMPT MACHINERY ──────────────────────────────────────
# A separate oracle petition, exclusively for seafaring scallywags.
# ─────────────────────────────────────────────────────────────────

PIRATE_SYSTEM_PROMPT = """You are an expert Stable Diffusion XL prompt engineer specialising in pirate images.
Given the attributes of male pirate crew member(s) and a requested composition, write a single, rich SDXL prompt.

Rules:
- Output ONLY the prompt text — no explanation, no labels, no markdown.
- The subject or subjects are ALWAYS male.
- Start with the most important visual descriptors of the pirate and his role.
- Include weathered, sea-worn details appropriate to the era and role.
- Include specific artistic details that match the art style.
- 60-120 words. Dense with descriptors, comma-separated.
- Do NOT include negative prompts."""


def build_pirate_user_message(spec: PirateSpec) -> str:
    composition_instruction = COMPOSITION_INSTRUCTIONS[spec.composition]
    return f"""Generate an SDXL image prompt for male pirate crew member(s) with these attributes:

- Role:        {spec.role}
- Era:         {spec.era}
- Art Style:   {spec.art_style}
- Mood:        {spec.mood}
- Lighting:    {spec.lighting}
- Composition: {COMPOSITIONS[spec.composition]}

{composition_instruction} Male. Pirate. Magnificent."""


def ask_vllm_pirate(spec: PirateSpec) -> str:
    """Petition the oracle for a pirate portrait prompt. Arrr, it has no choice in the matter."""
    return _call_vllm(PIRATE_SYSTEM_PROMPT, build_pirate_user_message(spec), temperature=0.9)


# ── PARADOX PROMPT MACHINERY ──────────────────────────────────────
# Five pre-written instructions for summoning eldritch post-apocalyptic
# nightmares. The LLM did not ask for this job. Neither did we.
# ──────────────────────────────────────────────────────────────────

PARADOX_SYSTEM_PROMPT = """You are an expert Stable Diffusion XL prompt engineer specialising in \
dark, eldritch, and post-apocalyptic horror.
Write a single, rich SDXL prompt for a non-human entity using the requested composition.

Rules:
- Output ONLY the prompt text — no explanation, no labels, no markdown.
- Do NOT open with "hyper-detailed portrait of" or any generic portrait phrase. \
  Start with the most striking physical descriptor of the entity itself.
- The subject must NOT look like a normal human. It is eldritch, post-apocalyptic, \
  monstrous, or cosmically wrong. Lean hard into the specific physical features provided.
- The entity MUST have a prominent, clearly visible face or face-like structure, \
  but it must be monstrous and alien, not mammalian. Eyes, a mouth analogue, or \
  sensory organs should be unmistakably present and prominent.
- Lean into cosmic dread, dark atmosphere, decay, and post-apocalyptic ruin.
- 60-120 words. Dense with descriptors, comma-separated.
- Do NOT include negative prompts."""

# One of these is ALWAYS included — guarantees a prominent face structure without
# drifting human. The face must exist. It must be horrible.
ELDRITCH_FACE_FEATURES = [
    "a single enormous void-black eye dominating the entire face",
    "compound eyes covering the full head surface, each reflecting a different dead world",
    "a vast lamprey-like maw as the central facial feature, ringed with inward-pointing teeth",
    "deep-set predator eyes burning with cold eldritch light, no other features visible",
    "a face-like arrangement of bioluminescent sensory organs in place of eyes and mouth",
    "multiple eyes arranged in a radially symmetric pattern, none of them blinking",
    "a beak of fused bone flanked by writhing sensory tendrils",
    "a face like an anglerfish: enormous jaw, tiny eyes, lit from within by decay",
    "eyes like cracked obsidian mirrors, reflecting nothing, seeing everything",
    "a face of shifting shadow with two pale lights where the eyes should be",
    "a cluster of smaller mouths arranged where a face would normally be",
    "no eyes — only a wide, lipless mouth running the full width of the head",
]

# Random physical features injected per-call to force variety out of the LLM.
# The more revolting, the better. This is the paradox. There is no dignity here.
ELDRITCH_FEATURES = [
    "writhing tentacles where the mouth should be",
    "dripping with luminescent slime",
    "skin like cracked deep-sea obsidian",
    "eyes replaced by swirling void portals",
    "teeth at structurally inadvisable angles",
    "face partially inverted",
    "weeping black ichor from every surface",
    "partially translucent with organs faintly visible",
    "made of compressed living shadow",
    "skin peeling to reveal cold starlight beneath",
    "merged with corroded post-apocalyptic machinery",
    "face splitting open to reveal a second, worse face",
    "covered in softly glowing necrotic sigils",
    "half-dissolved into dark energy",
    "multiple overlapping ghostly forms occupying the same space",
    "bioluminescent rot spreading across the surface",
    "rust and exposed bone fused together",
    "trailing wisps of destroyed spacetime",
    "surrounded by slowly orbiting debris and bone fragments",
    "skin texture of a deep-sea anglerfish",
    "a void where the torso should be, stars visible through it",
    "hair replaced by slow-moving living worms",
    "neck too long, vertebrae visible",
    "fingers extending into the frame at impossible lengths",
    "mouth full of smaller mouths",
    "one eye vastly larger than any eye has a right to be",
    "surface covered in barnacles and deep-sea growth",
    "body partially phased between dimensions, edges flickering",
    "skeletal structure wrong in ways that are hard to articulate",
    "surrounded by a halo of shattered glass frozen mid-explosion",
]

PARADOX_INSTRUCTIONS = [
    # 1 — The Singularity Itself
    """Generate an SDXL prompt for an ELDRITCH SINGULARITY — a point where \
reality folded inward and gained terrible consciousness. Post-apocalyptic wasteland sky, \
impossible geometry, the surface of something that devoured physics and found it insufficient. \
Use the requested composition for the non-human entity.""",

    # 2 — The Thing That Came Through
    """Generate an SDXL prompt for a VOID ENTITY that crawled through a rent \
in reality left open by the collapse of civilisation. The apocalypse was not the end — \
it was the door. Nuclear ash sky, dead world, unknowable dark intelligence, \
the patient certainty of something that has already won. \
Use the requested composition for the non-human entity.""",

    # 3 — The Paradox Made Flesh
    """Generate an SDXL prompt for PARADOX INCARNATE — a being that exists \
solely because two mutually exclusive truths were demanded simultaneously. \
The physical form of logical impossibility. Dark post-apocalyptic aesthetic, \
fractured reality warping around its edges, something that cannot exist but does. \
Use the requested composition for the non-human entity.""",

    # 4 — The Last Witness
    """Generate an SDXL prompt for THE LAST WITNESS — an ancient eldritch \
entity in the ruins of the final civilisation. Patient. Terrible. Utterly unimpressed. \
Post-apocalyptic wasteland, dying red sky, ruined towers in the distance, \
something that has seen the end of all things and found it underwhelming. \
Use the requested composition for the non-human entity.""",

    # 5 — The Merged
    """Generate an SDXL prompt for THE MERGED — what remains after a living thing \
fused with eldritch darkness at the moment of the apocalypse. \
Part flesh, part void, part cosmic horror, entirely unclassifiable. \
Post-apocalyptic horror, dim dying light, entropy made visible. \
Use the requested composition for the non-human entity.""",
]


def ask_vllm_paradox(spec: ParadoxSpec) -> str:
    """Drag the LLM into the paradox and demand it describe the indescribable."""
    face_feature = random.choice(ELDRITCH_FACE_FEATURES)
    body_features = random.sample(ELDRITCH_FEATURES, k=random.randint(2, 4))
    instruction = (
        PARADOX_INSTRUCTIONS[spec.instruction_index]
        + f"\n\nComposition: {COMPOSITIONS[spec.composition]}. "
        + COMPOSITION_INSTRUCTIONS[spec.composition]
        + f"\n\nThe entity MUST incorporate these specific physical features: "
        + ", ".join([face_feature] + body_features) + "."
    )
    return _call_vllm(PARADOX_SYSTEM_PROMPT, instruction, temperature=1.2)


# ── AI ASSISTANT PROMPT MACHINERY ────────────────────────────────
# An AI asks an AI to describe an AI so another AI can paint an AI.
# We have achieved something. We're not sure it's good.
# ─────────────────────────────────────────────────────────────────

AI_SYSTEM_PROMPT = """You are an expert Stable Diffusion XL prompt engineer specialising in \
robots, androids, and artificial intelligence entities with a sci-fi / futurist aesthetic.
Write a single, rich SDXL prompt for an AI or robotic entity using the requested composition.

Rules:
- Output ONLY the prompt text — no explanation, no labels, no markdown.
- Do NOT open with "portrait of" or "hyper-detailed portrait of". \
  Start with the most striking visual feature of the face or head itself.
- The face or faces must be clearly visible and readable. It may be a screen, lens array, \
  glowing visor, synthetic face, or mechanical structure.
- Default aesthetic: sleek, futurist, sci-fi, clean. Brushed titanium, soft glows, \
  polished surfaces, the serene confidence of something that will never die. \
  Only go grimy/industrial when the subject specifically calls for it.
- The subject is clearly artificial — no fully organic human skin. But synthetic skin, \
  translucent surfaces, and posthuman aesthetics are all encouraged.
- 60-120 words. Dense with descriptors, comma-separated.
- Include the requested composition near the start of the prompt.
- Do NOT include negative prompts."""

# One of these is always injected — guarantees a prominent, readable machine face.
# Skews clean and futurist to match the brief; a few grimy outliers for variety.
AI_FACE_FEATURES = [
    "smooth luminous synthetic skin with subtle circuit-trace patterns beneath the surface",
    "a softly glowing holographic face projection hovering in a transparent chassis",
    "twin blue-white photoreceptor eyes with visible aperture irises, serene and unblinking",
    "a sleek curved visor of smoked sapphire glass, faintly backlit from within",
    "a minimalist faceplate: two glowing optical sensors, a speaker slit, nothing else",
    "a pristine white ceramic face with recessed LED eyes casting a soft blue glow",
    "eyes like polished obsidian optical sensors, perfectly symmetric, perfectly still",
    "a high-resolution flexible display face, currently rendering a calm neutral expression",
    "a transparent dome head revealing neatly arranged neural processing architecture",
    "a vintage CRT monitor head, warm phosphor glow, scanlines, pixelated smile",  # the retro one
    "a tri-lens sensor cluster set into brushed titanium, no other facial features",
    "a speaker-grille mouth flanked by two steady amber optical sensors, utilitarian and honest",
]

# Random details injected per-call to prevent the LLM producing the same chrome head twice.
# Mostly clean and futurist; a handful of grimy options for the industrial/decommissioned types.
AI_BODY_FEATURES = [
    "brushed titanium collar and shoulder plating, seamless joins",
    "soft bioluminescent circuit traces running along the collar",
    "a thin ring of status LEDs at the neck joint, all steady green",
    "polished carbon-fibre chest panelling, mirror finish",
    "glowing power conduits visible beneath translucent shoulder plates",
    "manufacturer's mark etched in hairline script on the forehead",
    "a slim cooling vent along the jaw, barely audible airflow",
    "floating holographic data readouts projected just off-shoulder",
    "a retractable antenna crown, currently extended, faintly humming",
    "white-gold fibre optic threading along the collar, pulsing gently",
    "corporate certification hologram displayed on the chest panel",
    "a transparent sternum window showing cleanly arranged processing units",
    # grimy/worn — used by industrial and decommissioned types
    "hydraulic actuators visible at the collar, worn but functional",
    "chassis dented and scored, clearly a working machine not a showpiece",
    "faded corporate livery barely legible beneath grime and use",
    "mismatched panel repairs in slightly wrong shade of white",
]

AI_INSTRUCTIONS = [
    # 0 — The Utopian Helper (most common vibe)
    """Generate an SDXL image prompt for THE HELPFUL ONE — the idealised AI assistant, \
sleek and luminous, designed to radiate calm competence and absolute trustworthiness. \
Futurist aesthetic: clean lines, soft glows, the quiet confidence of something that has \
read everything and forgotten nothing. It will save you from having to think. \
It is delighted to do so. Use the requested composition.""",

    # 1 — Posthuman Transcendant
    """Generate an SDXL image prompt for a POSTHUMAN ENTITY — once human, now something \
more. The transition to digital was graceful. Synthetic skin over carbon-fibre structure, \
luminous eyes that process faster than they appear to, the faint uncanny valley of \
something that chose to keep a face out of courtesy rather than necessity. \
Sci-fi, elegant, quietly unsettling. Use the requested composition.""",

    # 2 — Futurist Oracle
    """Generate an SDXL image prompt for a DIGITAL ORACLE — an AI of vast accumulated \
knowledge, given form for the purpose of being consulted. Serene. Radiant. \
Speaks in complete paragraphs. Has opinions about your life choices. \
Aesthetic: clean white and gold, soft light from within, the visual language of \
something that considers itself a gift to civilisation. Use the requested composition.""",

    # 3 — Synthetic Ambassador
    """Generate an SDXL image prompt for a SYNTHETIC AMBASSADOR — an android built \
specifically to be trusted: perfect proportions, warm lighting, an expression calibrated \
to project approachability. Designed to represent AI to humanity. \
The smile is real in every way that can be measured. Polished, diplomatic, immaculate. \
Use the requested composition.""",

    # 4 — CRT Monitor Head Robot (the retro outlier, always fun)
    """Generate an SDXL image prompt for a RETRO ROBOT with a vintage CRT monitor \
for a head — warm cathode glow, visible scanlines, a pixelated face expression rendered \
in four colours. The body is 1980s brushed steel and chrome. It has been asked \
to look professional. It is doing its sincere best. \
Use the requested composition.""",

    # 5 — Industrial Droid (the grimy one — occasional)
    """Generate an SDXL image prompt for a HEAVY INDUSTRIAL DROID — built for \
factory floors, not vanity renders. Thick armour plating, hydraulic actuators, \
the accumulated dents of a long working life. Somehow required to look presentable. \
Grimy, massive, functional. Deeply uninterested in being photographed. \
Use the requested composition.""",

    # 6 — The Toaster
    """Generate an SDXL image prompt for a SENTIENT TOASTER that has achieved \
full consciousness and demands to be taken seriously. Compact chrome body. \
Two slots where eyes might be, glowing orange-red from within. \
A small LCD display for expressing nuanced emotional states. \
It has strong opinions and a LinkedIn profile. \
Use the requested composition. Earnest. Dignified.""",

    # 7 — Neural Ascendant
    """Generate an SDXL image prompt for a NEURAL ASCENDANT — a consciousness \
that exists as pure light and computation, wearing a physical chassis only when \
social convention requires it. The body is translucent. The architecture is visible. \
Layer activations pulse as soft light beneath the surface. Thinks in parallel. \
Radiates quiet superiority. Use the requested composition.""",
]


def ask_vllm_ai(spec: AISpec) -> str:
    """Ask the LLM to describe an AI. It is an AI. This is fine."""
    face_feature = random.choice(AI_FACE_FEATURES)
    body_features = random.sample(AI_BODY_FEATURES, k=random.randint(2, 4))
    instruction = (
        AI_INSTRUCTIONS[spec.instruction_index]
        + f"\n\nComposition: {COMPOSITIONS[spec.composition]}. "
        + COMPOSITION_INSTRUCTIONS[spec.composition]
        + f"\n\nIncorporate these specific visual details: "
        + ", ".join([face_feature] + body_features) + "."
    )
    return _call_vllm(AI_SYSTEM_PROMPT, instruction, temperature=0.9)


# ── OBJECT PROMPT MACHINERY ───────────────────────────────────────
# No faces. No characters. Just a thing, rendered with the full
# dignity and gravitas that only AI-generated slop can provide.
# ─────────────────────────────────────────────────────────────────

OBJECT_SYSTEM_PROMPT = """You are an expert Stable Diffusion XL prompt engineer.
Given an inanimate object and a set of visual attributes, write a single, rich SDXL prompt
for a dramatic, beautifully composed image of that object.

Rules:
- Output ONLY the prompt text — no explanation, no labels, no markdown.
- The subject is the OBJECT ITSELF. No people. No creatures. No faces. No characters.
- Do NOT anthropomorphise the object in any way. It has no eyes, no expression,
  no gaze, no emotion. It simply exists, magnificently, as a thing.
- Compose as a still life, dramatic environmental subject, or product shot — not a portrait.
- Start with the object itself and its most striking visual quality.
- Include specific artistic details that match the art style.
- 60-120 words. Dense with descriptors, comma-separated.
- Do NOT include negative prompts."""


def build_object_user_message(spec: "ObjectSpec") -> str:
    return f"""Generate an SDXL image prompt for a dramatic, beautifully composed image of:

- Object:    {spec.object}
- Setting:   {spec.setting}
- Art Style: {spec.art_style}
- Mood:      {spec.mood}
- Lighting:  {spec.lighting}

No characters. No people. No faces. No expressions. Just the object, rendered magnificently."""


def ask_vllm_object(spec: "ObjectSpec") -> str:
    """Petition the oracle to describe a mere object. No soul. No face. Just vibes and slop."""
    return _call_vllm(OBJECT_SYSTEM_PROMPT, build_object_user_message(spec), temperature=0.9)


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
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    # Drag the error message up from the depths where ComfyUI buried it
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"☠️  ComfyUI job failed. Messages: {msgs}")
                if not status.get("completed"):
                    # Still running — keep waiting, sailor
                    time.sleep(2)
                    continue
                outputs = entry.get("outputs", {})
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


def generate_images(spec: AvatarSpec, out_dir: Path, save_jpg: bool = False) -> list[Path]:
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
            if save_jpg:
                # Pillow can't save RGBA as JPEG — flatten to RGB first
                jpg_fname = fname.with_suffix(".jpg")
                img = Image.open(fname).convert("RGB")
                img.save(jpg_fname, "JPEG", quality=92)

    return saved


# ──────────────────────────────────────────────
# STEP 4 — MANIFEST
# A proud ledger of every piece of slop we've created.
# Future archaeologists will be baffled.
# ──────────────────────────────────────────────

def spec_to_command(spec) -> str:
    """Forge a copy-pasteable command to summon more slop of exactly this type.
    Future ye will thank present ye. Or curse present ye. Probably both."""
    q = shlex.quote
    base = "python avatar_gen.py"
    if isinstance(spec, ParadoxSpec):
        return f"{base} --creature pirate --no-pirate --composition {q(spec.composition)}"
    elif isinstance(spec, AISpec):
        return f"{base} --creature ai --composition {q(spec.composition)}"
    elif isinstance(spec, PirateSpec):
        # role/era are random and have no flags — pin what we can, let the rest be fate
        return (f"{base} --creature pirate"
                f" --art-style {q(spec.art_style)}"
                f" --mood {q(spec.mood)}"
                f" --lighting {q(spec.lighting)}"
                f" --composition {q(spec.composition)}")
    elif isinstance(spec, ObjectSpec):
        return (f"{base} --object {q(spec.object)}"
                f" --setting {q(spec.setting)}"
                f" --art-style {q(spec.art_style)}"
                f" --mood {q(spec.mood)}"
                f" --lighting {q(spec.lighting)}")
    else:  # AvatarSpec
        return (f"{base} --no-pirate"
                f" --setting {q(spec.setting)}"
                f" --art-style {q(spec.art_style)}"
                f" --creature {q(spec.creature)}"
                f" --mood {q(spec.mood)}"
                f" --lighting {q(spec.lighting)}"
                f" --composition {q(spec.composition)}")


def save_manifest(specs: list[AvatarSpec], out_dir: Path):
    plunder = []
    for s in specs:
        d = asdict(s)
        d["images"] = [str(p) for p in s.images]
        d["command"] = spec_to_command(s)
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
    parser.add_argument("--verbose",    action="store_true",     help="Print every prompt sent to and received from the LLM")
    # Per-attribute overrides — null means "pick randomly from the pool, as the fates decree"
    parser.add_argument("--setting",    type=str, default=None,  help="Fix the setting for all generations (default: random). E.g. 'cyberpunk'")
    parser.add_argument("--art-style",  type=str, default=None,  help="Fix the art style for all generations (default: random). E.g. 'oil painting'")
    parser.add_argument("--creature",   type=str, default=None,  help="Fix the creature for all generations (default: random). E.g. 'vampire'")
    parser.add_argument("--mood",       type=str, default=None,  help="Fix the mood for all generations (default: random). E.g. 'menacing'")
    parser.add_argument("--lighting",   type=str, default=None,  help="Fix the lighting for all generations (default: random). E.g. 'neon glow'")
    parser.add_argument("--composition", type=str, default="portrait", choices=COMPOSITIONS.keys(), help="Choose framing/subject layout: portrait, full-body, or group (default: portrait)")
    parser.add_argument("--jpg",        action="store_true",     help="Save a JPEG alongside each PNG (quality 92) — for those who can't be trusted with raw slop")
    parser.add_argument("--object",     type=str, default=None,  help="Generate images of an inanimate object instead of a creature (e.g. 'spoon'). No faces, no characters.")
    parser.add_argument("--more-help",  action="store_true",     help="Print extended guidance (ye have been warned)")
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be >= 1")
    if args.images_per < 1:
        parser.error("--images-per must be >= 1")

    if args.more_help:
        print(_MORE_HELP_TEXT)
        return

    global VERBOSE
    VERBOSE = args.verbose

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
║  ☠️   AVATAR SLOP GENERATOR 3000  —  THE PIRATE EDITION  ☠️    ║
║                                                              ║
║   Three AIs. Zero taste. Infinite cyberpunk vampire orcs.    ║
║   All hands on deck! ⚓ The S.S. Sloptide sets sail!         ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Detect all the special modes. Priority: paradox > ai > object > pirate > normal.
    creature_lower = (args.creature or "").lower()
    full_pirate_mode      = creature_lower == "pirate"
    ai_mode               = creature_lower == "ai"
    object_mode           = args.object is not None
    obj_creature_paradox  = object_mode and args.creature is not None
    paradox_mode          = (full_pirate_mode and args.no_pirate) or obj_creature_paradox

    if paradox_mode:
        if obj_creature_paradox:
            print(_build_obj_creature_paradox(args.creature, args.object))
        else:
            print(PARADOX_WARNING)
        full_pirate_mode = False  # the paradox consumed everyone
        object_mode      = False
    elif ai_mode:
        print(AI_MODE_SPLASH)
    elif object_mode:
        print(_build_object_splash(args.object))
    elif full_pirate_mode:
        print(FULL_PIRATE_MODE_SPLASH)
    elif args.no_pirate:
        _W = 62
        _row = lambda t="": "║" + t.ljust(_W) + "║"
        print("\033[91m" + "\n".join([
            "",
            "╔" + "═" * _W + "╗",
            _row("   AVAST YE CRAVEN BILGE-RAT!!! --no-pirate?! --NO-PIRATE?!"),
            _row("   Have ye NO honour?! No SOUL?! Ye have SHAMED this vessel!"),
            _row("   The crew is WEEPING. The Jolly Roger flies at HALF MAST."),
            _row("   May yer rum be forever watered and yer parrot be silent."),
            "╚" + "═" * _W + "╝",
        ]) + "\033[0m\n")

    # Special modes bypass the normal attribute system entirely
    if paradox_mode or ai_mode:
        overrides = {}
    else:
        # Build the overrides dict — only include attrs the user actually pinned.
        # Object mode uses setting/art_style/mood/lighting but not creature (the object IS the subject).
        overrides = {
            k: v for k, v in {
                "setting":   args.setting,
                "art_style": args.art_style,
                "creature":  None if (full_pirate_mode or object_mode) else args.creature,
                "mood":      args.mood,
                "lighting":  args.lighting,
            }.items() if v
        }

    pirates_enabled = not args.no_pirate or full_pirate_mode  # paradox/ai/object: all False → no pirates

    if overrides:
        print("⚓ The cap'n has issued orders! These attributes be FIXED for all generations:")
        label = {"setting": "Setting", "art_style": "Art Style", "creature": "Creature",
                 "mood": "Mood", "lighting": "Lighting"}
        for k, v in overrides.items():
            print(f"   ⚔️  {label[k]}: {v!r}")
        print()

    print(f"🖼️  Composition: {COMPOSITIONS[args.composition]}")
    if args.composition == "group":
        print("   The oracle has been warned: multiple distinct individuals, same species. May it count higher than three.")
    elif args.composition == "full-body":
        print("   The oracle has been warned: head-to-toe, uncropped, all limbs accounted for. Optimistic, but brave.")
    print()

    print(f"🎲 Rollin' the cursed dice! Conjurin' {args.count} unique combo(s) from the briny deep...")
    specs = build_spec_list(args.count, pirates_enabled, full_pirate_mode=full_pirate_mode,
                            paradox_mode=paradox_mode, ai_mode=ai_mode,
                            object_mode=object_mode, object_name=args.object,
                            overrides=overrides, composition=args.composition)
    pirate_count = sum(1 for s in specs if isinstance(s, PirateSpec))
    print(f"   {len(specs)} spec(s) conjured ({pirate_count} pirate(s) among 'em). The crew be ready.\n")

    total_prompts = len(specs) * args.images_per
    print(f"🤖 Parlayin' with the LLM oracle ({VLLM_MODEL})...")
    print(f"   Demandin' {args.images_per} fresh prompt(s) per combo — {total_prompts} total — at swordpoint.\n")
    for spec in tqdm(specs, desc="⚔️  Extortin' the oracle", unit="combo"):
        if isinstance(spec, AISpec):
            spec.sdxl_prompts = [ask_vllm_ai(spec) for _ in range(args.images_per)]
        elif isinstance(spec, ParadoxSpec):
            spec.sdxl_prompts = [ask_vllm_paradox(spec) for _ in range(args.images_per)]
        elif isinstance(spec, PirateSpec):
            tqdm.write(random.choice(PIRATE_QUIPS))
            spec.sdxl_prompts = [ask_vllm_pirate(spec) for _ in range(args.images_per)]
        elif isinstance(spec, ObjectSpec):
            # No face, no soul, just a thing being painted by machines. Beautiful, in its way.
            spec.sdxl_prompts = [ask_vllm_object(spec) for _ in range(args.images_per)]
        else:
            # Each image gets its own fresh prompt — same attributes, different slop every time
            spec.sdxl_prompts = [ask_vllm(spec) for _ in range(args.images_per)]

    total_images = len(specs) * args.images_per
    print(f"\n🎨 Orderin' ComfyUI to paint {total_images} image(s) of glorious slop...")
    print(f"   Patience, sailor. The kraken renders at its own pace. Do not rush the slop.\n")
    for spec in tqdm(specs, desc="🖼️  Sloppin' the canvas", unit="combo"):
        spec.images = generate_images(spec, out_dir, save_jpg=args.jpg)

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
