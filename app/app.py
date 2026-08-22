"""
VoiceForge -- Gradio demo for Hugging Face Spaces (ZeroGPU hardware).

Loads base model once with both the SFT and DPO adapters attached as
named, switchable adapters (same pattern as Day 10's eval notebook), lets
the visitor build an in-distribution brief via the same voice/format/
product/angle structure used throughout training, and generates all 3
variants (base / SFT / SFT+DPO) side by side.

Requires the Space's hardware set to ZeroGPU in Space settings -- the
@spaces.GPU decorator only does anything meaningful there. On CPU-only
hardware this will either fail to load (bitsandbytes 4-bit needs CUDA) or
run too slowly to be usable; see the "if no GPU" note near the bottom.
"""

import random

import gradio as gr
import spaces
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SFT_ADAPTER_REPO = "bikalpoudel/voiceforge-brand-voice-sft-lora"
DPO_ADAPTER_REPO = "bikalpoudel/voiceforge-brand-voice-dpo-lora"

# Copied verbatim from voice_guidelines.md / Day 6-10 notebooks -- keep in
# sync if the guidelines change, since generation quality depends on using
# the exact system prompt the models were trained against.
DISTILLED_SYSTEM_PROMPTS = {
    "cozy_crochet": (
        "You are a copywriter for a cozy handmade crochet shop. Voice: genuine, "
        "cozy, cute, simple, thoughtful. Short sentences, contractions and "
        "fragments are fine. Emoji: choose from \U0001F9F6 \U0001F338 \u2728 \U0001F49B, max 1-2 per post. Never say: "
        "elevate, premium, luxurious, game-changer, exclusive drop. Include one "
        "concrete detail (size, turnaround time, or material) when the format "
        "calls for it. Output only the copy itself, no labels or quotation marks."
    ),
    "romantic_floral": (
        "You are a copywriter for a romantic handmade bouquet shop. Voice: "
        "sentimental, heartfelt, aesthetic, warm, custom-focused. Emoji: choose "
        "from \U0001F490 \U0001F380 \U0001F48C \u2728, max 1-2 per post. Never say: unlock, revolutionary, "
        "unrivaled, cheap, bulk, flash sale. Include one concrete detail (size, "
        "turnaround time, or materials) when the format calls for it. Output "
        "only the copy itself, no labels or quotation marks."
    ),
}

# Copied from scripts/generate_briefs.py, so briefs built here stay
# in-distribution with what the models were actually trained on.
VOICE_LABELS = {"cozy_crochet": "Cozy Crochet", "romantic_floral": "Romantic Floral"}

PRODUCTS = {
    "cozy_crochet": [
        "crochet keychain", "crochet scarf", "crochet flower pot",
        "amigurumi plushie", "chunky lapghan blanket", "crochet coaster set",
        "crochet bag charm", "crochet cardigan", "custom reference order",
    ],
    "romantic_floral": [
        "ribbon bouquet", "pipe-cleaner bouquet", "money bouquet",
        "handwritten-note gift pack", "anniversary bouquet special",
        "graduation bouquet special", "Valentine's Day bouquet",
        "Mother's Day bouquet", "sympathy arrangement",
        "wedding/proposal bouquet", "teacher appreciation gift",
        "custom reference order",
    ],
}

FORMATS = {
    "ig_caption": {"label": "an Instagram caption", "length_hint": "20-45 words, 2-4 sentences", "cta": True, "extra": ""},
    "tiktok_hook": {
        "label": "a TikTok video hook (opening line only)", "length_hint": "under 10 words, 1 line", "cta": False,
        "extra": "This is only the first line viewers see before the video plays -- pure scroll-stopper, not a full post.",
    },
    "tiktok_caption": {"label": "a TikTok caption", "length_hint": "15-30 words", "cta": "optional", "extra": ""},
    "etsy_listing": {
        "label": "an Etsy-style product listing description", "length_hint": "40-70 words, 3-5 sentences", "cta": True,
        "extra": "Must include at least one concrete, practical detail (turnaround time, size, material, or care note).",
    },
    "promo_announcement": {
        "label": "a story/promo announcement", "length_hint": "10-20 words", "cta": "optional",
        "extra": "Time-sensitive framing, but not salesy or hypey.",
    },
    "restock_announcement": {
        "label": "a restock/drop announcement", "length_hint": "15-30 words", "cta": True,
        "extra": "Frame it as 'back' or 'new'.",
    },
}
FORMAT_LABELS = {
    "ig_caption": "Instagram Caption",
    "tiktok_hook": "TikTok Hook",
    "tiktok_caption": "TikTok Caption",
    "etsy_listing": "Etsy Listing",
    "promo_announcement": "Promo Announcement",
    "restock_announcement": "Restock Announcement",
}

ANGLES = [
    "new design reveal", "restock of a bestseller", "custom order / reference-photo feature",
    "behind-the-scenes of how it's made", "gift guide for an upcoming occasion",
    "customer feature / happy customer story", "care tip or fun fact about the piece",
    "limited custom-order slots reminder",
]

CTA_INSTRUCTION = {
    True: "Include a natural CTA encouraging DMs for orders or custom requests.",
    False: "Do NOT include any CTA -- this format has none.",
    "optional": "A light CTA is okay here if it fits naturally, but isn't required.",
}


def article_for(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def build_brief(voice_key, format_key, product, angle):
    """Same template as scripts/generate_briefs.py's make_brief -- kept
    in sync so demo inputs stay in-distribution with training data."""
    fmt = FORMATS[format_key]
    voice_tag = f"[Voice: {VOICE_LABELS[voice_key]}]"
    art = article_for(product)
    parts = [
        voice_tag,
        f"Write {fmt['label']} ({fmt['length_hint']}) for {art} {product}.",
        f"Campaign angle: {angle}.",
        CTA_INSTRUCTION[fmt["cta"]],
    ]
    if fmt["extra"]:
        parts.append(fmt["extra"])
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Model loading -- LAZY, inside the GPU-decorated call, not at module level.
#
# ZeroGPU only attaches a real CUDA device to this process for the duration
# of an @spaces.GPU-decorated function call. Loading a bitsandbytes 4-bit
# model at module level (outside that context) fails with
# "RuntimeError: No CUDA GPUs are available", because quantization during
# from_pretrained needs an actual GPU that isn't attached yet at import
# time. Cached in globals so this only happens once -- the Space's Python
# process persists between calls, only GPU attachment is per-call.
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    print("Loading tokenizer and base model (first call only)...")
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    _model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_REPO, adapter_name="sft")
    _model.load_adapter(DPO_ADAPTER_REPO, adapter_name="dpo")
    _model.eval()
    _model.config.use_cache = True
    print("Model + both adapters loaded.")
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Generation (GPU-decorated -- this and _load_model above are the only
# things that ever touch CUDA)
# ---------------------------------------------------------------------------

@spaces.GPU(duration=120)  # generous first-call budget: cold model load + quantization + 3 generations
def generate_all_variants(voice_key, brief, max_new_tokens=150):
    model, tokenizer = _load_model()

    messages = [{"role": "system", "content": DISTILLED_SYSTEM_PROMPTS[voice_key]}, {"role": "user", "content": brief}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    def run():
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.8, top_p=0.9, pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    with model.disable_adapter():
        base_out = run()
    model.set_adapter("sft")
    sft_out = run()
    model.set_adapter("dpo")
    dpo_out = run()

    return base_out, sft_out, dpo_out


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def on_generate(voice_display, format_display, product, angle):
    voice_key = {v: k for k, v in VOICE_LABELS.items()}[voice_display]
    format_key = {v: k for k, v in FORMAT_LABELS.items()}[format_display]
    if not product.strip():
        raise gr.Error("Enter a product (e.g. 'crochet scarf' or 'ribbon bouquet').")

    brief = build_brief(voice_key, format_key, product.strip(), angle)
    base_out, sft_out, dpo_out = generate_all_variants(voice_key, brief)
    return brief, base_out, sft_out, dpo_out


def on_random_example():
    voice_key = random.choice(list(VOICE_LABELS))
    format_key = random.choice(list(FORMATS))
    product = random.choice(PRODUCTS[voice_key])
    angle = random.choice(ANGLES)
    return VOICE_LABELS[voice_key], FORMAT_LABELS[format_key], product, angle


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

EVAL_SUMMARY_MD = """\
### Evaluation summary (25 held-out test prompts, never seen in training)

| Variant | Rule-clean rate | Tone | Vocabulary | CTA correctness | Concrete detail |
|---|---|---|---|---|---|
| Base | 92% | 4.08 | 4.84 | 4.72 | 3.16 |
| SFT | 100% | 4.80 | 5.00 | 4.88 | 4.08 |
| SFT + DPO | 96% | 4.48 | 4.92 | 4.72 | 3.40 |

**Pairwise preference (LLM-as-judge, position-randomized): SFT+DPO wins 60%** of head-to-head
comparisons against SFT-only (SFT wins 36%, ties 4%).

Worth noting honestly: DPO wins more pairwise match-ups (largely on stricter format/length
discipline -- several judge rationales cite word-count and CTA-rule compliance specifically),
but scores slightly *lower* than SFT-only on every absolute rubric dimension, most notably
**concrete detail** (3.40 vs. 4.08). The likely explanation: DPO's preference pairs were labeled
against banned-phrase/CTA/length rules only, never against detail-richness -- so DPO reliably
learned what it was actually shown a preference for, and nothing else. A good illustration that
preference data shapes exactly what gets optimized.
"""

with gr.Blocks(title="VoiceForge") as demo:
    gr.Markdown(
        "# VoiceForge\n"
        "A QLoRA + DPO fine-tuned brand-voice copywriter for two micro-artisan personas: "
        "**Cozy Crochet** and **Romantic Floral**. Pick a voice, format, and product, and "
        "compare the base model against the SFT and SFT+DPO fine-tunes on the same brief."
    )

    with gr.Tab("Try it"):
        with gr.Row():
            with gr.Column(scale=1):
                voice_input = gr.Radio(list(VOICE_LABELS.values()), label="Voice", value="Cozy Crochet")
                format_input = gr.Dropdown(list(FORMAT_LABELS.values()), label="Format", value="Instagram Caption")
                product_input = gr.Textbox(label="Product", placeholder="e.g. crochet scarf")
                angle_input = gr.Dropdown(ANGLES, label="Campaign angle", value=ANGLES[0])
                with gr.Row():
                    example_btn = gr.Button("🎲 Random example")
                    generate_btn = gr.Button("Generate", variant="primary")
                brief_display = gr.Textbox(label="Constructed brief (what the model actually sees)", interactive=False)

            with gr.Column(scale=2):
                base_output = gr.Textbox(label="Base (no fine-tuning)", lines=4)
                sft_output = gr.Textbox(label="SFT (voice fine-tuned)", lines=4)
                dpo_output = gr.Textbox(label="SFT + DPO (preference aligned)", lines=4)

        example_btn.click(on_random_example, outputs=[voice_input, format_input, product_input, angle_input])
        generate_btn.click(
            on_generate,
            inputs=[voice_input, format_input, product_input, angle_input],
            outputs=[brief_display, base_output, sft_output, dpo_output],
        )

    with gr.Tab("Training & evaluation"):
        gr.Markdown(EVAL_SUMMARY_MD)
        with gr.Row():
            gr.Image("images/sft_loss_curve.png", label="SFT training loss", show_label=True)
            gr.Image("images/dpo_loss_curve.png", label="DPO training loss", show_label=True)
        gr.Image("images/dpo_reward_margin.png", label="DPO reward margin (chosen - rejected)", show_label=True)

if __name__ == "__main__":
    demo.launch()