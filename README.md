# VoiceForge

**A two-voice brand-copywriting LLM, fine-tuned end-to-end with QLoRA SFT + DPO on a free Colab GPU.**

🔗 **[Live demo](https://huggingface.co/spaces/bikalpoudel/VoiceForge)** — try it yourself, no setup needed.

---

## What this is

Micro-artisans (handmade crochet sellers, custom bouquet makers) spend more time writing
Instagram captions and TikTok hooks than they'd like, and generic LLM output reads
corporate and kills the warm, handmade feel that actually drives their sales.

VoiceForge fine-tunes `Qwen2.5-7B-Instruct` into a single adapter that writes on-brand
copy for **two distinct personas** — Cozy Crochet and Romantic Floral — routed via a
`[Voice: ...]` tag trained directly into the model's weights, not swapped via a system
prompt at inference time. The pipeline runs a full two-stage alignment process:
**Supervised Fine-Tuning** to teach voice and format structure, then **Direct Preference
Optimization** to sharpen rule adherence (banned phrases, CTA rules, length targets)
beyond what SFT alone achieved — all trained on a free Google Colab T4 GPU via 4-bit
QLoRA.

## Results

Evaluated on 25 held-out test prompts never seen during SFT or DPO training.

| Variant | Rule-clean rate | Tone | Vocabulary | CTA correctness | Concrete detail |
|---|---|---|---|---|---|
| Base (no fine-tuning) | 92% | 4.08 | 4.84 | 4.72 | 3.16 |
| SFT | 100% | 4.80 | 5.00 | 4.88 | 4.08 |
| SFT + DPO | 96% | 4.48 | 4.92 | 4.72 | 3.40 |

**Pairwise preference (LLM-as-judge, position-randomized): SFT+DPO wins 60%** of
head-to-head comparisons against SFT-only (SFT wins 36%, ties 4%).

### An honest finding, not just a headline number

DPO wins more pairwise match-ups — largely on stricter format/length discipline, with
several judge rationales specifically citing word-count and CTA-rule compliance. But it
scores *lower* than SFT-only on every absolute rubric dimension, most notably **concrete
detail** (3.40 vs. 4.08).

The likely cause: Day 8's DPO preference pairs were labeled against banned-phrase, CTA,
and length rules only — never against detail-richness. DPO reliably learned exactly what
it was shown a preference for, and nothing else. It's a useful, concrete illustration
that preference data shapes precisely what gets optimized: DPO isn't a general "make it
better" dial, it's as targeted as the pairs you feed it.

## Architecture

```
Base model (Qwen2.5-7B-Instruct, 4-bit NF4)
        │
        ▼
   QLoRA SFT ──────────► bikalpoudel/voiceforge-brand-voice-sft-lora
        │
        ▼
   QLoRA DPO ──────────► bikalpoudel/voiceforge-brand-voice-dpo-lora
        │
        ▼
Gradio app on HF Spaces (ZeroGPU) — base vs. SFT vs. SFT+DPO, side by side
```

- **Base model:** [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (ungated, 4-bit QLoRA)
- **SFT dataset:** [`bikalpoudel/voiceforge-brand-voice-sft`](https://huggingface.co/datasets/bikalpoudel/voiceforge-brand-voice-sft)
- **DPO dataset:** [`bikalpoudel/voiceforge-brand-voice-dpo`](https://huggingface.co/datasets/bikalpoudel/voiceforge-brand-voice-dpo)
- **Training:** `trl` (`SFTTrainer`, `DPOTrainer`) + `peft` + `bitsandbytes`, entirely on a free Colab T4
- **Deployment:** Gradio on Hugging Face Spaces, ZeroGPU hardware

## Repository structure

```
voiceforge/
├── README.md                 <- you are here
├── docs/
│   ├── problem_statement.md
│   ├── voice_guidelines.md   <- the rubric used for teacher generation, DPO labeling, and eval
│   └── eval_results.md
├── scripts/                  <- data generation, cleaning, labeling, dataset publishing
├── notebooks/                <- Colab notebooks: SFT setup/training, DPO candidate gen/training, eval
├── data/                     <- intermediate .jsonl artifacts (raw prompts, cleaned/split SFT data, DPO pairs)
├── app/                    <- Gradio app deployed to HF Spaces
    └── images/                   <- training curves (loss, DPO reward margin)
```

## Pipeline

| Stage | What happens | Where |
|---|---|---|
| Voice design | Two full brand-voice specs: tone, vocabulary, banned phrases, emoji/CTA/length rules per format | `docs/voice_guidelines.md` |
| Brief generation | Combinatorial (voice × product × format × angle), 250 unique, voice-balanced | `scripts/generate_briefs.py` |
| Teacher generation | Gemini API, persona-specific system prompt parsed straight from the guidelines doc | `scripts/generate_completions.py` |
| Cleaning & splitting | Automated banned-phrase/CTA/length checks + a human review pass, stratified 80/10/10 split | `scripts/clean_and_split.py` |
| SFT | QLoRA on the cleaned split | `notebooks/day5_sft_colab_setup.ipynb`, `day6_sft_full_run.ipynb` |
| DPO candidates | Two completions per brief at different sampling temperatures, from the SFT model itself | `notebooks/day7_generate_dpo_candidates.ipynb` |
| DPO labeling | Two-tier: rule-based first, LLM-judge fallback only when rules tie; near-duplicates and both-flawed pairs discarded | `scripts/label_dpo_pairs.py` |
| DPO training | Continues from the SFT adapter's weights | `notebooks/day9_dpo_training.ipynb` |
| Evaluation | 3-way generation (base/SFT/SFT+DPO) + pairwise win-rate + rubric scoring | `notebooks/day10_generate_eval_outputs.ipynb`, `scripts/evaluate_and_report.py` |
| Deployment | Gradio app, both adapters loaded as switchable PEFT adapters on one base model | `app/app.py` |

## Reproducing this

```bash
git clone <this repo>
cd voiceforge
cp .env.example .env   # fill in GEMINI_API_KEY and HF_TOKEN
pip install -r app/requirements.txt

python scripts/generate_briefs.py
python scripts/generate_completions.py
python scripts/clean_and_split.py
python scripts/clean_and_split.py --exclude_ids_file data/manual_rejects.txt
python scripts/push_dataset_to_hub.py --repo_id <your-username>/voiceforge-brand-voice-sft
# then run notebooks/day5 and day6 in Colab (GPU required)

# after SFT: generate + label DPO pairs
# run notebooks/day7 in Colab, then:
python scripts/label_dpo_pairs.py
python scripts/push_dpo_dataset_to_hub.py --repo_id <your-username>/voiceforge-brand-voice-dpo
# then run notebooks/day9 in Colab

# evaluation
# run notebooks/day10_generate_eval_outputs.ipynb in Colab, then:
python scripts/evaluate_and_report.py
```

## What I'd do differently

- Add "includes a concrete detail" as an explicit rule-based check in Day 8's labeling,
  not just a rubric-scoring afterthought — would likely close the gap DPO opened there.
- The DPO preference set (~215 pairs after cleaning) is small; a second candidate-generation
  pass with a wider temperature gap would produce more decisive, less ambiguous pairs.
- Try `beta` sweep (0.05 vs 0.1 vs 0.2) — only tested one KL-penalty setting.

## License

Educational/portfolio project. Base model and its license terms belong to Qwen; this
repo's code is MIT unless noted otherwise.