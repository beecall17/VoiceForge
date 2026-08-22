---
title: VoiceForge
emoji: 🧶
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: QLoRA + DPO brand-voice copywriter for micro-artisans
---

# VoiceForge

A QLoRA-fine-tuned, DPO-aligned brand-voice copywriter for two micro-artisan
personas: **Cozy Crochet** and **Romantic Floral**. Compare base model
output against SFT and SFT+DPO fine-tunes on the same brief, side by side.

**Important:** this Space requires **ZeroGPU hardware** (Settings ->
Space hardware -> ZeroGPU) to run -- the 4-bit quantized 7B model needs a
real CUDA device to load and generate. It will not work on CPU Basic.

## What's under the hood

- Base model: `Qwen/Qwen2.5-7B-Instruct`, loaded in 4-bit (QLoRA/NF4)
- SFT adapter: `bikalpoudel/voiceforge-brand-voice-sft-lora`
- DPO adapter: `bikalpoudel/voiceforge-brand-voice-dpo-lora` (continues
  from the SFT weights via preference alignment)
- Full pipeline (dataset curation, SFT, DPO, evaluation) documented in the
  project repo.

See the "Training & evaluation" tab in the app for eval numbers and
training curves.
