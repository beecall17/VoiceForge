# VoiceForge: AI Marketing Assistant for Micro-Artisans

## 1. Executive Summary
**VoiceForge** is a domain-adapted, multi-voice LLM fine-tuned to generate authentic, social-first marketing copy for small-scale handmade businesses. Utilizing a two-stage alignment pipeline (**Supervised Fine-Tuning + Direct Preference Optimization**), the model learns to avoid generic "AI-sounding" marketing jargon and instead adopt controllable, niche brand personalities—specifically designed for handmade crochet artisans and custom bouquet makers.

---

## 2. Real-World Motivation & Local Context
Local micro-artisans—such as handmade crochet creators (*@moonscrochets*, *@mimiyyy_12*) and custom bouquet designers (*@vistavibes816*)—face a fundamental operational bottleneck: **content marketing fatigue**.

* **The Skill Gap:** Artisans spend 80%+ of their time crafting physical, high-effort goods by hand. Writing engaging Instagram captions, TikTok video hooks, and promotional copy daily is mentally draining.
* **The "Generic AI" Failure:** Off-the-shelf LLMs (e.g., default ChatGPT) output corporate, overly formal copy (*"Elevate your gifting experience with our exquisite handcrafted arrangement!"*). This destroys the warm, cozy, authentic personal connection that drives small-business sales.
* **Resource Constraints:** Micro-businesses operate on thin margins and cannot afford dedicated marketing agencies or copywriters.

---

## 3. The Technical Solution
Rather than relying on verbose prompt engineering, **VoiceForge** fine-tunes a small open-weights model (`Llama-3.1-8B-Instruct` / `Qwen2.5-7B-Instruct`) to embed controllable, steerable brand personalities directly into the model weights using **QLoRA**.

### Key Technical Architecture:
1. **Multi-Voice Conditioning:** The model learns to route tone and style based on explicit voice tags in the input prompt:
   * `[Voice: Cozy Crochet]` $\rightarrow$ Soft, playful, warm, wholesome (yarn puns, squishy desk-buddy vibes).
   * `[Voice: Romantic Floral]` $\rightarrow$ Aesthetic, sentimental, emotional, gift-centric (everlasting blooms, memory preservation).
2. **Two-Stage Fine-Tuning Pipeline:**
   * **Stage 1 (SFT):** Teaches the model the structural syntax of social captions, short hooks, and product descriptions across artisan categories.
   * **Stage 2 (DPO):** Directly aligns the model using preferred vs. rejected completion pairs to penalize corporate buzzwords ("game-changer", "elevate", "unlock") and reward authentic human-like phrasing.
3. **Low-Compute Infrastructure:** Fully trained on free Google Colab T4 GPUs using 4-bit quantization (`bitsandbytes`), parameter-efficient fine-tuning (`peft`), and `DPOTrainer` (`trl`).

---

## 4. Expected Business & Engineering Impact
* **For the Business Owner:** Reduces copywriting time from 30+ minutes per post to under 10 seconds, delivering ready-to-post, platform-formatted social copy.
* **For the ML Portfolio:** Proves end-to-end AI engineering competence—from ground-truth dataset curation and QLoRA tuning to preference alignment (DPO), quantitative evaluation, and public deployment on Hugging Face Spaces.