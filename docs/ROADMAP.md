# VoiceForge: 12-Day Execution Roadmap

> **Project Goal:** Fine-tune and align a 7B/8B LLM using a two-stage pipeline (SFT $\rightarrow$ DPO) to strictly enforce a target brand voice, evaluating alignment improvements and deploying an interactive demo to Hugging Face Spaces.

---

## Roadmap at a Glance

| Phase | Days | Focus Area | Total Est. Time | Key Deliverable |
| :--- | :---: | :--- | :---: | :--- |
| **Phase 1** | Days 1–6 | Voice Guideline, Dataset Curation, & QLoRA SFT | **16.5 Hours** | SFT Adapter on Hugging Face Hub |
| **Phase 2** | Days 7–12 | DPO Alignment, Evaluation, & HF Deployment | **17.0 Hours** | Live Interactive Gradio Space |

---

## Phase 1: Data Curation & Supervised Fine-Tuning (SFT)

> **Phase Focus:** Establish a ground-truth voice rubric, generate high-quality dataset pairs, set up the QLoRA pipeline on Google Colab, and execute the base SFT run.

### - Day 1: Brand Voice Specification
* **Estimated Time:** 2.0 Hours
* **Objective:** Concretely define the target brand persona to serve as the evaluation rubric for all downstream tasks.
* **Key Tasks:**
  * Select a distinctive persona (e.g., *Witty, concise Gen-Z DTC skincare brand* OR *Authoritative, technical Enterprise B2B SaaS*).
  * Draft a 1-page voice guideline defining:
    * Core tone words & personality traits.
    * Sentence length preferences & syntactic rules.
    * Banned buzzwords and forbidden phrases.
    * 3–5 hand-crafted exemplar copy snippets.
* **Deliverable:** `voice_guidelines.md`

---

### - Day 2: Campaign Brief Synthesis (SFT Prompts)
* **Estimated Time:** 3.0 Hours
* **Objective:** Build a diverse set of prompts to ensure style generalization across different formats.
* **Key Tasks:**
  * Write or synthesize 150–300 diverse campaign briefs covering multiple marketing formats (product launches, email subject lines, social captions, ad copy, taglines).
  * Vary products, industries, and angles within your target voice to prevent vocabulary overfitting.
* **Deliverable:** `raw_prompts.jsonl`

---

### - Day 3: Target Completion Generation (SFT Labels)
* **Estimated Time:** 3.0 Hours
* **Objective:** Generate ground-truth "on-voice" target completions for the SFT dataset.
* **Key Tasks:**
  * Prompt a strong teacher model (e.g., `GPT-4o-mini` or `Claude-3.5-Sonnet` API) using `voice_guidelines.md` as the system context. -- Used `gemini-api-key` for using free key.
  * Generate 1 gold-standard completion for each synthesized campaign brief.
* **Deliverable:** `sft_raw.jsonl` (~150–300 prompt-completion pairs)

---

### - Day 4: Data Curation, Cleaning, & Splitting
* **Estimated Time:** 2.5 Hours
* **Objective:** Enforce strict dataset quality through manual filtering and structural formatting.
* **Key Tasks:**
  * Manually audit $\sim 20\%$ of completions against `voice_guidelines.md`; rewrite or purge off-voice completions.
  * Deduplicate near-identical briefs and filter malformed tokens.
  * Format into standard instruction/output JSON schema.
  * Perform an 80 / 10 / 10 train / validation / test split.
* **Deliverables:** `sft_train.jsonl`, `sft_val.jsonl`, `sft_test.jsonl`

---

### - Day 5: Environment Setup & QLoRA Smoke Test
* **Estimated Time:** 3.0 Hours
* **Objective:** Configure the Colab environment and verify memory constraints before the full training run.
* **Key Tasks:**
  * Configure Google Colab environment (`transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`).
  * Load `Llama-3.1-8B-Instruct` (or `Qwen2.5-7B-Instruct`) in 4-bit NF4 precision.
  * Set up QLoRA configuration ($r=16, \alpha=32$, targeting attention and MLP projections).
  * Run a 5-step smoke test loop to verify gradient accumulation and ensure no CUDA Out-Of-Memory (OOM) errors occur.
* **Deliverable:** Fully functional training script (`train_sft.py` or Colab Notebook)

---

### - Day 6: Full SFT Run & Adapter Export
* **Estimated Time:** 3.0 Hours
* **Objective:** Train the base SFT model and verify output quality on held-out test prompts.
* **Key Tasks:**
  * Run full SFT for 2 epochs, saving checkpoints directly to Google Drive.
  * Export the trained QLoRA adapter weights.
  * Perform qualitative sanity checks on held-out test prompts (`sft_test.jsonl`).
  * Push adapter weights and tokenizer configuration to the Hugging Face Hub.
* **Deliverables:** 
  * `sft-adapter` repository on Hugging Face Hub
  * Training checkpoints saved on Google Drive

---

## Phase 2: Preference Alignment (DPO), Evaluation, & Deployment

> **Phase Focus:** Collect A/B candidate completions, construct a DPO preference dataset, align the model using `DPOTrainer`, benchmark performance, and deploy an interactive demo.

### - Day 7: DPO Candidate Sampling
* **Estimated Time:** 2.5 Hours
* **Objective:** Sample multiple candidate responses per prompt from the SFT model to create preference pairs.
* **Key Tasks:**
  * Select a subset of prompts ($\sim 150\text{--}250$).
  * Inference the SFT adapter using different sampling settings (e.g., varying temperature $T=0.3$ vs. $T=0.9$, top-$p$) to produce candidate pair A and candidate pair B per prompt.
* **Deliverable:** `dpo_candidates.jsonl`

---

### - Day 8: Preference Pair Labeling (`chosen` vs. `rejected`)
* **Estimated Time:** 3.0 Hours
* **Objective:** Label candidate pairs to create the preference dataset required for DPO.
* **Key Tasks:**
  * Score A/B completions against `voice_guidelines.md` using an LLM-as-a-Judge script (or manual review).
  * Classify candidate pairs into `chosen` (more on-voice) and `rejected` (off-voice / generic).
  * Discard ambiguous pairs where both completions are equally good or bad (noisy pairs degrade DPO stability).
* **Deliverable:** `dpo_pairs.jsonl` formatted with `(prompt, chosen, rejected)`

---

### - Day 9: Direct Preference Optimization (DPO) Training
* **Estimated Time:** 3.0 Hours
* **Objective:** Align the SFT adapter using Direct Preference Optimization to maximize target voice features.
* **Key Tasks:**
  * Initialize `DPOTrainer` from `trl`, resuming from the Stage 1 SFT adapter.
  * Configure hyperparameters: $\beta = 0.1$, low learning rate ($5 \times 10^{-6} \text{ to } 1 \times 10^{-5}$), 1 epoch.
  * Monitor VRAM usage carefully (DPO maintains both policy and reference model representations).
  * Save and export the finalized DPO adapter.
* **Deliverable:** `dpo-adapter` repository pushed to Hugging Face Hub

---

### - Day 10: Multi-Stage Evaluation & Metric Collection
* **Estimated Time:** 2.5 Hours
* **Objective:** Quantify performance improvements across Base, SFT, and SFT+DPO model states.
* **Key Tasks:**
  * Run inference across `sft_test.jsonl` for three model variants: Base Model, SFT Adapter, and SFT+DPO Adapter.
  * Compute LLM-as-a-Judge pairwise win-rate (% of times SFT+DPO is preferred over SFT-only).
  * Score outputs on a 1–5 rubric scale across voice-guideline dimensions (Tone, Sentence Structure, Banned-Word Avoidance).
  * Collect 4–6 qualitative side-by-side output examples for documentation.
* **Deliverable:** `eval_results.md` with metric summary tables and qualitative comparisons

---

### - Day 11: Interactive Gradio Application Development
* **Estimated Time:** 3.0 Hours
* **Objective:** Build a local user interface enabling side-by-side model outputs for live demonstration.
* **Key Tasks:**
  * Develop a local `app.py` script using Gradio or Streamlit.
  * Create UI components:
    * Campaign brief text input box.
    * Temperature / length adjustment controls.
    * Side-by-side generation view (Base vs. SFT vs. SFT+DPO).
  * Implement inference loading via `peft` and `bitsandbytes`.
* **Deliverable:** Fully functional local `app.py`

---

### - Day 12: HF Spaces Deployment & Repository Finalization
* **Estimated Time:** 3.0 Hours
* **Objective:** Deploy the interactive application to Hugging Face Spaces and finalize public documentation.
* **Key Tasks:**
  * Deploy `app.py` to Hugging Face Spaces (CPU Basic or ZeroGPU).
  * Verify cold-start reliability and latency.
  * Finalize the public `README.md` featuring:
    * Headline metric (e.g., "73% Pairwise Win-Rate over SFT Baseline").
    * System architecture diagram (SFT $\rightarrow$ DPO workflow).
    * Link to live Hugging Face Space.
  * Record a short 30-second screen capture/GIF for resume and social portfolio sharing.
* **Deliverables:**
  * Live Hugging Face Space
  * Finalized public `README.md`
  * 30-second demo recording (`demo.gif`)

---

## Daily Execution Tracker

- [x] **Day 1:** `voice_guidelines.md` created
- [x] **Day 2:** `raw_prompts.jsonl` generated
- [x] **Day 3:** `sft_raw.jsonl` synthesized
- [ ] **Day 4:** Data cleaned and split into `sft_{train,val,test}.jsonl`
- [ ] **Day 5:** Colab notebook verified with 5-step smoke test
- [ ] **Day 6:** SFT run completed; `sft-adapter` pushed to HF Hub
- [ ] **Day 7:** `dpo_candidates.jsonl` sampled
- [ ] **Day 8:** `dpo_pairs.jsonl` labeled and filtered
- [ ] **Day 9:** DPO run completed; `dpo-adapter` pushed to HF Hub
- [ ] **Day 10:** `eval_results.md` generated with pairwise win-rates
- [ ] **Day 11:** Local `app.py` tested
- [ ] **Day 12:** HF Space live; `README.md` updated with demo link