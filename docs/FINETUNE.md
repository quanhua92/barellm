# Spinoff Phase — Context-Insufficiency Refusal Fine-Tuning

This spinoff project outlines the preparation, training, and validation pipeline to teach **Qwen3-0.6B** to reliably refuse (`"No"`) when a question cannot be answered using the provided context, preventing hallucinations while maintaining high QA accuracy when context *is* sufficient.

---

## Objective

1. **Reduce Hallucinations:** Train the model to identify when context is insufficient and return a clean refusal response (`"No"`).
2. **Preserve Capability:** Ensure accuracy on answerable questions does not degrade.
3. **Local Execution:** Validate that the entire workflow (data prep, training, validation, export) runs locally on a single GPU setup (RTX 4060 16GB).

---

## Phase A — Dataset Preparation & Curation

**Goal:** Build a balanced, clean instruction-following dataset for refusal behavior.

- [ ] **Data Sourcing:**
  - Retrieve and parse the Hugging Face [SQuAD 2.0 (`rajpurkar/squad_v2`)](https://huggingface.co/datasets/rajpurkar/squad_v2) dataset.
- [ ] **Data Formatting:**
  - Convert selected entries into the ChatML schema compatible with Qwen:
    ```
    <|im_start|>system
    Use only the provided context to answer the question. If the information is not present in the context, reply with 'No'.<|im_end|>
    <|im_start|>user
    Context: {context}
    
    Question: {question}<|im_end|>
    <|im_start|>assistant
    {target_output}<|im_end|>
    ```
- [ ] **Data Balancing:**
  - Build a curated split containing exactly 50% answerable questions (where `target_output` is the answer) and 50% unanswerable questions (where `target_output` is strictly `"No"`).
- [ ] **Splits Creation:**
  - Partition the dataset: **800 pairs for training** and **200 pairs for validation**.

---

## Phase B — Baseline Benchmarking

**Goal:** Establish evaluation metrics on the unmodified model.

- [ ] **Evaluation Script:**
  - Build a validation runner that feeds the 200 validation prompts to the base model.
- [ ] **Using HF Evaluate:**
  - Use Hugging Face's official [evaluate](https://github.com/huggingface/evaluate) library to load SQuAD v2 metrics (`evaluate.load("squad_v2")`).
  - **Mapping Outputs:** Map the model's textual outputs back to the SQuAD evaluation format:
    - If the model responds with `"No"`, map the prediction text to `""` (empty string, which represents "no answer" in SQuAD v2).
    - Otherwise, pass the generated text directly.
- [ ] **Define Metrics:**
  - **Exact Match (EM) & F1:** Overall alignment with the ground truth.
  - **No-Answer Exact Match (`NoAns_exact`):** The percentage of unanswerable questions correctly identified (refused) by the model.
  - **Has-Answer Exact Match (`HasAns_exact`):** The correctness of answers on the subset where information is present in the context.
- [ ] **Run Baseline:**
  - Run the validation on unmodified `Qwen3-0.6B`. (Expect high hallucination rates, meaning low `NoAns_exact` scores).

---

## Phase C — Local Fine-Tuning Setup

**Goal:** Execute the training pipeline on an RTX 4060 (16GB VRAM) locally.

- [ ] **Dependency Setup:**
  - Set up a clean virtual environment containing PyTorch, Hugging Face `transformers`, `peft` (LoRA), and `trl` (`SFTTrainer`).
- [ ] **Training Script (`train.py`):**
  - Configure training parameters optimized for 16GB VRAM:
    - **Base model precision:** 4-bit (QLoRA) or 16-bit BF16.
    - **LoRA configuration:** Rank ($r=8$ or $16$), Alpha ($\alpha=16$ or $32$), targeting all linear modules (`q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
    - **Hyperparameters:** Batch size = 4 (or micro-batch 2 with gradient accumulation), learning rate = `2e-4`, Cosine learning rate scheduler, 3 epochs.
    - **Optimizations:** Gradient checkpointing = `True`.
- [ ] **Run Training:**
  - Run the local fine-tuning script. Monitor GPU memory usage using `nvidia-smi` to ensure VRAM peaks safely below 16GB.

---

## Phase D — Post-Training Validation & Analysis

**Goal:** Validate and compare the fine-tuned adapter.

- [ ] **Post-Tune Benchmarking:**
  - Run the validation script using the fine-tuned model (base model loaded with the LoRA adapter).
- [ ] **Compare Metrics:**
  - Contrast Refusal Rate, QA Accuracy, and Over-Refusal Rate against baseline results.
- [ ] **Weight Fusing:**
  - Merge the trained LoRA adapters back into the base model weights and save the merged model as a standard HF/Safetensors directory.

---

## Phase E — Integration with BareLLM

**Goal:** Run the fine-tuned model using our custom inference engine.

- [ ] **Model Loading:**
  - Load the fused safetensors weights (from Phase D) into the `barellm/model/` transformer.
- [ ] **Inference Verification:**
  - Ensure the custom generation loop works correctly with the newly trained refusal weights.
