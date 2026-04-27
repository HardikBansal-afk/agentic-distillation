# Concept Distillation with Chain-of-Thought Reasoning on GSM8K

## 📌 Project Overview
This project explores **Sequence-to-Sequence Knowledge Distillation**, specifically attempting to transfer complex mathematical reasoning (Chain-of-Thought) from a massive Large Language Model (Teacher) to a significantly smaller, edge-deployable model (Student). 

The pipeline uses the **GSM8K (Grade School Math 8K)** dataset as the benchmark for mathematical reasoning.

### Model Architecture
* **Teacher Model:** `google/flan-t5-xl` (3 Billion Parameters) - Used to generate step-by-step rationales.
* **Student Model:** `google/flan-t5-base` (250 Million Parameters) - Fine-tuned using Seq2Seq learning to mimic the Teacher's reasoning and predict final answers.

---

## 🚀 Features & Optimizations
This pipeline was specifically engineered to run within strict hardware constraints (e.g., a single 15GB NVIDIA T4 GPU on Google Colab) while mirroring enterprise-grade DGX workflows.
* **8-Bit Quantization:** Integrates `bitsandbytes` to compress the 11GB Teacher model into ~5.5GB of VRAM, preventing Out-Of-Memory (OOM) crashes.
* **Gradient Accumulation:** Simulates large batch sizes (32+) for stable Seq2Seq fine-tuning without exceeding memory limits.
* **Dynamic Padding Management:** Implements `-100` label masking to prevent padding tokens from artificially collapsing the loss function during training.
* **Repetition Penalty Handling:** Evaluation parameters are tuned to break recursive token loops (Catastrophic Forgetting) common in small-scale models.
* **Automated Evaluation:** Uses exact-match accuracy extraction (via Regex) alongside `ROUGE-L` scores to measure linguistic degradation.

---

## 🛠️ Requirements & Setup

### Hardware Requirements
* **GPU:** NVIDIA GPU required (Tested on NVIDIA T4 - 15GB VRAM).
* **Environment:** Google Colab or Local Jupyter Environment with CUDA enabled.

### Dependencies
```bash
pip install transformers torch datasets evaluate accelerate rouge_score matplotlib requests bitsandbytes
```

---

💻 How to Run
Open the Jupyter Notebook (Concept_Distillation_GSM8K.ipynb).

If using Google Colab, navigate to Runtime > Change runtime type and select T4 GPU.

Run Cell 1 to initialize the environment and verify GPU allocation.

If bitsandbytes was freshly installed, Restart the Session, then run all cells sequentially.

The pipeline is fully automated and will generate a final Matplotlib performance comparison graph at the end of execution.

📊 Pipeline Structure
Data Ingestion: Securely fetches raw JSONL GSM8K data directly via HTTP to prevent library caching conflicts.

Teacher Rationale Generation: The T5-XL model iterates through the training set, generating step-by-step logic paths for each math problem.

Memory Cleanup: Aggressive Garbage Collection (gc.collect() and torch.cuda.empty_cache()) clears the Teacher model from VRAM.

Student Fine-Tuning: The T5-Base model is trained via Seq2SeqTrainer using the Teacher's rationales as target labels.

Comparative Evaluation: An untrained Baseline Student and the Fine-Tuned Distilled Student are evaluated against the test set.

🔬 Key Academic Findings
This project serves as an experimental exploration into the Capacity Limits of Small Language Models.

Initial runs utilizing Flan-T5-Small (60M parameters) demonstrated textbook Catastrophic Forgetting / Model Collapse. The model lacked the neural parameter space to memorize multi-step math logic, resulting in the destruction of its pre-trained linguistic weights (evidenced by repeating token loops like 4 - 4 - 4...).

Upgrading the student to Flan-T5-Base (250M parameters) and expanding the dataset stabilized the gradient updates, proving that transferring complex reasoning via distillation requires a strict minimum parameter threshold.
