# Parameter-Efficient Fine-Tuning of SmolVLM-500M for Visual Multiple-Choice Science QA

**NYU Deep Learning Spring 2026 — Final Project**

Fine-tuning SmolVLM-500M-Instruct on a ScienceQA-derived visual multiple-choice benchmark using LoRA via PEFT, under a 5M trainable-parameter cap on a single Tesla T4 GPU.

[Link to adapters and outputs on Google Drive](https://drive.google.com/drive/folders/1SmzQLMMpoZDNWgxE5WqZnmIioFgJyYGn?usp=drive_link)

**Best Score: 0.831** (v9-ablation, live-weights inference) · **Best simple SFT: 0.829** (v4) · **Best ensemble: 0.827** (v4 + v9 logit average)

---

## Results

| Version | Method | Zero-shot | Best Val | Test | Key Change |
|---------|--------|-----------|----------|------|------------|
| v1  | LoRA r=16, q/k/v/o      | 0.508 | 0.751 | 0.803     | Baseline (no CoT-as-input, 2 epochs) |
| v2  | LoRA r=16, q/k/v/o      | 0.508 | 0.756 | 0.795     | + CoT-as-input, 4 epochs, lr=5e-5 |
| v3  | LoRA r=24, q/k/v/o      | 0.508 | 0.773 | 0.811     | rank 24, alpha 48, 3 epochs |
| v4  | LoRA r=24, q/k/v/o      | 0.508 | 0.785 | **0.829** | 5 epochs (default recipe, best simple model) |
| v5  | LoRA r=24, q/k/v/o      | 0.513 | 0.787 | 0.813     | image edge=512, choice augmentation, dropout=0.10 |
| v6  | LoRA r=24, q/k/v/o      | 0.432 | 0.769 | 0.793     | Compact question-first prompt format |
| v7  | LoRA r=24, q/k/v/o      | 0.508 | 0.778 | 0.803     | Seed 1337 (vs. default 42) |
| v8  | LoRA r=16, MLP-only     | 0.508 | 0.782 | 0.823     | MLP-only LoRA target modules |
| v9  | LoRA r=24, q/k/v/o      | 0.508 | 0.818\* | 0.805   | + train ∪ val, EMA, TTA |
| v9-ablation | (v9 adapter, clean inference) | — | — | **0.831** | Live weights, no EMA, no TTA |
| v10 | LoRA r=24, q/k/v/o      | 0.575 | 0.763 | 0.805     | Raw-image prompt format (no chat template) |
| v10-ablation | (v10 adapter, clean inference) | — | — | 0.825 | Live weights, no EMA, no TTA |
| v11 | LoRA r=24, q/k/v/o      | 0.517 | 0.719 | 0.767     | Answer-first generative CoT |
| v12 | LoRA r=24, q/k/v/o      | 0.575 | 0.717 | 0.757     | Reasoning-first generative CoT |
| v13 | LoRA r=16, q/v + MLP    | 0.508 | 0.792 | 0.823     | Self-generated captions + mixed LoRA topology |
| v4 + v5 ensemble | logit average | — | — | 0.807 | Logit-averaged predictions |
| v4 + v9 ensemble | logit average | — | — | 0.827 | Logit-averaged predictions |

\*v9 trained on train ∪ val, so its best-val number is in-distribution and not directly comparable to the others; we report it for completeness.

### Ablation: Inference Pipeline (Same Adapter, Two Inference Recipes)

The largest inference-side effect in our study comes from running the same trained adapter under two different inference recipes:

| Recipe | Adapter | Test Accuracy |
|--------|---------|---------------|
| Live weights, no EMA, no TTA           | v9  | **0.831** |
| EMA shadow weights + TTA over choice permutations | v9  | 0.805 |
| Live weights, no EMA, no TTA           | v10 | 0.825 |
| EMA shadow weights + TTA over choice permutations | v10 | 0.805 |
| Live weights + TTA only (no EMA)       | v10 | 0.825 (TTA inert) |

**Finding:** A "polish" inference recipe with EMA decay 0.999 and choice-permutation TTA cost 2.0–2.6 percentage points relative to plain greedy inference on the same adapter. TTA contributed nothing in isolation — the entire regression is from EMA. Mechanism: an EMA averaging window of ~1,000 steps is comparable in length to the run itself (~1,300 steps), so the shadow weights average over still-improving early-training weights rather than reflecting the converged adapter. See [Key Findings](#key-findings) #2 below.

---

## Key Findings

1. **Capacity and Training Duration are the Main Levers (+0.026 test points).** Going from rank 16, 2 epochs (v1) to rank 24, 5 epochs (v4) gained 2.6 percentage points on test, the largest single positive contribution in the study. No other intervention we tried matched this magnitude. At this base-model scale, additional rank and additional epochs dominate any architectural creativity in the LoRA target set.

2. **EMA Dilution Silently Cost 2.6 Points.** The same trained v9 adapter scored 0.805 with EMA + TTA at inference and 0.831 without. We caught this only by holding out adapters and re-running inference on saved weights. EMA shadow weights at decay 0.999 over a 1,300-step run necessarily include substantial mass on still-improving early-training weights — an averaging window comparable in length to the run is not a regularizer, it is a drag on the converged adapter. We suspect this failure mode is common but rarely diagnosed in single-shot evaluation pipelines.

3. **Generative Chain-of-Thought Actively Hurt at 500M Parameters (-0.062 / -0.072).** Both answer-first (v11) and reasoning-first (v12) generative CoT objectives regressed substantially relative to single-token letter-CE training. Confusion-matrix analysis showed that the failure is **diffuse degradation** — v11 and v12 retain v4's off-diagonal pattern (dominant 0↔1 and 1↔2 confusion, complete index-4 abandonment) but lose 15–30 correct predictions in every gold class. The model loses sharpness on existing failure modes; it does not develop new ones. Consistent with prior reports that CoT benefits emerge at much larger scale.

4. **Prompt Format Effects Largely Don't Carry Through Fine-Tuning.** Zero-shot validation accuracy spans 14 percentage points across our three prompt formats (v6 compact: 0.432, v5_baseline: 0.508, v10 raw-image: 0.575). After 5 epochs of LoRA, the fine-tuned test scores converge to within 4 points (0.793, 0.829, 0.825). The adapter learns to read whatever surface form it is shown.

5. **The Five-Choice Slice is a Single-Capability Punnett Collapse.** The five-choice slice is 92–99% Punnett-square problems across all splits, and v4 reaches only 0.25 accuracy on it (vs. 0.78–0.85 on 2/3/4-choice). Closing this single capability gap would meaningfully improve test scores, but it requires the model to interpret a square diagram, apply Mendelian inheritance rules, and string-match shuffled ratios — capabilities that did not emerge under our training recipe.

---

## Repository Structure

```
├── notebooks/
│   ├── smolvlm_starter_ipynb_v1.ipynb              # Rank 16, no CoT-as-input, 2 epochs
│   ├── smolvlm_starter_ipynb_v2.ipynb              # + CoT-as-input, 4 epochs, lr=5e-5
│   ├── smolvlm_starter_ipynb_v3.ipynb              # Rank 24, alpha 48, 3 epochs
│   ├── smolvlm_starter_ipynb_v4.ipynb              # 5 epochs (default recipe, best simple model)
│   ├── smolvlm_starter_ipynb_v5.ipynb              # image=512, choice aug, dropout 0.10
│   ├── smolvlm_starter_ipynb_v6.ipynb              # Compact question-first prompt
│   ├── smolvlm_starter_ipynb_v7.ipynb              # Seed 1337
│   ├── smolvlm_starter_ipynb_v8.ipynb              # MLP-only LoRA
│   ├── smolvlm_starter_ipynb_v9.ipynb              # train ∪ val + EMA + TTA
│   ├── smolvlm_starter_ipynb_v9_ablations.ipynb    # Inference-only ablations on saved v9 adapter
│   ├── smolvlm_starter_ipynb_v10.ipynb             # Raw-image prompt format (no chat template)
│   ├── smolvlm_starter_ipynb_v10_ablations.ipynb   # Inference-only ablations on saved v10 adapter
│   ├── smolvlm_starter_ipynb_v11.ipynb             # Answer-first generative CoT
│   ├── smolvlm_starter_ipynb_v12.ipynb             # Reasoning-first generative CoT
│   └── smolvlm_starter_ipynb_v13.ipynb             # Captions + mixed q/v + MLP LoRA
├── scripts/
│   ├── analyze.ipynb                # Per-version metrics analysis & cross-version plots
│   └── dataset_analyze.ipynb        # EDA on splits, distribution drift, slice composition, image dimensions
├── figures/
│   ├── comparison/
│   │   ├── best_val_accuracy_bars.png
│   │   ├── val_accuracy_curves.png
│   │   └── summary_table.csv
│   ├── v1/
│   │   ├── confusion_matrix.png
│   │   ├── training_dynamics.png
│   │   └── val_breakdown.png
│   ├── v2/ ... v13/                 # Same three files per version
│   ├── answer_dist.png              # Train vs. val answer-index distribution
│   ├── data_numchoices_drift.png    # num_choices distribution across splits
│   ├── dist_drift.png               # 4-panel split distribution drift (num_choices, subject, task, grade)
│   ├── image_dims.png               # Image aspect ratio and resolution per split
│   └── text_lengths.png             # Composed-prompt text length distribution per split
├── report/
│   └── ECE_GY_7123_Final.pdf
└── README.md
```

---

## Reproduction

### Requirements

- Google Colab Free with a single Tesla T4 GPU (16 GB)
- Google Drive for dataset and adapter checkpoint storage
- Python 3.10+, `transformers==4.49.0`, `peft==0.13.2`, PyTorch with `fp16` autocast support

### Quick Start (Inference Only)

1. Download the v9 adapter from Google Drive
2. Open `notebooks/smolvlm_starter_ipynb_v9_ablations.ipynb` in Colab
3. Update `ADAPTER_PATH` to point to your adapter directory
4. Run all cells — generates 1,008 predictions in ~10 minutes on T4 and writes `submission.csv`

### Full Training (v4 — Best Simple Model)

1. Upload the `pixels-to-predictions/` dataset directory (CSVs + images) to Google Drive
2. Open `notebooks/smolvlm_starter_ipynb_v4.ipynb` in Colab
3. Update `PROJECT_DIR` to your Drive path
4. Run all cells — trains in ~1.4 hours on T4, then runs inference in the same session and writes `submission.csv`

### Full Training (v9 — Best Score)

1. Same dataset upload as above
2. Open `notebooks/smolvlm_starter_ipynb_v9.ipynb` in Colab
3. Update `PROJECT_DIR` and confirm `train_data = "train_plus_val"` in the config
4. Run all cells — trains in ~1.8 hours on T4
5. Run `notebooks/smolvlm_starter_ipynb_v9_ablations.ipynb` to evaluate the saved adapter under live-weights inference (the headline 0.831 submission)

### Ablation Experiments

Each version's notebook is end-to-end self-contained. To reproduce a specific ablation pair, run both notebooks involved and compare their `submission.csv` outputs against the public test leaderboard:

- **Capacity:** v1 vs. v3 (rank 16 → 24)
- **Duration:** v3 vs. v4 (3 → 5 epochs)
- **LoRA topology:** v4 vs. v8 (q/k/v/o → MLP-only at rank 16); v4 vs. v13 (q/k/v/o → mixed q/v + MLP, with self-generated captions)
- **Prompt format:** v4 vs. v6 (chat-templated → compact question-first); v4 vs. v10_ablations (chat-templated → raw-image)
- **Training data:** v4 vs. v9_ablations (train → train ∪ val)
- **Training objective:** v4 vs. v11 (letter-CE → answer-first generative CoT); v4 vs. v12 (letter-CE → reasoning-first generative CoT)
- **Inference pipeline:** v9 vs. v9_ablations (drop EMA + TTA); v10 vs. v10_ablations (same)
- **Seed:** v4 vs. v7 (seed 42 → 1337)

### Generating Report Figures

1. Place each version's `metrics.json` in a per-version directory under the path the notebook expects (`<ROOT>/v<N>/metrics.json`)
2. Run `scripts/analyze.ipynb` to produce per-version `training_dynamics.png`, `val_breakdown.png`, and `confusion_matrix.png`, plus the cross-version `comparison/` plots and `summary_table.csv`
3. Run `scripts/dataset_analyze.ipynb` to produce `data_numchoices_drift.png`, `dist_drift.png`, `image_dims.png`, `text_lengths.png`, and `answer_dist.png` from the raw splits

---

## Competition Details

- **Task:** Visual multiple-choice science QA — given an image, a question, optional supporting context, and 2–5 candidate answers, predict the index of the correct answer.
- **Scoring:** Single-choice accuracy on the public test leaderboard. The competition also maintains a private leaderboard which we do not examine.
- **Constraints:** Base model must be `HuggingFaceTB/SmolVLM-500M-Instruct`; at most 5 M trainable parameters; all training and inference on a single Tesla T4 GPU via Google Colab Free.
- **Dataset:** ScienceQA-derived, 5,165 examples total — 3,109 train / 1,048 val / 1,008 test (60/20/20). Schema: `id`, `image_path`, `question`, `choices`, `num_choices`, `answer` (omitted on test), `hint`, `lecture`, `solution`, plus metadata fields (`task`, `grade`, `subject`, `topic`, `category`, `skill`).

## Training Configuration (v9 — Best Score)

| Parameter | Value |
|-----------|-------|
| Base model | HuggingFaceTB/SmolVLM-500M-Instruct |
| Adaptation | LoRA via PEFT (`task_type=CAUSAL_LM`, `bias='none'`) |
| Quantization | None (fp16 autocast forward+backward; fp32 optimizer state) |
| LoRA rank | 24 |
| LoRA alpha | 48 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` (LM decoder only) |
| LoRA dropout | 0.05 |
| Trainable params | 4,915,200 / ~500M (~1%) |
| Optimizer | AdamW (`weight_decay=0`) |
| Learning rate | 1e-4 (cosine schedule, 3% warmup) |
| Gradient clip | 1.0 (post-unscale) |
| Epochs | 5 |
| Effective batch size | 16 (micro-batch 2 × grad accum 8) |
| Max text tokens | 1,280 |
| Image edge | 384 (longest side) |
| CoT-as-input | `cot_prob=0.5` (training only; test split has no `solution`) |
| Training samples | 4,157 (train ∪ val) |
| Training time | 1.82 hours on Tesla T4 |
| Peak GPU memory | 6.02 GB |
| Best val accuracy | 0.818 (in-distribution; not held out) |
| Test accuracy | **0.831** (live-weights inference, no EMA, no TTA) |

---

## AI Tooling Disclosure

- **Claude (Anthropic):** Coding assistance, debugging, error analysis, report drafting, and figure verification.
