# PAPER_CONTEXT.md

## 0) Purpose of this file
This file is for paper writing assistance (not reproduction).
The goal is to help Codex understand the code-to-paper mapping, contribution boundaries, and safe claims.

---

## 1) Paper scope (must-follow)
- Target venue style: IEEE JSAC / IEEE TMC / IEEE IoTJ.
- Core task: Large-model-enhanced spatial zero-shot traffic forecasting for AIoT systems.
- Evaluated transfer settings in this work:
  1. Cross-city transfer: PeMSD4 (source) -> PeMSD8 (target)
  2. Cross-domain transfer: PeMSD4 domain shift
  3. Cross-domain transfer: Milan domain shift

Do NOT describe intra-city zero-shot or region-level zero-shot as primary evaluated settings in this paper.

---

## 2) High-level method narrative (M1-M5)
Use this module naming consistently in writing:

- M1: ST-Encoder
  Spatio-temporal feature extraction + alignment to language model space.

- M2: Context Augmentation
  Prompt context construction with task anchor + retrieval/context cues.

- M3: LLM Generator (LoRA)
  Generates candidate future traffic sequence from prompt + encoded history.

- M4: SCoRe RL Refinement (training-time optimization)
  Two-pass generation with reward improvement signal and KL-like regularization.

- M5: Continuous Head Output
  Deterministic continuous-value refinement/output for final multi-step forecast.

---

## 3) Code-to-method mapping (authoritative)
### M1/M3/M5 core model
- `model_llm.py`
  - Encoder/alignment blocks (M1)
  - LLM generation path (M3)
  - Continuous head path via `predict_continuous(...)` (M5)

### M4 SCoRe training logic
- `train_SCoRe.py`
  - SFT stage + RL stage + Stage-2 SCoRe refinement
  - Two-pass generation (`y1`, refine prompt, `y2`)
  - Reward computation and policy-gradient update

### Continuous-head focused training
- `train_pred_head.py`
  - Continuous head tuning and related regression setup

### Data and split pipeline
- `dataset.py`
- `config.py`

### Evaluation and metrics export
- `eval_SCoRe.py`
- `eval_visualize_legacy.py`
- `eval_common.py`
- `result_visualize.py`

### Supporting scripts for paper tables/plots
- `scripts/collect_ablation_results.py`
- `scripts/plot_ablation_bars.py`
- `scripts/plot_horizon_matched.py`
- `scripts/plot_mae_rmse_from_eval_json.py`
- `scripts/eval_moirai_spatial_zs.py`
- `scripts/export_moirai_to_baseline_json.py`

### Baselines (minimal uploaded)
- `baselines/run_baseline.py`
- `baselines/data_loader.py`

---

## 4) SCoRe objective (paper-safe formulation)
Main expression:

`adv = r2 - beta * KL + alpha * clip(scale * (r2 - r1))`

Where:
- `r1`: reward after first-pass generation `y1`
- `r2`: reward after refined second-pass generation `y2`
- `KL`: regularization term against reference policy (adapter-off reference path)
- `(r2-r1)`: explicit self-improvement signal used in Stage-2
- `alpha`, `beta`, `scale`/`clip`: configurable coefficients and stabilizers

Implementation notes:
- Policy-gradient update uses token log-probabilities of the second pass (`y2`).
- Stage-1 and Stage-2 differ: Stage-2 adds the explicit `(r2-r1)` improvement term.
- Reward function is shape-aware and includes forecasting-quality components.

---

## 5) Training schedule (for figure/caption consistency)
Bottom training timeline narrative:

- Stage-0: SFT warm-up
- Stage-1: RL
- Stage-2: SCoRe
- Head tuning
- Validation/Test

---

## 6) Claim boundaries (very important)
### Allowed claims
- Large-model-enhanced transfer forecasting framework for AIoT traffic sensing.
- Two-pass SCoRe refinement improves generation quality through explicit self-improvement signal.
- Continuous head provides deterministic multi-step output/refinement.

### Avoid over-claiming
- Do not claim settings that were not evaluated as primary experiments.
- Do not invent additional datasets, metrics, or gains.
- Keep all quantitative statements tied to provided result files/tables.

---

## 7) Figure alignment rules
### Method architecture figure
- Keep M1-M5 structure and training/inference semantics consistent with code.
- M4 should be clearly labeled as training-time optimization.

### Scenario figure
- Primary panels must match evaluated settings only:
  - Cross-city: PeMSD4 -> PeMSD8
  - Cross-domain: PeMSD4 and Milan domain shifts
- Keep source/train vs target/test color semantics consistent.

---

## 8) Writing style constraints for Codex
- IEEE style: concise, evidence-backed, technically specific.
- Prefer precise wording over marketing language.
- Mark uncertain statements as `[NEED_EVIDENCE]`.
- Never fabricate numeric values.

---

## 9) Quick glossary (use consistent terms)
- Source/seen/train sensors
- Target/unseen/test sensors
- Cross-city transfer
- Cross-domain transfer
- SCoRe refinement
- Continuous head output
- Multi-step horizons `H`
