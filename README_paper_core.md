# README_paper_core.md

## What this repository subset is for
This repository subset is prepared for paper writing support (code understanding for manuscript drafting), not full reproduction.

---

## Fast reading order (for Codex or co-authors)
1. `PAPER_CONTEXT.md` (writing constraints and claim boundaries)
2. `train_SCoRe.py` (main training pipeline: SFT -> RL -> SCoRe)
3. `model_llm.py` (M1/M3/M5 implementation)
4. `train_pred_head.py` (continuous head tuning details)
5. `dataset.py`, `config.py` (data setup and key switches)
6. `eval_SCoRe.py`, `eval_common.py`, `eval_visualize_legacy.py` (evaluation path)
7. `result_visualize.py` + `scripts/*.py` (table/plot generation logic)
8. `baselines/run_baseline.py` (comparison context)

---

## M1-M5 file mapping
- M1 ST-Encoder: `model_llm.py`
- M2 Context Augmentation: `config.py`, `rag_ref_search.py`, `graph_neighbors.py`, prompt construction in `train_SCoRe.py`
- M3 LLM Generator (LoRA): `model_llm.py`, `train_SCoRe.py`
- M4 SCoRe RL Refinement: `train_SCoRe.py`
- M5 Continuous Head Output: `model_llm.py`, `train_pred_head.py`, evaluation calls in `eval_common.py` / `eval_visualize_legacy.py`

---

## What is included for writing
- Core algorithm files
- Core evaluation pipeline
- Supporting plotting/aggregation scripts for paper results
- Minimal baseline runner files for fair-comparison narrative

---

## What is intentionally excluded
- Large checkpoints / pretrained weights
- Full raw datasets and heavy artifacts
- Large generated outputs not required for method understanding

---

## Evaluated settings (paper-facing)
Primary settings used in this paper:
- Cross-city: PeMSD4 -> PeMSD8
- Cross-domain: PeMSD4 domain shift
- Cross-domain: Milan domain shift

---

## Writing guardrails
- Keep claims aligned with available code + uploaded result files.
- Do not describe non-evaluated settings as primary experiments.
- Keep terminology consistent with figures and `PAPER_CONTEXT.md`.

---

## Suggested prompt to Codex (copy-ready)
Use only information from this repository and `PAPER_CONTEXT.md`.
Do not invent dataset details, metrics, or numerical gains.
When evidence is missing, explicitly mark it as `[NEED_EVIDENCE]`.
Draft in IEEE style with concise, technical, and claim-evidence aligned wording.
