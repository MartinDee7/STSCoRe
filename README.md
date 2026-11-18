# STSCoRe

Implementation and training code for ST-SCoRe, including source training and target adaptation.

## Project Structure

```text
STSCoRe/
├── stscore/
│   ├── __init__.py
│   ├── data.py                     # Data loading, window sampling, and retrieval
│   ├── engine.py                   # Context construction and forecasting flow
│   ├── model.py                    # ST encoder, shared LLM, and continuous head
│   ├── objectives.py               # Forecast rewards and policy objectives
│   ├── prepare.py                  # Data and spatial feature preparation
│   └── train.py                    # SFT, RL, head tuning, and target adaptation
├── configs/
│   ├── training.json               # Model and training settings
│   ├── selected_main_budget.json   # Target adaptation budget
│   ├── split_pems04.json           # PeMSD4 sensor split
│   ├── split_pems08.json           # PeMSD8 sensor split
│   └── split_milan.json            # Milan spatial split
├── tests/                          # Implementation tests
├── config.py                       # Configuration interfaces
├── dataset.py                      # Data interfaces
├── model_llm.py                    # Model interfaces
├── train_SCoRe.py                  # Training and adaptation entry point
├── train_pred_head.py              # Continuous-head tuning entry point
├── requirements.txt                # Dependencies
└── README.md
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Source training

```bash
python train_SCoRe.py --data processed_data/pems04_raw.npz \
  --split configs/split_pems04.json --config configs/training.json \
  --stage all --seed "$TRAIN_SEED" --output runs/pems04 --device cuda
```

## Target adaptation

```bash
python train_SCoRe.py --data processed_data/pems08_raw.npz \
  --split configs/split_pems08.json --stage adapt \
  --checkpoint runs/pems04/head_best.pt --seed "$TRAIN_SEED" \
  --selected-budget configs/selected_main_budget.json \
  --output runs/pems04_to_pems08_20pct --device cuda
```

Set `TRAIN_SEED` and adjust the paths before running. Training settings are in `configs/training.json`.
