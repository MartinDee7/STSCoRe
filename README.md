# ST-SCoRe

Implementation and training code for ST-SCoRe, including source training and target adaptation.

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
