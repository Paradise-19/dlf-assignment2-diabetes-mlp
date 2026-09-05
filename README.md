# Deep Learning Assignment 2 — MLP for Diabetes Prediction

Graduate coursework project (Master of Data Science, The University of Adelaide) implementing a Multi-Layer Perceptron (MLP) from scratch in PyTorch to predict diabetes onset from clinical measurements (Pima Indians Diabetes dataset, National Institute of Diabetes and Digestive and Kidney Diseases).

## Overview

The task: preprocess a real-world clinical dataset (8 numerical features, binary `Outcome` label), implement an MLP using only basic PyTorch building blocks (`nn.Linear`, `nn.ReLU`, `nn.Sequential` — no pretrained models), and compare at least three hyperparameter configurations to analyze how they affect learning.

## Project Structure

```
├── notebook/
│   └── assignment2_diabetes_mlp.ipynb   # Preprocessing → model → hyperparameter search → evaluation
├── data/
│   ├── train_data.csv        # 614 rows, 8 features + Outcome
│   ├── test_data.csv         # 154 rows
│   └── data_description.txt  # Feature descriptions (source: NIDDK)
├── docs/
│   ├── Assignment_2_Instruction.pdf              # Original assignment brief
│   └── Deep_Learning_Fundamental_Assignment_2_Report.pdf   # Submitted technical report
├── model/
│   └── best_mlp_diabetes.pth   # Saved weights of the best-performing model (Param Set 1)
└── requirements.txt
```

## Data Preprocessing

Several columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) use `0` to represent a missing measurement rather than a real physiological value (e.g. blood pressure of 0 is not physiologically possible) — these zeros are replaced with the **median of the non-zero values** in that column, computed from the training set only. All 8 features are then standardized with z-score normalization (`(x - mean) / std`), using statistics from the training set. The cleaned pool is split 80/20 into train/validation; the test set is loaded separately and never touched during training or tuning.

## Model

A configurable MLP (`nn.Linear` → `nn.ReLU` blocks, stacked per a `hidden` tuple, ending in a 2-class output layer), trained with `nn.CrossEntropyLoss` and `torch.optim.Adam`.

## Hyperparameter Search — Results

Three configurations were trained and compared by final validation accuracy:

| Param Set | Hidden Layers | LR | Epochs | Val Accuracy (final epoch) |
|---|---|---|---|---|
| 1 | (32) | 1e-3 | 30 | **74.6%** |
| 2 | (64, 32) | 5e-4 | 40 | 74.6% |
| 3 | (128, 64, 32) | 1e-3 | 50 | 70.5% |

Param Set 1 was selected (simplest model, tied for best validation accuracy — Set 3's larger network overfits, with training accuracy climbing to 96.7% while validation accuracy degrades over epochs).

**Test accuracy (best model, Param Set 1): 74.0%** (loss 0.464)

These numbers are exactly what the committed notebook prints when run end-to-end (see the "Experiments" section output) — they are reproducible from `model/best_mlp_diabetes.pth` without retraining.

## Tools & Libraries

Python, PyTorch (`nn.Linear`, `nn.ReLU`, `nn.Sequential`, `nn.CrossEntropyLoss`, `torch.optim.Adam`, `torch.utils.data.Dataset`/`DataLoader`) — no pretrained models, no third-party ML libraries.

## Running the Notebook

```bash
pip install -r requirements.txt
jupyter notebook notebook/assignment2_diabetes_mlp.ipynb
```

Run all cells top to bottom. The preprocessing cell prints the resulting train/validation/test split sizes and class balance; the experiments cell prints per-epoch training/validation metrics for all three hyperparameter sets, followed by a summary and the final held-out test accuracy.
