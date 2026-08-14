# Thermal Immersion Cooling — Temperature Prediction Benchmark
=======================================================

---

## 1. Project Layout
------------

```
open_source/
├── GiTT.py                  # Physics-Informed model: train / inference
├── Data_Driven.py           # Data-driven model: train / inference
├── ELIC.py                  # Physics baseline
├── evaluate.py               # Unified evaluator for GiTT + Data_Driven
├── utils/
│   ├── data_utils.py       # Data preprocessing + feature tensor helpers
│   └── models.py          # Shared backbone, base model class, and train loop
│
├── raw_data/                # Raw sensor data, never modified by code
│   ├── PINN/                # Raw TXT logs (input to preprocess_immersion_data)
│   │   ├── 2060s_2100A.txt
│   │   ├── 2060s_7100.txt
│   │   ├── 3090_2100A.txt
│   │   └── 3090_7100.txt
│   └── NVIDIA_GeForce_RTX_{GPU}_{Coolant}/   # Per-GPU/coolant experiment folders
│       ├── train/            # Per-operator CSVs (add, bmm, linear, …)
│       ├── test/             # Per-operator CSVs for held-out operators
│       ├── train_dataset.csv
│       └── test_dataset.csv
│
├── utlized data/            # Preprocessed CSVs used by every training/inference script
│   ├── train_data/          # Training CSVs. Two pre-bundled datasets:
│   │   ├── SccData.csv          (small-scale dataset)
│   │   └── ComData.csv          (full-scale dataset)
│   │   All other dataset names you see referenced (e.g. 1/2, 1/4,
│   │   1/8, 1/16, 1/32) must be sampled yourself via
│   │   preprocess_immersion_data (see section 3).
│   └── test_data/            # Held-out test CSVs (6 files):
│       ├── 2100A_BERT.csv,  7100_BERT.csv
│       ├── 2100A_SD.csv,    7100_SD.csv
│       └── 2100A_ConvNext.csv, 7100_ConvNext.csv
│
├── GiTT_outputs/            # Created after running GiTT training
│   └── <DATASET_NAME>/
│       ├── immersion_pinn_new_pde.pth   # model weights
│       ├── model_meta.json              # category maps + input order
│       ├── learned_k_cp.json             # fitted parameters per GPU and per coolant
│       ├── loss_history.csv              # per-step loss (train + val)
│       └── test__<TEST_STEM>.csv          # inference result (written by evaluate.py)
│
├── Data_Driven_outputs/     # Created after running Data_Driven training
│   └── <DATASET_NAME>/
│       ├── immersion_data_driven.pth
│       ├── model_meta.json
│       ├── loss_history.csv
│       └── test__<TEST_STEM>.csv
│
├── ELIC_outputs/            # Created if ELIC is used inside evaluate.py
│   └── <DATASET_NAME>/
│       ├── elic_params.json, model_meta.json, loss_history.csv
│       └── test__<TEST_STEM>.csv
│
└── evaluation_outputs/      # Written by evaluate.py
    ├── evaluate_summary.csv        # 1 row per (model, train dataset, test CSV)
    └── evaluate_subgroups.csv      # 1 row per (gpu × coolant) subgroup
```

---

## 2. Prerequisites
-------------

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CUDA optional)
- pandas, numpy, scipy



The preprocessing routines require a coolant property file named
`PINN/coolant.json` (create it yourself at the repo root before sampling
your own datasets). It should be a JSON object keyed by coolant name, each
entry containing `rho_l` and `C_L`.

---

## 3. Data Preprocessing — Sample Your Own Datasets
----------------------------------------------

The training code consumes CSVs stored in `utlized data/train_data/`. Two
datasets (`SccData.csv`, `ComData.csv`) are already pre-bundled. Any other
training dataset (e.g. 1/2, 1/4, 1/8, 1/16, 1/32) must be built yourself
from the raw TXT logs in `raw_data/PINN/`.

The helper used for sampling is `utils.data_utils.preprocess_immersion_data`.
Each sample it emits is a single `(T0, T_now)` pair where:

- `initial_temperature` is the temperature at row j (the start point).
- `temperature` is the temperature at row i (the prediction target).
- `time` is `Δt = t_i − t_j`, measured in seconds.
- `gpu_power`, `coolant_density`, `gpu_type`, `coolant_type` are taken
  directly from row i / TXT filename / coolant.json.

### 3.1 Random-history-window sampling

Use the preprocess helper on the TXT folder to create a new processed CSV.
Tune its parameters to control dataset size and how far back the "initial
temperature" can be sampled. Key parameters:

- `samples_per_file`: cap of how many rows to keep per raw TXT file
  (balances dataset size across GPU/coolant combos).
- `history_window`: how many seconds back `j` is allowed to be relative
  to `i` (default 1800 ≈ 30 seconds).
- `seed`: random seed for reproducible jitter.

The resulting processed CSV is written to the path you specify as
`output_csv_path`; save it under `utlized data/train_data/` with a
descriptive name (e.g. `1/8.csv`) so you can reference it later during
training and evaluation.

### 3.2 Strided-group sampling

If you prefer deterministic stride-based instead of random jitter (useful
for building the 6 held-out test CSVs or ablation sets), use the strided
variant. Its parameters:

- `row_stride`: keep every N-th row (downsampling factor, default 10).
- `dt_max`: j must come from within dt_max seconds before i (enforces
  close-pair physics).
- `group_seconds`: rows are grouped into time buckets of this width;
  pairs are only formed within a bucket.


---

## 4. Training GiTT (Physics-Informed)
-------------------------------

GiTT's training loss combines two terms: a physics residual (PDE term)
plus a standard temperature regression term, each with its own weight.
Trainable physical parameters are per-GPU thermal conductivity `k` and
per-coolant volumetric heat capacity `Cp`; both are constrained positive
and initialized from coolant.json plus a sensible default for k.

### How to train

Open the `GiTT.py` script, update the `DATA_DIR` variable near the top
to match the processed training CSV name you want to use (e.g. `SccData`
or one you sampled yourself and dropped into `utlized data/train_data/`).
Then run the script from the repo root:

    python GiTT.py

It will:
1. Create `GiTT_outputs/<DATA_DIR>/` if missing.
2. Run the standard training loop (600 steps, batch 256, lr 1e-3, PDE
   weight 0.01, temperature weight 1.0 — tweak in the `train_model` call
   inside `main()` if you want different behavior).
3. Save 4 artifacts into the output folder:
   - model weights checkpoint,
   - a meta file with the GPU/coolant category maps (required at inference time),
   - a JSON file with the learned k per GPU and Cp per coolant,
   - a per-step loss history CSV with both training and validation metrics.
4. Immediately run inference on the validation split and write an
   `inference_results.csv` as a quick sanity check.

If you prefer to call training programmatically instead of running the
script, use the `train_model` function exported from `GiTT.py` — it
accepts the same parameters that the script sets in its `main()` block.

---

## 5. Training Data_Driven (Pure MLP)
------------------------------

The Data_Driven model shares the exact same backbone, input format, and
training loop as GiTT; the only difference is that the loss contains
only the temperature regression term (no physics residual).

### How to train

Open `Data_Driven.py` and change the `DATA_DIR` variable to the training
CSV you want to train on, then run:

    python Data_Driven.py

It creates `Data_Driven_outputs/<DATA_DIR>/` and writes the same
artifacts as GiTT except for the learned-k/Cp JSON file (since the
data-driven variant has no explicit physics parameters).

Like GiTT, all hyperparameters live inside the `train_model` call in
`main()`; change them there and re-run to reproduce ablations.

---

## 6. Evaluating All Models Across Train/Test Splits
--------------------------------------------------------

`evaluate.py` runs inference for every combination of model, training
dataset checkpoint, and test CSV, then writes aggregated MSE and RMSE
to two summary CSVs.

The axes it sweeps over:

- **Models:** Default is all models.
- **Training datasets:** Default is SccData. Evaluate.py will skip any missing one with
  a clear message.
- **Test CSVs:** the 6 files in `utlized data/test_data/` — one per
  coolant (2100A, 7100) × workload (BERT, SD, ConvNext).

### Run the full sweep

Execute from the repo root:

    python evaluate.py

By default this re-runs inference for every combination (overwriting any
existing cached inference CSVs) and writes:

- `evaluation_outputs/evaluate_summary.csv`: one row per combination
  with overall MSE/RMSE, min/max true/pred values, and the output CSV
  path used.
- `evaluation_outputs/evaluate_subgroups.csv`: one additional row per
  gpu_type × coolant_type subgroup when the test CSV has the relevant
  columns, for fine-grained diagnostics.

At the end of the run, a compact RMSE/MSE quick-view table is also
printed to the terminal for fast inspection.

### Common CLI filters

Instead of running the full sweep, pass any combination of the following:

- `--model` — evaluate only one model.
- `--train-dataset` — evaluate the checkpoint for specific training set.
- `--test-csv` — evaluate on a specific test
  file (basename inside `utlized data/test_data/`).
- `--test-data-dir <folder>` — point evaluator at a different test-data
  folder than the default, for example to evaluate against a custom
  copy of the 6 test CSVs with your own modifications.
- `--overwrite / --no-overwrite` — force or skip re-running inference
  when a cached `test__<stem>.csv` already exists for a combination.
- `--output <path>` / `--output-subgroups <path>` — change where the
  two summary CSVs are written (both relative to the project root, or
  absolute).

---

## 7. Running Inference Standalone
-----------------------------

Both models export a top-level `inference` function with the same
signature, which is what `evaluate.py` calls under the hood. You can
also call it yourself for ad-hoc predictions:

- Argument 1: path to a processed CSV (any CSV with the standard
  columns, including `gpu_type`, `coolant_type`, the 5 numerical inputs,
  and the ground-truth `temperature`).
- Argument 2: where to write the output CSV (absolute or project-relative).
- Argument 3: the BASE_DIR under which the model weights, meta, and (for
  GiTT) learned k/cp JSON live — e.g. `GiTT_outputs/SccData`.

The output CSV contains every input column, and additionally appends the
model predictions. GiTT writes `predicted_temperature`,
`predicted_ratio_alpha`, `predicted_theta`, and `dT_dt`; Data_Driven
writes `predicted_temperature` only.

---