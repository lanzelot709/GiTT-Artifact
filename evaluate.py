import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import GiTT as gitt_mod
import Data_Driven as dd_mod
import ELIC as elic_mod


TRAIN_DATASETS = ["1-32", "1-16", "1-8", "1-4", "1-2", "SccData", "ComData"]
MODELS = ["GiTT", "Data_Driven", "ELIC"]
TEST_CSVS = [
    "2100A_BERT.csv",
    "7100_BERT.csv",
    "2100A_SD.csv",
    "7100_SD.csv",
    "2100A_ConvNext.csv",
    "7100A_ConvNext.csv",
]
TEST_CSVS_EXIST = []
for _candidate in [
    "2100A_BERT.csv", "7100_BERT.csv",
    "2100A_SD.csv", "7100_SD.csv",
    "2100A_ConvNext.csv", "7100A_ConvNext.csv", "7100_ConvNext.csv",
]:
    p = PROJECT_ROOT / "utlized data" / "test_data" / _candidate
    if p.exists():
        TEST_CSVS_EXIST.append(str(p.name))


def per_file_metrics(out_df: pd.DataFrame) -> dict:
    true = out_df["temperature"].astype(float).values
    pred = out_df["predicted_temperature"].astype(float).values
    err = pred - true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    return {
        "n_rows": int(len(out_df)),
        "MSE": round(mse, 6),
        "RMSE": round(rmse, 6),
        "true_min": round(float(true.min()), 4),
        "true_max": round(float(true.max()), 4),
        "pred_min": round(float(pred.min()), 4),
        "pred_max": round(float(pred.max()), 4),
    }


def sub_group_metrics(out_df: pd.DataFrame):
    rows = []
    if "gpu_type" in out_df.columns and "coolant_type" in out_df.columns:
        for (gpu, cool), grp in out_df.groupby(["gpu_type", "coolant_type"]):
            m = per_file_metrics(grp)
            m = {"gpu_type": gpu, "coolant_type": cool, **m}
            rows.append(m)
    return rows


def evaluate_one(model_name: str, train_dataset: str, test_csv_name: str,
                 test_csv_path: str, overwrite: bool) -> tuple[dict, list, str]:
    model_root = PROJECT_ROOT / f"{model_name}_outputs" / train_dataset
    stem = Path(test_csv_name).stem
    out_csv_path = model_root / f"test__{stem}.csv"
    if out_csv_path.exists() and not overwrite:
        print(f"  [skip existing] {out_csv_path.name}")
        df = pd.read_csv(out_csv_path)
    else:
        t0 = time.time()
        if model_name == "GiTT":
            gitt_mod.inference(str(test_csv_path), str(out_csv_path), str(model_root))
        elif model_name == "Data_Driven":
            dd_mod.inference(str(test_csv_path), str(out_csv_path), str(model_root))
        elif model_name == "ELIC":
            model_root.mkdir(parents=True, exist_ok=True)
            params_file = model_root / "elic_params.json"
            if not params_file.exists():
                # Auto-train on the fly: take train CSV from utlized data/train_data/{train_dataset}.csv
                train_csv = PROJECT_ROOT / "utlized data" / "train_data" / f"{train_dataset}.csv"
                meta_file = model_root / "model_meta.json"
                loss_file = model_root / "loss_history.csv"
                elic_mod.train_model(
                    str(train_csv),
                    str(params_file),
                    str(meta_file),
                    str(loss_file),
                )
            elic_mod.inference(str(test_csv_path), str(out_csv_path), str(model_root))
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        dt = time.time() - t0
        print(f"  [done {dt:5.1f}s] saved {out_csv_path}")
        df = pd.read_csv(out_csv_path)

    overall = per_file_metrics(df)
    overall_row = {
        "model": model_name,
        "train_dataset": train_dataset,
        "test_csv": test_csv_name,
        **overall,
        "output_csv": str(out_csv_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    sub_rows = []
    for s in sub_group_metrics(df):
        sub_rows.append({
            "model": model_name,
            "train_dataset": train_dataset,
            "test_csv": test_csv_name,
            **s,
        })
    return overall_row, sub_rows, str(out_csv_path)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate GiTT & Data-Driven models and write results CSV.")
    p.add_argument("--test-csv", type=str, default='2100A_BERT.csv',
                   help="Run on one specific test CSV only (name without folder). Default: run all 6.")
    p.add_argument("--train-dataset", type=str, default='SccData',
                   help="Run on one train dataset only. Default: SccData")
    p.add_argument("--model", type=str, default=None, choices=["GiTT", "Data_Driven", "ELIC"],
                   help="Run only GiTT, Data_Driven, or ELIC. Default: all.")
    p.add_argument("--test-data-dir", type=str, default='utlized data/test_data',
                   help="Folder where test CSVs are stored (default: utlized data/test_data).")
    p.add_argument("--overwrite", dest="overwrite", action="store_true", default=True,
                   help="Force re-run inference even if test__*.csv already exists (default skip).")
    p.add_argument("--output", type=str, default=None,
                   help="Output summary CSV path (relative to project root or absolute). "
                        "Default: evaluation_outputs/evaluate_summary.csv")
    p.add_argument("--output-subgroups", type=str, default=None,
                   help="Output subgroup (per gpu+coolant) summary CSV path. "
                        "Default: evaluation_outputs/evaluate_subgroups.csv")
    return p.parse_args()


def main():
    args = parse_args()
    if args.test_data_dir is None:
        test_data_dir = PROJECT_ROOT / "utlized data" / "test_data"
    else:
        test_data_dir = Path(args.test_data_dir)
        if not test_data_dir.is_absolute():
            test_data_dir = PROJECT_ROOT / test_data_dir

    if args.output is None:
        out_overall = PROJECT_ROOT / "evaluation_outputs" / "evaluate_summary.csv"
    else:
        out_overall = Path(args.output)
        if not out_overall.is_absolute():
            out_overall = PROJECT_ROOT / out_overall

    if args.output_subgroups is None:
        out_sub = PROJECT_ROOT / "evaluation_outputs" / "evaluate_subgroups.csv"
    else:
        out_sub = Path(args.output_subgroups)
        if not out_sub.is_absolute():
            out_sub = PROJECT_ROOT / out_sub

    out_overall.parent.mkdir(parents=True, exist_ok=True)
    out_sub.parent.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else ["GiTT", "Data_Driven", "ELIC"]
    datasets = [args.train_dataset] if args.train_dataset else TRAIN_DATASETS
    tests = [args.test_csv] if args.test_csv else list(TEST_CSVS_EXIST)

    print(f"Models:       {models}")
    print(f"Train sets:   {datasets}")
    print(f"Test CSVs:    {tests}")
    print(f"Overwrite:    {args.overwrite}")
    print(f"Output file:  {out_overall}")
    print()

    overall_rows = []
    sub_rows = []

    total_jobs = len(models) * len(datasets) * len(tests)
    job = 0
    for model_name in models:
        for train_ds in datasets:
            for test_csv in tests:
                job += 1
                test_csv_path = test_data_dir / test_csv
                if not test_csv_path.exists():
                    print(f"[{job}/{total_jobs}] {model_name} train={train_ds} test={test_csv}: missing test CSV, skip")
                    continue
                print(f"[{job}/{total_jobs}] {model_name} train={train_ds:7s} test={test_csv:22s}")
                try:
                    row, subs, out_path = evaluate_one(
                        model_name, train_ds, test_csv, test_csv_path, args.overwrite
                    )
                    overall_rows.append(row)
                    sub_rows.extend(subs)
                except Exception as e:
                    print(f"  !! ERROR: {type(e).__name__}: {e}")
                    overall_rows.append({
                        "model": model_name,
                        "train_dataset": train_ds,
                        "test_csv": test_csv,
                        "n_rows": 0,
                        "MSE": None, "RMSE": None,
                        "true_min": None, "true_max": None, "pred_min": None, "pred_max": None,
                        "output_csv": None,
                        "error": f"{type(e).__name__}: {e}",
                    })
                    continue

    over_df = pd.DataFrame(overall_rows)
    sub_df = pd.DataFrame(sub_rows)
    over_df.to_csv(out_overall, index=False, encoding="utf-8-sig")
    print(f"\nOverall summary saved: {out_overall}  ({len(over_df)} rows)")
    if len(sub_df) > 0:
        sub_df.to_csv(out_sub, index=False, encoding="utf-8-sig")
        print(f"Subgroup summary saved: {out_sub}  ({len(sub_df)} rows)")

    if len(over_df) > 0 and "RMSE" in over_df.columns:
        pivot_cols = [c for c in ["train_dataset", "test_csv", "model", "RMSE", "MSE"] if c in over_df.columns]
        print("\n=== RMSE/MSE quick view ===")
        disp = over_df[pivot_cols].copy() if len(pivot_cols) > 0 else over_df
        with pd.option_context("display.width", 200, "display.max_columns", None, "display.max_rows", 200):
            print(disp.to_string(index=False))


if __name__ == "__main__":
    main()
