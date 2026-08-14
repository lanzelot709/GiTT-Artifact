import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, Bounds


DATA_DIR = "SccData"
BASE_DIR = os.path.join("ELIC_outputs", DATA_DIR)
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR, exist_ok=True)


def _detect_run_starts(df: pd.DataFrame):
    """Return a boolean series indicating the first row of each independent run.

    A new run starts when: row 0, OR time diff < 0 (time column resets),
    OR initial_temperature changes, OR gpu/coolant/source_file changes.
    For each run-start row, we use its initial_temperature column as T0,
    and integrate forward row-by-row within that run (with P(t) per-row).
    """
    n = len(df)
    starts = np.zeros(n, dtype=bool)
    if n == 0:
        return starts
    starts[0] = True
    if "time" in df.columns:
        dt = df["time"].astype(float).diff().to_numpy()
        starts |= (dt < 0)
        med_dt = float(np.nanmedian(np.abs(dt[1:]))) if n > 2 else 0.0
        gap_thresh = max(med_dt * 20, 10.0) if med_dt > 0 else 10.0
        starts |= (dt > gap_thresh)
    for col in ["initial_temperature", "gpu_type", "coolant_type", "source_file"]:
        if col in df.columns:
            col_s = df[col]
            try:
                changed = col_s.ne(col_s.shift(1)).fillna(False).to_numpy()
            except Exception:
                changed = np.array([False] + [str(col_s.iloc[i]) != str(col_s.iloc[i-1]) for i in range(1, n)])
            starts |= changed
    return starts


def integrate(df: pd.DataFrame, a, b, c):
    """Predict temperature for every row in df via per-row integration.

    Per-row ODE step (Q is constant within each run since Q = b*(T0 - c)^3):
        For each row i within a run starting at row s:
          T0 = initial_temperature_s   (read directly from the run-start row)
          dt_i = max(time_i - time_{i-1}, 0)   (first row in run: dt = 0)
          slope_i = a * (P_i - b*(T0 - c)^3)
          cum_i = cum_{i-1} + slope_i * dt_i
          T_pred_i = T0 + cum_i

    Returns:
        T_pred  (float ndarray, shape=n_rows)
        dTdt    (float ndarray, shape=n_rows)
    """
    n = len(df)
    T_pred = np.zeros(n, dtype=float)
    dTdt = np.zeros(n, dtype=float)
    if n == 0:
        return T_pred, dTdt
    starts = _detect_run_starts(df)
    time_arr = df["time"].astype(float).to_numpy() if "time" in df.columns else np.arange(n, dtype=float)
    P_arr = (df["gpu_power"].astype(float).to_numpy() if "gpu_power" in df.columns
             else np.zeros(n, dtype=float))
    init_arr = (df["initial_temperature"].astype(float).to_numpy() if "initial_temperature" in df.columns
                else df["temperature"].astype(float).to_numpy())
    a_use = max(float(a), 1e-12)
    b_use = max(float(b), 1e-12)
    c_use = float(c)

    T0_cur = float(init_arr[0])
    Q_cur = b_use * ((T0_cur - c_use) ** 3)
    slope0 = a_use * (float(P_arr[0]) - Q_cur)
    T_pred[0] = T0_cur
    dTdt[0] = slope0
    last_t = float(time_arr[0])
    cum = 0.0
    for i in range(1, n):
        if starts[i]:
            T0_cur = float(init_arr[i])
            Q_cur = b_use * ((T0_cur - c_use) ** 3)
            cum = 0.0
            last_t = float(time_arr[i])
            T_pred[i] = T0_cur
            dTdt[i] = a_use * (float(P_arr[i]) - Q_cur)
            continue
        dt = max(float(time_arr[i]) - last_t, 0.0)
        P_i = float(P_arr[i])
        slope_i = a_use * (P_i - Q_cur)
        cum += slope_i * dt
        T_pred[i] = T0_cur + cum
        dTdt[i] = slope_i
        last_t = float(time_arr[i])
    return T_pred, dTdt


# -------- training / fitting ----------
def train_model(train_csv_path: str,
                params_save_path: str,
                meta_save_path: str = None,
                loss_csv_path: str = None):
    """Fit a, b, c over the training CSV using pure per-row integration.
    Save parameters to JSON. Return dict {params, meta}. """
    df = pd.read_csv(train_csv_path)
    req = ["temperature", "time", "gpu_power", "initial_temperature"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"train CSV missing required columns: {miss}")

    T_true = df["temperature"].astype(float).to_numpy()
    n = len(df)
    if n < 2:
        raise RuntimeError("Training CSV has fewer than 2 rows.")

    c_init = 55.5
    a_init = 1.82e-2
    b_init = 6.29e-4
    x0 = np.array([a_init, b_init, c_init], dtype=float)
    bounds = Bounds(lb=[1.72e-2, 5.8e-4, 55.0], ub=[1.92e-2, 6.8e-4, 56.0])

    def _res(params):
        a, b, c = params
        pred, _ = integrate(df, a, b, c)
        return (pred - T_true).astype(float)

    result = least_squares(
        _res, x0, bounds=bounds, method="trf", max_nfev=5000, ftol=1e-10, xtol=1e-10
    )
    a_fit, b_fit, c_fit = [float(v) for v in result.x]
    final_mse = float(np.mean(result.fun ** 2)) if len(result.fun) > 0 else float("nan")
    final_rmse = float(np.sqrt(final_mse))

    starts = _detect_run_starts(df)
    n_runs = int(starts.sum())

    payload = {
        "model": "ELIC_baseline",
        "formula": "dT/dt = a * (P - Q), Q = b * (T0 - c)^3, T0 = initial_temperature @ run start",
        "train_csv": str(train_csv_path),
        "n_runs": n_runs,
        "n_rows": int(n),
        "parameters": {"a": a_fit, "b": b_fit, "c": c_fit},
        "optim_status": {
            "success": bool(result.success),
            "nfev": int(result.nfev),
            "message": str(result.message),
            "final_residual_size": int(len(result.fun)),
            "final_cost": float(result.cost),
            "final_mse": final_mse,
            "final_rmse": final_rmse,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(params_save_path)) or ".", exist_ok=True)
    with open(params_save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"ELIC params saved -> {params_save_path}")
    print(f"  a={a_fit:.6e}  b={b_fit:.6e}  c={c_fit:.4f}")
    print(f"  train RMSE = {final_rmse:.4f} degC  (n_rows={n},  runs={n_runs})")

    if meta_save_path:
        if "source_file" in df.columns:
            src = df.loc[starts, "source_file"].astype(str).tolist()
        else:
            src = [""] * n_runs
        inits = df.loc[starts, "initial_temperature"].astype(float).tolist()
        run_sizes = np.diff(np.append(np.where(starts)[0], n)).tolist()
        meta = {
            "model_name": "ELIC_baseline",
            "train_csv": str(train_csv_path),
            "params_file": str(params_save_path),
            "parameters": {"a": a_fit, "b": b_fit, "c": c_fit},
            "n_runs": n_runs,
            "n_rows": int(n),
            "train_rmse": final_rmse,
            "train_mse": final_mse,
            "run_sizes_first100": run_sizes[:100],
            "run_initials_first100": inits[:100],
            "run_sources_first100": src[:100],
        }
        os.makedirs(os.path.dirname(os.path.abspath(meta_save_path)) or ".", exist_ok=True)
        with open(meta_save_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"ELIC meta saved -> {meta_save_path}")

    if loss_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(loss_csv_path)) or ".", exist_ok=True)
        pd.DataFrame([{
            "step": 0,
            "total_loss": float(result.cost) * 2.0 / max(1, n),
            "temp_loss": final_mse,
            "rmse": final_rmse,
        }]).to_csv(loss_csv_path, index=False)
        print(f"ELIC history saved -> {loss_csv_path}")

    return payload


def load_params_for_inference(BASE_DIR: str):
    params_path = os.path.join(BASE_DIR, "elic_params.json")
    with open(params_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    p = payload["parameters"]
    return float(p["a"]), float(p["b"]), float(p["c"])


def run_inference(data_df: pd.DataFrame, output_csv_path: str, a: float, b: float, c: float):
    """Like GiTT.run_inference. Pure per-row integration:
       - no groupby/split;
       - T0 read from each run-start row's initial_temperature column;
       - per-row integrate using its own gpu_power and time delta.
    """
    out_df = data_df.copy()
    T_pred, dTdt = integrate(data_df, a, b, c)
    out_df["predicted_temperature"] = T_pred.astype(float)
    out_df["predicted_ratio_alpha"] = 0.0
    out_df["predicted_theta"] = 0.0
    out_df["dT_dt"] = dTdt.astype(float)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)) or ".", exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)
    pred_series = out_df["predicted_temperature"].astype(float)
    true_series = out_df["temperature"].astype(float)
    pred_std = float(pred_series.std()) if len(pred_series) > 1 else 0.0
    pred_min = float(pred_series.min())
    pred_max = float(pred_series.max())
    mse = float(np.mean((pred_series.values - true_series.values) ** 2))
    print(f"inference mse={mse:.6f}")
    print(f"predicted_temperature std={pred_std:.6f} min={pred_min:.6f} max={pred_max:.6f}")
    print("theta std=0.000000 alpha std=0.000000")
    if pred_std <= 1e-6:
        print("warning: predicted temperature has near-zero variation")


def inference(processed_csv_path: str, output_csv_path: str, BASE_DIR: str):
    a, b, c = load_params_for_inference(BASE_DIR)
    data_df = None
    if processed_csv_path and os.path.exists(processed_csv_path):
        data_df = pd.read_csv(processed_csv_path)
    if data_df is None or data_df.empty:
        raise FileNotFoundError(f"Need a valid processed CSV to run ELIC inference; got {processed_csv_path}")
    run_inference(data_df, output_csv_path, a, b, c)
    print(f"inference csv saved: {output_csv_path}")


# -------- convenience scripts --------
def main():
    """By default train on DATA_DIR; inference on test_data 2100A_BERT.csv."""
    train_csv_path = os.path.join("utlized data", "train_data", DATA_DIR + ".csv")
    params_path = os.path.join(BASE_DIR, "elic_params.json")
    meta_path = os.path.join(BASE_DIR, "model_meta.json")
    loss_csv = os.path.join(BASE_DIR, "loss_history.csv")
    train_model(train_csv_path, params_path, meta_path, loss_csv)

    test_csv = os.path.join("utlized data", "test_data", "2100A_BERT.csv")
    infer_csv = os.path.join(BASE_DIR, "inference_results.csv")
    inference(test_csv, infer_csv, BASE_DIR)
    print(f"quick inference saved: {infer_csv}")


if __name__ == "__main__":
    main()
