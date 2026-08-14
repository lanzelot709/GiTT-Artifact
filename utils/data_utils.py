import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


INPUT_ORDER = ["time", "gpu_power", "z", "initial_temperature", "coolant_density"]


@dataclass
class CategoryMaps:
    gpu_to_idx: dict
    idx_to_gpu: dict
    coolant_to_idx: dict
    idx_to_coolant: dict
    init_k: np.ndarray
    init_cp: np.ndarray


def _load_coolant_props(coolant_json_path: str):
    with open(coolant_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_coolant_key(coolant_type: str, coolant_props: dict):
    if coolant_type in coolant_props:
        return coolant_type
    c = coolant_type.lower()
    keys = list(coolant_props.keys())
    if "7100" in c:
        for k in keys:
            if "7100" in k.lower():
                return k
    if "2100" in c:
        for k in keys:
            if "7000" in k.lower():
                return k
    for k in keys:
        if "7000" in k.lower():
            return k
    return keys[0]


def _find_column(columns, target_substring):
    t = target_substring.lower()
    for c in columns:
        if t in c.lower():
            return c
    raise ValueError(f"Column containing '{target_substring}' not found")


def _read_gpu_csv(txt_path: str):
    encodings = ["utf-8", "utf-8-sig", "gbk", "cp936", "latin-1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(
                txt_path,
                sep=",",
                engine="python",
                skipinitialspace=True,
                encoding=enc,
                encoding_errors="replace",
            )
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except LookupError:
            continue
    raise last_err if last_err is not None else RuntimeError(f"Failed to read {txt_path}")


def _parse_gpu_coolant(stem: str):
    parts = stem.split("_", 1)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def _standardize_txt(txt_path: str):
    df = _read_gpu_csv(txt_path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[[c for c in df.columns if not c.lower().startswith("unnamed")]]
    date_col = _find_column(df.columns, "date")
    temp_col = _find_column(df.columns, "gpu temperature")
    power_col = _find_column(df.columns, "gpu chip power draw")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
    df[power_col] = pd.to_numeric(df[power_col], errors="coerce")
    df = df.dropna(subset=[date_col, temp_col, power_col]).reset_index(drop=True)
    return df, date_col, temp_col, power_col


def preprocess_immersion_data(
    input_dir: str,
    output_csv_path: str,
    coolant_json_path: str,
    samples_per_file: int = 4096,
    history_window: int = 1800,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    coolant_props = _load_coolant_props(coolant_json_path)
    rows = []
    txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".txt")]
    txt_files.sort()
    for txt_name in txt_files:
        txt_path = os.path.join(input_dir, txt_name)
        stem = os.path.splitext(txt_name)[0]
        parsed = _parse_gpu_coolant(stem)
        if parsed is None:
            continue
        gpu_type, coolant_type = parsed
        coolant_key = _resolve_coolant_key(coolant_type, coolant_props)
        rho_l = float(coolant_props[coolant_key]["rho_l"])
        df, date_col, temp_col, power_col = _standardize_txt(txt_path)
        if len(df) <= history_window:
            continue
        eligible = np.arange(history_window, len(df))
        sample_n = min(samples_per_file, len(eligible))
        sample_idx = rng.choice(eligible, size=sample_n, replace=False)
        for i in sample_idx:
            j = int(rng.integers(i - history_window, i))
            dt = (df.at[i, date_col] - df.at[j, date_col]).total_seconds()
            if dt <= 0:
                continue
            rows.append(
                {
                    "gpu_type": gpu_type,
                    "coolant_type": coolant_type,
                    "coolant_key": coolant_key,
                    "time": float(dt),
                    "gpu_power": float(df.at[i, power_col]),
                    "z": 0.0,
                    "initial_temperature": float(df.at[j, temp_col]),
                    "coolant_density": float(rho_l),
                    "temperature": float(df.at[i, temp_col]),
                    "source_file": txt_name,
                }
            )
    out_df = pd.DataFrame(rows)
    out_df = out_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out_df.to_csv(output_csv_path, index=False)
    return out_df


def preprocess_immersion_data_strided(
    input_dir: str,
    output_csv_path: str,
    coolant_json_path: str,
    row_stride: int = 10,
    dt_max: float = 15.0,
    group_seconds: float = 15.0,
):
    coolant_props = _load_coolant_props(coolant_json_path)
    rows = []
    txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".txt")]
    txt_files.sort()
    for txt_name in txt_files:
        txt_path = os.path.join(input_dir, txt_name)
        stem = os.path.splitext(txt_name)[0]
        parsed = _parse_gpu_coolant(stem)
        if parsed is None:
            continue
        gpu_type, coolant_type = parsed
        coolant_key = _resolve_coolant_key(coolant_type, coolant_props)
        rho_l = float(coolant_props[coolant_key]["rho_l"])
        df, date_col, temp_col, power_col = _standardize_txt(txt_path)
        if len(df) < 2:
            continue
        sampled_idx = list(range(0, len(df), row_stride))
        if len(sampled_idx) < 2:
            continue
        sampled_df = df.iloc[sampled_idx].reset_index(drop=True)
        t0 = sampled_df.at[0, date_col]
        group_ids = ((sampled_df[date_col] - t0).dt.total_seconds() // group_seconds).astype(int)
        sampled_df["_group"] = group_ids.values
        for gid, grp in sampled_df.groupby("_group"):
            grp = grp.reset_index(drop=True)
            if len(grp) < 2:
                continue
            for i_pos in range(1, len(grp)):
                i_time = grp.at[i_pos, date_col]
                i_temp = float(grp.at[i_pos, temp_col])
                i_power = float(grp.at[i_pos, power_col])
                best_j_pos = None
                best_dt = None
                for j_pos in range(i_pos):
                    j_time = grp.at[j_pos, date_col]
                    dt = (i_time - j_time).total_seconds()
                    if dt <= 0 or dt > dt_max:
                        continue
                    if best_dt is None or dt > best_dt:
                        best_dt = dt
                        best_j_pos = j_pos
                if best_j_pos is None:
                    continue
                j_temp = float(grp.at[best_j_pos, temp_col])
                rows.append(
                    {
                        "gpu_type": gpu_type,
                        "coolant_type": coolant_type,
                        "coolant_key": coolant_key,
                        "time": float(best_dt),
                        "gpu_power": float(i_power),
                        "z": 0.0,
                        "initial_temperature": float(j_temp),
                        "coolant_density": float(rho_l),
                        "temperature": float(i_temp),
                        "source_file": txt_name,
                    }
                )
    out_df = pd.DataFrame(rows)
    out_df = out_df.reset_index(drop=True)
    out_df.to_csv(output_csv_path, index=False)
    return out_df


def build_category_maps(df: pd.DataFrame, coolant_json_path: str):
    coolant_props = _load_coolant_props(coolant_json_path)
    gpu_types = sorted(df["gpu_type"].unique().tolist())
    coolant_types = sorted(df["coolant_type"].unique().tolist())
    gpu_to_idx = {g: i for i, g in enumerate(gpu_types)}
    coolant_to_idx = {c: i for i, c in enumerate(coolant_types)}
    idx_to_gpu = {i: g for g, i in gpu_to_idx.items()}
    idx_to_coolant = {i: c for c, i in coolant_to_idx.items()}
    init_k = np.ones(len(gpu_types), dtype=np.float32) * 0.2
    init_cp = np.ones(len(coolant_types), dtype=np.float32) * 1000.0
    for c, idx in coolant_to_idx.items():
        ck = _resolve_coolant_key(c, coolant_props)
        init_cp[idx] = float(coolant_props[ck].get("C_L", 1000.0))
    return CategoryMaps(
        gpu_to_idx=gpu_to_idx,
        idx_to_gpu=idx_to_gpu,
        coolant_to_idx=coolant_to_idx,
        idx_to_coolant=idx_to_coolant,
        init_k=init_k,
        init_cp=init_cp,
    )


def df_to_tensors(df: pd.DataFrame, maps: CategoryMaps, device: str):
    if df is None:
        raise ValueError("df_to_tensors got None DataFrame. Please pass in a non-null DataFrame (or check your processed CSV path).")
    if len(df) == 0:
        raise ValueError("df_to_tensors got an empty DataFrame. Please check your input CSV/data.")
    if "z" not in df.columns:
        df = df.copy()
        df["z"] = 0.0
    gpu_mapped = df["gpu_type"].astype(str).str.strip().map(maps.gpu_to_idx)
    coolant_mapped = df["coolant_type"].astype(str).str.strip().map(maps.coolant_to_idx)

    if gpu_mapped.isna().any():
        unknown = sorted(df.loc[gpu_mapped.isna(), "gpu_type"].astype(str).unique().tolist())
        raise ValueError(f"Unknown gpu_type in inference data: {unknown}. Known: {list(maps.gpu_to_idx.keys())}")

    if coolant_mapped.isna().any():
        unknown = sorted(df.loc[coolant_mapped.isna(), "coolant_type"].astype(str).unique().tolist())
        raise ValueError(f"Unknown coolant_type in inference data: {unknown}. Known: {list(maps.coolant_to_idx.keys())}")

    gpu_idx = torch.tensor(gpu_mapped.values, dtype=torch.long, device=device)
    coolant_idx = torch.tensor(coolant_mapped.values, dtype=torch.long, device=device)
    x_num = torch.tensor(
        df[INPUT_ORDER].values,
        dtype=torch.float32,
        device=device,
    )
    t_true = torch.tensor(df["temperature"].values, dtype=torch.float32, device=device)
    return gpu_idx, coolant_idx, x_num, t_true
