import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.models import (
    INPUT_ORDER,
    CategoryMaps,
    MLP,
    BaseImmersionModel,
    build_category_maps,
    df_to_tensors,
    preprocess_immersion_data,
    preprocess_immersion_data_strided,
    train_template,
)


DATA_DIR = "1-32"
BASE_DIR = os.path.join("Data_Driven_outputs", DATA_DIR)
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)


class ImmersionDataDriven(BaseImmersionModel):
    def __init__(self, num_gpus: int, num_coolants: int):
        super().__init__(num_gpus=num_gpus, num_coolants=num_coolants, embed_dim=4, num_numerical=5)

    def _mlp_out_dim(self) -> int:
        return 1

    def forward(self, gpu_idx, x_num):
        x_scaled, emb = self._encode(gpu_idx, x_num)
        out = self.model(torch.cat([emb, x_scaled], dim=-1))
        t_pred = out[:, 0]
        return t_pred

    def loss_terms(self, gpu_idx, coolant_idx, x_num, t_true):
        t_pred = self.forward(gpu_idx, x_num)
        l_temp = torch.mean((t_pred - t_true) ** 2)
        return l_temp, t_pred


def build_loss_fn_data_driven():
    def _loss(model, gb, cb, xb, yb, train: bool):
        l_temp, t_pred = model.loss_terms(gb, cb, xb, yb)
        loss = l_temp
        return {
            "loss": loss,
            "l_temp": l_temp,
            "val_l_temp": l_temp,
        }
    return _loss


def train_model(
    processed_csv_path: str,
    coolant_json_path: str,
    model_save_path: str,
    meta_save_path: str,
    loss_csv_path: str,
    steps: int = 600,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    def _build_model(maps: CategoryMaps) -> ImmersionDataDriven:
        return ImmersionDataDriven(
            num_gpus=len(maps.gpu_to_idx),
            num_coolants=len(maps.coolant_to_idx),
        )

    loss_csv_header = ["step", "loss", "l_temp", "val_l_temp"]

    def _log_cols():
        return ["l_temp"]

    model, maps, val_df = train_template(
        processed_csv_path=processed_csv_path,
        coolant_json_path=coolant_json_path,
        model_save_path=model_save_path,
        meta_save_path=meta_save_path,
        loss_csv_path=loss_csv_path,
        build_model_fn=_build_model,
        build_loss_fn=build_loss_fn_data_driven(),
        build_csv_header_fn=lambda: loss_csv_header,
        build_log_cols_fn=_log_cols,
        loss_csv_header=loss_csv_header,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        grad_clip=1.0,
        seed=42,
        log_every=20,
        extra_model_save_fn=None,
    )
    return model, maps, val_df


def load_model_for_inference(model_path: str, meta_path: str, device: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    gpu_to_idx = meta["gpu_to_idx"]
    coolant_to_idx = meta["coolant_to_idx"]
    idx_to_gpu = {v: k for k, v in gpu_to_idx.items()}
    idx_to_coolant = {v: k for k, v in coolant_to_idx.items()}
    init_k = np.ones(len(idx_to_gpu), dtype=np.float32) * 0.2
    init_cp = np.ones(len(idx_to_coolant), dtype=np.float32) * 1000.0
    model = ImmersionDataDriven(len(gpu_to_idx), len(coolant_to_idx)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    maps = CategoryMaps(gpu_to_idx, idx_to_gpu, coolant_to_idx, idx_to_coolant, init_k, init_cp)
    return model, maps


def run_inference(model, maps: CategoryMaps, data_df: pd.DataFrame, output_csv_path: str):
    device = next(model.parameters()).device
    g_idx, c_idx, x_num, t_true = df_to_tensors(data_df, maps, device)
    model.eval()
    with torch.no_grad():
        t_pred = model.forward(g_idx, x_num)
    out_df = data_df.copy()
    out_df["predicted_temperature"] = t_pred.detach().cpu().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)) or ".", exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)
    pred_std = float(out_df["predicted_temperature"].std())
    pred_min = float(out_df["predicted_temperature"].min())
    pred_max = float(out_df["predicted_temperature"].max())
    mse = float(np.mean((out_df["predicted_temperature"].values - out_df["temperature"].values) ** 2))
    print(f"inference mse={mse:.6f}")
    print(f"predicted_temperature std={pred_std:.6f} min={pred_min:.6f} max={pred_max:.6f}")
    if pred_std <= 1e-6:
        print("warning: predicted temperature has near-zero variation")


def main():
    input_dir = os.path.join("immersion_data", "PINN")
    coolant_json_path = os.path.join("PINN", "coolant.json")
    processed_csv_path = os.path.join("utlized data", "train_data", DATA_DIR + ".csv")
    model_path = os.path.join(BASE_DIR, "immersion_data_driven.pth")
    meta_path = os.path.join(BASE_DIR, "model_meta.json")
    loss_csv = os.path.join(BASE_DIR, "loss_history.csv")
    infer_csv = os.path.join(BASE_DIR, "inference_results.csv")
    print(f"preprocessed csv saved: {processed_csv_path}")
    model, maps, val_df = train_model(
        processed_csv_path=processed_csv_path,
        coolant_json_path=coolant_json_path,
        model_save_path=model_path,
        meta_save_path=meta_path,
        loss_csv_path=loss_csv,
        steps=600,
        batch_size=256,
        lr=1e-3,
    )
    print(f"trained model saved: {model_path}")
    run_inference(model, maps, val_df, infer_csv)
    print(f"inference csv saved: {infer_csv}")


def inference(processed_csv_path: str, output_csv_path: str, BASE_DIR: str):
    model_path = os.path.join(BASE_DIR, "immersion_data_driven.pth")
    meta_path = os.path.join(BASE_DIR, "model_meta.json")
    infer_csv = output_csv_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, maps = load_model_for_inference(model_path, meta_path, device)
    if processed_csv_path and os.path.exists(processed_csv_path):
        src_df = pd.read_csv(processed_csv_path)
    else:
        gpu_type = list(maps.gpu_to_idx.keys())[0]
        coolant_type = list(maps.coolant_to_idx.keys())[0]
        coolant_density = 1601.0
        time_values = np.arange(0.0, 30.0, 0.1, dtype=np.float32)
        n = len(time_values)
        initial_temperature = 82.9
        gpu_power = 252.6
        src_df = pd.DataFrame(
            {
                "gpu_type": [gpu_type] * n,
                "coolant_type": [coolant_type] * n,
                "coolant_key": [coolant_type] * n,
                "time": time_values,
                "gpu_power": np.full(n, gpu_power, dtype=np.float32),
                "z": np.zeros(n, dtype=np.float32),
                "initial_temperature": np.full(n, initial_temperature, dtype=np.float32),
                "coolant_density": np.full(n, coolant_density, dtype=np.float32),
                "temperature": np.full(n, initial_temperature, dtype=np.float32),
                "source_file": ["synthetic_inference"] * n,
            }
        )
    run_inference(model, maps, src_df, infer_csv)
    print(f"inference csv saved: {infer_csv}")


def temp():
    input_dir = os.path.join("immersion_data", "test")
    coolant_json_path = os.path.join("PINN", "coolant.json")
    processed_csv_path = r"utlized data\test_data.csv"
    print(processed_csv_path)
    preprocess_immersion_data_strided(
        input_dir=input_dir,
        output_csv_path=processed_csv_path,
        coolant_json_path=coolant_json_path,
        row_stride=1,
        dt_max=15.0,
        group_seconds=15.0,
    )


def evaluate(processed_csv_path: str):
    dirs = ["1-32", "1-16", "1-8", "1-4", "1-2", "SccData", "ComData"]
    for dir in dirs:
        BASE_DIR = os.path.join("Data_Driven_outputs", dir)
        if not os.path.exists(BASE_DIR):
            os.makedirs(BASE_DIR)
        # processed_csv_path = r"utlized data\test_data\2100A_BERT.csv"
        output_csv_path = os.path.join(BASE_DIR, "test_data_results.csv")
        inference(processed_csv_path, output_csv_path, BASE_DIR)


if __name__ == "__main__":
    main()
  
