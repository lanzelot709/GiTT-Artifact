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
    _inv_softplus,
    _load_coolant_props,
    build_category_maps,
    df_to_tensors,
    preprocess_immersion_data,
    preprocess_immersion_data_strided,
    train_template,
)


DATA_DIR = "ComData"
BASE_DIR = os.path.join("GiTT_outputs", DATA_DIR)
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)


class ImmersionPINNNewPDE(BaseImmersionModel):
    def __init__(self, num_gpus: int, num_coolants: int, init_k: np.ndarray, init_cp: np.ndarray):
        super().__init__(num_gpus=num_gpus, num_coolants=num_coolants, embed_dim=4, num_numerical=5)
        k_raw = torch.tensor([_inv_softplus(v) for v in init_k], dtype=torch.float32)
        cp_raw = torch.tensor([_inv_softplus(v) for v in init_cp], dtype=torch.float32)
        self.k_raw = nn.Parameter(k_raw)
        self.cp_raw = nn.Parameter(cp_raw)

    def _mlp_out_dim(self) -> int:
        return 2

    def k_values(self, gpu_idx):
        return F.softplus(self.k_raw[gpu_idx]) + 1e-6

    def cp_values(self, coolant_idx):
        return F.softplus(self.cp_raw[coolant_idx]) + 1e-6

    def forward(self, gpu_idx, x_num):
        x_scaled, emb = self._encode(gpu_idx, x_num)
        out = self.model(torch.cat([emb, x_scaled], dim=-1))
        theta = torch.tanh(out[:, 0])
        alpha = F.softplus(out[:, 1]) + 1e-6
        theta = out[:, 0]
        t = x_num[:, 0]
        t0 = x_num[:, 3]
        t_pred = t0 + t * theta
        return t_pred, alpha, theta

    def physics_residual(self, gpu_idx, coolant_idx, x_num):
        x_req = x_num.clone().detach().requires_grad_(True)
        t_pred, alpha, _ = self.forward(gpu_idx, x_req)
        rho = x_req[:, 4]
        q = x_req[:, 1]
        cp = self.cp_values(coolant_idx)
        k = self.k_values(gpu_idx)
        grad_t = torch.autograd.grad(
            t_pred, x_req, grad_outputs=torch.ones_like(t_pred), create_graph=True, retain_graph=True
        )[0]
        d_t_dt = grad_t[:, 0]
        d_t_dz = grad_t[:, 2]
        inner = k * alpha * d_t_dz
        grad_inner = torch.autograd.grad(
            inner, x_req, grad_outputs=torch.ones_like(inner), create_graph=True, retain_graph=True
        )[0]
        d_dz_inner = grad_inner[:, 2]
        residual = alpha * rho * cp * d_t_dt - (-t_pred * k + q)
        return residual, t_pred, alpha, d_t_dz

    def loss_terms(self, gpu_idx, coolant_idx, x_num, t_true):
        residual, t_pred, alpha, d_t_dz = self.physics_residual(gpu_idx, coolant_idx, x_num)
        l_pde = torch.mean(residual ** 2)
        l_temp = torch.mean((t_pred - t_true) ** 2)
        return l_pde, l_temp, t_pred, alpha, d_t_dz

    def diagnostic_values(self, gpu_idx_all, x_num_all, n: int, device: str):
        self.eval()
        with torch.no_grad():
            if gpu_idx_all.shape[0] > n:
                perm = torch.randperm(gpu_idx_all.shape[0], device=device)[:n]
            else:
                perm = slice(None)
            _, alpha_s, theta_s = self.forward(gpu_idx_all[perm], x_num_all[perm])
        return {"theta_std": float(theta_s.std().item()), "alpha_std": float(alpha_s.std().item())}


GiTTModel = ImmersionPINNNewPDE


def build_loss_fn_pinn(w_pde: float, w_temp: float):
    def _loss(model, gb, cb, xb, yb, train: bool):
        l_pde, l_temp, _, _, _ = model.loss_terms(gb, cb, xb, yb)
        loss = w_pde * l_pde + w_temp * l_temp
        return {
            "loss": loss,
            "l_pde": l_pde,
            "l_temp": l_temp,
            "val_l_pde": l_pde,
            "val_l_temp": l_temp,
        }
    return _loss


def train_model(
    processed_csv_path: str,
    coolant_json_path: str,
    model_save_path: str,
    meta_save_path: str,
    learned_params_json_path: str,
    loss_csv_path: str,
    steps: int = 600,
    batch_size: int = 256,
    lr: float = 1e-3,
    w_pde: float = 1.0,
    w_temp: float = 1.0,
):
    def _build_model(maps: CategoryMaps) -> ImmersionPINNNewPDE:
        return ImmersionPINNNewPDE(
            num_gpus=len(maps.gpu_to_idx),
            num_coolants=len(maps.coolant_to_idx),
            init_k=maps.init_k,
            init_cp=maps.init_cp,
        )

    def _extra_save(model: ImmersionPINNNewPDE, maps: CategoryMaps):
        learned = {
            "k_by_gpu": {maps.idx_to_gpu[i]: float(F.softplus(model.k_raw[i]).item()) for i in maps.idx_to_gpu},
            "cp_by_coolant": {
                maps.idx_to_coolant[i]: float(F.softplus(model.cp_raw[i]).item()) for i in maps.idx_to_coolant
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(learned_params_json_path)) or ".", exist_ok=True)
        with open(learned_params_json_path, "w", encoding="utf-8") as f:
            json.dump(learned, f, indent=2)

    loss_csv_header = ["step", "loss", "l_pde", "l_temp", "val_l_pde", "val_l_temp", "theta_std", "alpha_std"]

    def _log_cols():
        return ["l_pde", "l_temp"]

    model, maps, val_df = train_template(
        processed_csv_path=processed_csv_path,
        coolant_json_path=coolant_json_path,
        model_save_path=model_save_path,
        meta_save_path=meta_save_path,
        loss_csv_path=loss_csv_path,
        build_model_fn=_build_model,
        build_loss_fn=build_loss_fn_pinn(w_pde, w_temp),
        build_csv_header_fn=lambda: loss_csv_header,
        build_log_cols_fn=_log_cols,
        loss_csv_header=loss_csv_header,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        grad_clip=1.0,
        seed=42,
        log_every=20,
        extra_model_save_fn=_extra_save,
    )
    return model, maps, val_df


def load_model_for_inference(model_path: str, meta_path: str, learned_path: str, device: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    gpu_to_idx = meta["gpu_to_idx"]
    coolant_to_idx = meta["coolant_to_idx"]
    idx_to_gpu = {v: k for k, v in gpu_to_idx.items()}
    idx_to_coolant = {v: k for k, v in coolant_to_idx.items()}
    with open(learned_path, "r", encoding="utf-8") as f:
        learned = json.load(f)
    init_k = np.array([learned["k_by_gpu"][idx_to_gpu[i]] for i in range(len(idx_to_gpu))], dtype=np.float32)
    init_cp = np.array(
        [learned["cp_by_coolant"][idx_to_coolant[i]] for i in range(len(idx_to_coolant))], dtype=np.float32
    )
    model = ImmersionPINNNewPDE(len(gpu_to_idx), len(coolant_to_idx), init_k, init_cp).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    maps = CategoryMaps(gpu_to_idx, idx_to_gpu, coolant_to_idx, idx_to_coolant, init_k, init_cp)
    return model, maps


def run_inference(model, maps: CategoryMaps, data_df: pd.DataFrame, output_csv_path: str):
    device = next(model.parameters()).device
    g_idx, c_idx, x_num, t_true = df_to_tensors(data_df, maps, device)
    model.eval()
    with torch.enable_grad():
        x_req = x_num.clone().detach().requires_grad_(True)
        t_pred, alpha, theta = model.forward(g_idx, x_req)
        grad_t = torch.autograd.grad(
            t_pred, x_req, grad_outputs=torch.ones_like(t_pred), create_graph=False, retain_graph=False
        )[0]
        d_t_dt = grad_t[:, 0]
    out_df = data_df.copy()
    out_df["predicted_temperature"] = t_pred.detach().cpu().numpy()
    out_df["predicted_ratio_alpha"] = alpha.detach().cpu().numpy()
    out_df["predicted_theta"] = theta.detach().cpu().numpy()
    out_df["dT_dt"] = d_t_dt.detach().cpu().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)) or ".", exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)
    pred_std = float(out_df["predicted_temperature"].std())
    pred_min = float(out_df["predicted_temperature"].min())
    pred_max = float(out_df["predicted_temperature"].max())
    mse = float(np.mean((out_df["predicted_temperature"].values - out_df["temperature"].values) ** 2))
    print(f"inference mse={mse:.6f}")
    print(f"predicted_temperature std={pred_std:.6f} min={pred_min:.6f} max={pred_max:.6f}")
    print(
        f"theta std={out_df['predicted_theta'].std():.6f} "
        f"alpha std={out_df['predicted_ratio_alpha'].std():.6f}"
    )
    if pred_std <= 1e-6:
        print("warning: predicted temperature has near-zero variation")


def main():
    input_dir = os.path.join("immersion_data", "PINN")
    coolant_json_path = os.path.join("PINN", "coolant.json")
    processed_csv_path = os.path.join("utlized data", "train_data", DATA_DIR + ".csv")
    model_path = os.path.join(BASE_DIR, "immersion_pinn_new_pde.pth")
    meta_path = os.path.join(BASE_DIR, "model_meta.json")
    learned_path = os.path.join(BASE_DIR, "learned_k_cp.json")
    loss_csv = os.path.join(BASE_DIR, "loss_history.csv")
    infer_csv = os.path.join(BASE_DIR, "inference_results.csv")
    print(f"preprocessed csv saved: {processed_csv_path}")
    model, maps, val_df = train_model(
        processed_csv_path=processed_csv_path,
        coolant_json_path=coolant_json_path,
        model_save_path=model_path,
        meta_save_path=meta_path,
        learned_params_json_path=learned_path,
        loss_csv_path=loss_csv,
        steps=600,
        batch_size=256,
        lr=1e-3,
        w_pde=1e-2,
        w_temp=1.0,
    )
    print(f"trained model saved: {model_path}")
    print(f"learned k/cp json saved: {learned_path}")
    run_inference(model, maps, val_df, infer_csv)
    print(f"inference csv saved: {infer_csv}")


def inference(processed_csv_path: str, output_csv_path: str, BASE_DIR: str):
    model_path = os.path.join(BASE_DIR, "immersion_pinn_new_pde.pth")
    meta_path = os.path.join(BASE_DIR, "model_meta.json")
    learned_path = os.path.join(BASE_DIR, "learned_k_cp.json")
    infer_csv = output_csv_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, maps = load_model_for_inference(model_path, meta_path, learned_path, device)
    gpu_type = list(maps.gpu_to_idx.keys())[0]
    coolant_type = list(maps.coolant_to_idx.keys())[0]
    coolant_density = 1601.0
    src_temp_values = None
    src_df = None
    if processed_csv_path and os.path.exists(processed_csv_path):
        src_df = pd.read_csv(processed_csv_path)
        if not src_df.empty:
            if "gpu_type" in src_df.columns:
                gpu_type = str(src_df["gpu_type"].iloc[0])
            if "coolant_type" in src_df.columns:
                coolant_type = str(src_df["coolant_type"].iloc[0])
            if "coolant_density" in src_df.columns:
                coolant_density = float(src_df["coolant_density"].iloc[0])
            if "temperature" in src_df.columns:
                src_temp_values = src_df["temperature"].astype(np.float32).head(300).to_numpy()

    time_values = np.arange(0.0, 30.0, 0.1, dtype=np.float32)
    n = len(time_values)
    initial_temperature = 82.9
    gpu_power = 252.6
    if src_temp_values is None or len(src_temp_values) == 0:
        src_temp_values = np.full(n, initial_temperature, dtype=np.float32)
    elif len(src_temp_values) < n:
        pad = np.full(n - len(src_temp_values), src_temp_values[-1], dtype=np.float32)
        src_temp_values = np.concatenate([src_temp_values, pad], axis=0)
    else:
        src_temp_values = src_temp_values[:n]

    _ = pd.DataFrame(
        {
            "gpu_type": [gpu_type] * n,
            "coolant_type": [coolant_type] * n,
            "coolant_key": [coolant_type] * n,
            "time": time_values,
            "gpu_power": np.full(n, gpu_power, dtype=np.float32),
            "z": np.zeros(n, dtype=np.float32),
            "initial_temperature": np.full(n, initial_temperature, dtype=np.float32),
            "coolant_density": np.full(n, coolant_density, dtype=np.float32),
            "temperature": src_temp_values,
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
        BASE_DIR = os.path.join("GiTT_outputs", dir)
        if not os.path.exists(BASE_DIR):
            os.makedirs(BASE_DIR)
        # processed_csv_path = r"utlized data\test_data\2100A_BERT.csv"
        output_csv_path = os.path.join(BASE_DIR, "test_data_results.csv")
        inference(processed_csv_path, output_csv_path, BASE_DIR)


if __name__ == "__main__":
    main()