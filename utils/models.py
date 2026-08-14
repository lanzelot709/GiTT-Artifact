import csv
import math
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from utils.data_utils import (
    INPUT_ORDER,
    CategoryMaps,
    _load_coolant_props,
    build_category_maps,
    df_to_tensors,
    preprocess_immersion_data,
    preprocess_immersion_data_strided,
)


def _inv_softplus(x: float) -> float:
    x = max(float(x), 1e-6)
    if x > 20.0:
        return x
    return math.log(math.exp(x) - 1.0)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=2, hidden=128, depth=5):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden] * (depth - 1) + [out_dim]
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.Tanh()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class BaseImmersionModel(nn.Module):
    def __init__(self, num_gpus: int, num_coolants: int, embed_dim: int = 4, num_numerical: int = 5):
        super().__init__()
        self.num_numerical = num_numerical
        self.gpu_embed = nn.Embedding(num_gpus, embed_dim)
        self.register_buffer("x_mean", torch.zeros(num_numerical, dtype=torch.float32))
        self.register_buffer("x_std", torch.ones(num_numerical, dtype=torch.float32))
        self._embed_dim = embed_dim
        mlp_in_dim = embed_dim + num_numerical
        self.model = MLP(in_dim=mlp_in_dim, out_dim=self._mlp_out_dim(), hidden=128, depth=6)

    def _mlp_out_dim(self) -> int:
        raise NotImplementedError

    def set_feature_stats(self, x_num):
        mean = x_num.mean(dim=0)
        std = x_num.std(dim=0).clamp(min=1e-6)
        self.x_mean.copy_(mean.detach())
        self.x_std.copy_(std.detach())

    def _encode(self, gpu_idx, x_num):
        x_scaled = (x_num - self.x_mean) / self.x_std
        emb = self.gpu_embed(gpu_idx)
        return x_scaled, emb

    def loss_terms(self, gpu_idx, coolant_idx, x_num, t_true):
        raise NotImplementedError


def _build_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr)


def _build_scheduler(opt, steps):
    return optim.lr_scheduler.StepLR(opt, step_size=max(steps // 3, 1), gamma=0.5)


def train_template(
    processed_csv_path: str,
    coolant_json_path: str,
    model_save_path: str,
    meta_save_path: str,
    loss_csv_path: str,
    build_model_fn,
    build_loss_fn,
    build_csv_header_fn,
    build_log_cols_fn,
    loss_csv_header: list,
    steps: int = 600,
    batch_size: int = 256,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    seed: int = 42,
    log_every: int = 20,
    extra_model_save_fn=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(processed_csv_path)
    maps = build_category_maps(df, coolant_json_path)
    np.random.seed(seed)
    idx = np.arange(len(df))
    np.random.shuffle(idx)
    split = int(len(df) * 0.8)
    train_df = df.iloc[idx[:split]].reset_index(drop=True)
    val_df = df.iloc[idx[split:]].reset_index(drop=True)
    g_tr, c_tr, x_tr, y_tr = df_to_tensors(train_df, maps, device)
    g_va, c_va, x_va, y_va = df_to_tensors(val_df, maps, device)

    model = build_model_fn(maps)
    model = model.to(device)
    model.set_feature_stats(x_tr)

    opt = _build_optimizer(model, lr)
    sched = _build_scheduler(opt, steps)

    os.makedirs(os.path.dirname(os.path.abspath(loss_csv_path)) or ".", exist_ok=True)
    with open(loss_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(loss_csv_header)

    n_train = x_tr.shape[0]
    for step in range(1, steps + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        totals = {k: 0.0 for k in build_log_cols_fn()}
        n_batches = 0
        for s in range(0, n_train, batch_size):
            b = perm[s : s + batch_size]
            gb = g_tr[b]
            cb = c_tr[b]
            xb = x_tr[b]
            yb = y_tr[b]
            opt.zero_grad()
            loss_dict = build_loss_fn(model, gb, cb, xb, yb, train=True)
            loss = loss_dict["loss"]
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total_loss += float(loss.detach().item())
            for k, v in loss_dict.items():
                if k in totals:
                    totals[k] += float(v.detach().item())
            n_batches += 1
        sched.step()
        if step % log_every == 0 or step == 1 or step == steps:
            model.eval()
            with torch.no_grad():
                val_loss_dict = build_loss_fn(model, g_va, c_va, x_va, y_va, train=False)
            avg_loss = total_loss / max(n_batches, 1)
            avgs = {k: (totals[k] / max(n_batches, 1)) for k in build_log_cols_fn()}
            vals = {k: float(val_loss_dict[k].detach().item()) for k in build_log_cols_fn() if k in val_loss_dict}
            extra_line = []
            extra_csv = []
            if hasattr(model, "diagnostic_values"):
                diag = model.diagnostic_values(g_tr, x_tr, n=min(512, n_train), device=device)
                extra_line = [f"{k}={v:.6e}" for k, v in diag.items()]
                extra_csv = [v for v in diag.values()]
            line_parts = [f"[{step}/{steps}] loss={avg_loss:.6e}"]
            for k in build_log_cols_fn():
                line_parts.append(f"{k}={avgs[k]:.6e}")
                val_k = f"val_{k}"
                if val_k in vals:
                    line_parts.append(f"val_{k}={vals[val_k]:.6e}")
            if extra_line:
                line_parts.extend(extra_line)
            print(" ".join(line_parts))
            with open(loss_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                row = [step, avg_loss]
                for k in build_log_cols_fn():
                    row.append(avgs[k])
                    val_k = f"val_{k}"
                    if val_k in vals:
                        row.append(vals[val_k])
                row.extend(extra_csv)
                writer.writerow(row)

    torch.save(model.state_dict(), model_save_path)
    meta = {
        "gpu_to_idx": maps.gpu_to_idx,
        "coolant_to_idx": maps.coolant_to_idx,
        "input_order": INPUT_ORDER,
    }
    os.makedirs(os.path.dirname(os.path.abspath(meta_save_path)) or ".", exist_ok=True)
    with open(meta_save_path, "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f, indent=2)
    if extra_model_save_fn is not None:
        extra_model_save_fn(model, maps)
    return model, maps, val_df
