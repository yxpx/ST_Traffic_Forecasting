import argparse
import csv
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.stgcn_model import SimpleSTGCN, train_one_epoch
from src.data_loader import create_feature_bundle


class SlidingWindowDataset(Dataset):
    """Generates sliding windows on-the-fly to avoid materializing the full array."""

    def __init__(self, features: np.ndarray, target: np.ndarray, window: int, horizon: int):
        self.features = features.astype(np.float32)
        self.target = target.astype(np.float32)
        self.window = window
        self.horizon = horizon
        self.length = features.shape[0] - window - horizon + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window]
        y = self.target[idx + self.window : idx + self.window + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple ST-GCN baseline.")
    parser.add_argument("--h5", default="data/raw/METR-LA.h5")
    parser.add_argument("--adj", default="data/raw/adj_METR-LA.pkl")
    parser.add_argument("--weather", default="data/raw/weather_CA_2019.csv")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--save", default="models/stgcn_weights.pt")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--observable-out", default="frontend-observable/src/metrics.csv")
    args = parser.parse_args()

    print("Loading data...", flush=True)
    bundle = create_feature_bundle(args.h5, args.adj, args.weather)
    print(f"Features shape: {bundle.features.shape}, Target shape: {bundle.target.shape}", flush=True)

    total_samples = bundle.features.shape[0] - args.window - args.horizon + 1
    split = int(total_samples * 0.8)

    train_ds = SlidingWindowDataset(
        bundle.features[: split + args.window + args.horizon - 1],
        bundle.target[: split + args.window + args.horizon - 1],
        args.window, args.horizon,
    )
    val_ds = SlidingWindowDataset(
        bundle.features[split :],
        bundle.target[split :],
        args.window, args.horizon,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=0)
    edge_index = torch.tensor(bundle.edge_index, dtype=torch.long)

    model = SimpleSTGCN(in_features=bundle.features.shape[-1], hidden_size=32, horizon=args.horizon)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Training: {len(train_ds)} samples, Validation: {len(val_ds)} samples", flush=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        n_batches = len(train_loader)
        for i, (xb, yb) in enumerate(train_loader):
            # Skip batches with NaN/Inf
            if torch.isnan(xb).any() or torch.isnan(yb).any():
                continue
            loss = train_one_epoch(model, optimizer, xb, yb, edge_index)
            if not np.isnan(loss):
                train_losses.append(loss)
            if (i + 1) % 50 == 0 or (i + 1) == n_batches:
                print(f"  epoch {epoch} batch {i+1}/{n_batches} loss={loss:.4f}", flush=True)

        model.eval()
        val_losses, val_mae, val_rmse = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb, edge_index)
                pred_y = pred.permute(0, 2, 1)
                loss = ((pred_y - yb) ** 2).mean().item()
                mae = (pred_y - yb).abs().mean().item()
                rmse = float(torch.sqrt(((pred_y - yb) ** 2).mean()).item())
                val_losses.append(loss)
                val_mae.append(mae)
                val_rmse.append(rmse)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
            "val_mae": float(np.mean(val_mae)) if val_mae else float("nan"),
            "val_rmse": float(np.mean(val_rmse)) if val_rmse else float("nan"),
        }
        history.append(row)
        print(
            "epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mae={val_mae:.4f} val_rmse={val_rmse:.4f}".format(
                **row
            ), flush=True
        )

    os.makedirs(args.report_dir, exist_ok=True)
    csv_path = os.path.join(args.report_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    observable_path = args.observable_out
    if observable_path:
        os.makedirs(os.path.dirname(observable_path), exist_ok=True)
        with open(observable_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save(model.state_dict(), args.save)
    print(f"saved weights to {args.save}")


if __name__ == "__main__":
    main()
