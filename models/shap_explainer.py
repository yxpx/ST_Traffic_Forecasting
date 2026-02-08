from typing import Callable, Dict

import numpy as np
import shap
import torch


def build_predict_fn(model: torch.nn.Module, edge_index: torch.Tensor, node_index: int) -> Callable:
    def _predict(batch_np: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            x = torch.tensor(batch_np, dtype=torch.float32)
            pred = model(x, edge_index)
            node_pred = pred[:, node_index, :]
            return node_pred.cpu().numpy()

    return _predict


def explain_local(
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    background: np.ndarray,
    sample: np.ndarray,
    node_index: int,
) -> Dict[str, np.ndarray]:
    predict_fn = build_predict_fn(model, edge_index, node_index)
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(sample, nsamples=100)
    return {
        "shap_values": shap_values,
        "expected_value": explainer.expected_value,
    }
