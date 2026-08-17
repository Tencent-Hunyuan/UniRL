"""unirl Utilities - Miscellaneous utility functions."""

import gc
import importlib
import logging
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_function(path: str) -> Any:
    """Dynamically load a class or function from a module path."""
    if path is None or path == "":
        raise ValueError("Path cannot be None or empty")

    parts = path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid path format: {path}. Expected 'module.path.ClassName'")

    module_path, class_name = parts

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}': {e}")

    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise AttributeError(f"Module '{module_path}' has no attribute '{class_name}'")

    return cls


def _needs_h20_cu128_workaround() -> bool:
    if not torch.cuda.is_available() or torch.version.cuda != "12.8":
        return False
    if not torch.__version__.startswith("2.10.0"):
        return False
    return "H20" in torch.cuda.get_device_name(torch.cuda.current_device()).upper()


def set_seed(seed: Optional[int]) -> None:
    """Set random seeds, except for deterministic cuBLAS on the broken H20 torch-2.10/cu128 stack."""
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "big") & 0x7FFFFFFF
    h20_cu128 = _needs_h20_cu128_workaround()
    if h20_cu128:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = not h20_cu128
        torch.backends.cudnn.benchmark = h20_cu128
        if h20_cu128:
            # SDPA MATH uses the same broken strided-batched GEMM path.
            torch.backends.cuda.enable_math_sdp(False)
    torch.use_deterministic_algorithms(not h20_cu128, warn_only=not h20_cu128)


def configure_logger(
    level: int = logging.INFO,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None,
) -> None:
    """Configure logging for the training run."""
    handlers = [logging.StreamHandler()]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
    )

    logging.getLogger("ray").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def clear_memory() -> None:
    """Clear GPU memory cache with synchronization and GC."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def aggregate_numeric_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """Average numeric metric keys across repeated metric dictionaries."""
    aggregated: Dict[str, float] = {}
    if not metrics_list:
        return aggregated

    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())

    for key in all_keys:
        values: List[float] = []
        for metrics in metrics_list:
            if key not in metrics:
                continue
            value = metrics[key]
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.mean().item()
            if isinstance(value, bool):
                values.append(float(value))
            elif isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            aggregated[key] = sum(values) / len(values)

    return aggregated
