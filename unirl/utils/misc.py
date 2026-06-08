"""
unirl Utilities - Miscellaneous utility functions.
"""

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
    """
    Dynamically load a class or function from a module path.

    Args:
        path: Full path to the class/function, e.g., "unirl.algorithms.flowgrpo.FlowGRPO"

    Returns:
        The loaded class or function

    Example:
        >>> algo_cls = load_function("unirl.algorithms.flowgrpo.FlowGRPO")
    """
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


def set_seed(seed: Optional[int]) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value. ``None`` means "draw a fresh seed from OS
            entropy" — this run is internally deterministic (random.seed /
            np.random.seed / torch.manual_seed all share the same drawn int)
            but not reproducible across re-runs.

    Note:
        ``CUBLAS_WORKSPACE_CONFIG`` is set via ``setdefault`` as a
        belt-and-suspenders measure, but the real guarantee must come
        from setting it in the process environment BEFORE Python imports
        torch (e.g., via Ray ``runtime_env={"env_vars": ...}``). Once
        cuBLAS has initialized, changing this env var has no effect.
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "big") & 0x7FFFFFFF
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # warn_only lets non-deterministic ops fall back gracefully instead of hard-failing.
    torch.use_deterministic_algorithms(True, warn_only=True)


def configure_logger(
    level: int = logging.INFO,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for the training run.

    Args:
        level: Logging level
        format_str: Log format string
        log_file: Optional file path to write logs
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
    )

    # Reduce verbosity of some libraries
    logging.getLogger("ray").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def clear_memory() -> None:
    """Clear GPU memory cache with synchronization and GC."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """
    Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator between keys

    Returns:
        Flattened dictionary
    """
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
