"""CLI entrypoint: `python -m reward_service --config configs/service.yaml`."""

from __future__ import annotations

import argparse

import uvicorn

from reward_service.config import load_config
from reward_service.logging_utils import get_logger
from reward_service.server import create_app

logger = get_logger(__name__)

# Match reward_service.logging_utils so uvicorn's access / error / default
# loggers print with the same timestamp + level prefix as our own logs —
# stdout stays a single uniform stream even under `python -m reward_service
# > logs/service.log`. Without this, uvicorn emits its own `INFO: ...` format
# and lines like `"POST /score HTTP/1.1" 200 OK` land without timestamps, so
# you can't tell when each request arrived.
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

UVICORN_LOG_CONFIG: dict = {
    "version": 1,
    # Must stay False so reward_service.* loggers already configured by
    # logging_utils._configure_root survive uvicorn's dictConfig call.
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": _LOG_FORMAT, "datefmt": _DATE_FORMAT},
        "access":  {"format": _LOG_FORMAT, "datefmt": _DATE_FORMAT},
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "class": "logging.StreamHandler",
            "formatter": "access",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error":  {"level": "INFO"},
        "uvicorn.access": {"handlers": ["access"],  "level": "INFO", "propagate": False},
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward Service")
    parser.add_argument("--config", required=True, help="path to service YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    app = create_app(cfg)
    logger.info("starting uvicorn on %s:%d", cfg.server.host, cfg.server.port)
    uvicorn.run(
        app,
        host=cfg.server.host,
        port=cfg.server.port,
        log_config=UVICORN_LOG_CONFIG,
    )


if __name__ == "__main__":
    main()
