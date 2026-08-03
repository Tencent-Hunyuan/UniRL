"""Lazy native FastVideo runtime boundary."""

from __future__ import annotations

from typing import Any, Callable, Dict


def _import_fastvideo_runtime() -> Dict[str, Any]:
    """Install UniRL patches before importing objects that spawn workers."""

    from unirl.rollout.engine.fastvideo._patches import FastVideoHijack

    FastVideoHijack.hijack()

    from fastvideo import VideoGenerator
    from fastvideo.fastvideo_args import FastVideoArgs

    return {"VideoGenerator": VideoGenerator, "FastVideoArgs": FastVideoArgs}


class FastVideoBackend:
    """Own FastVideo imports, worker boot, execution, and lifecycle."""

    def __init__(self, generator: Any, fastvideo_args: Any, runtime: Dict[str, Any]) -> None:
        self._generator = generator
        self.fastvideo_args = fastvideo_args
        self._runtime = runtime

    @classmethod
    def boot(
        cls,
        kwargs: Dict[str, Any],
        *,
        configure_args: Callable[[Any], None],
        reserve_port: Callable[[], int],
        max_port_attempts: int = 5,
    ) -> "FastVideoBackend":
        runtime = _import_fastvideo_runtime()
        fastvideo_args = runtime["FastVideoArgs"].from_kwargs(**kwargs)
        configure_args(fastvideo_args)
        generator = cls._start_generator(
            runtime,
            fastvideo_args,
            reserve_port=reserve_port,
            max_port_attempts=max_port_attempts,
        )
        return cls(generator, fastvideo_args, runtime)

    @staticmethod
    def _start_generator(
        runtime: Dict[str, Any],
        fastvideo_args: Any,
        *,
        reserve_port: Callable[[], int],
        max_port_attempts: int,
    ) -> Any:
        for attempt in range(1, max_port_attempts + 1):
            try:
                return runtime["VideoGenerator"].from_fastvideo_args(fastvideo_args)
            except Exception as exc:  # noqa: BLE001
                port_in_use = "eaddrinuse" in str(exc).lower() or "address already in use" in str(exc).lower()
                if not port_in_use or attempt == max_port_attempts:
                    raise
                fastvideo_args.master_port = int(reserve_port())
        raise RuntimeError("unreachable")

    def execute(self, forward_batch: Any) -> Any:
        if self._generator is None:
            raise RuntimeError("FastVideo backend is offloaded")
        return self._generator.executor.execute_forward(forward_batch, self.fastvideo_args)

    def update_weights_from_path(self, checkpoint_path: str) -> None:
        if self._generator is None:
            raise RuntimeError("FastVideo backend is offloaded")
        self._generator.update_transformer_weights_from_path(checkpoint_path)

    def sleep(self) -> None:
        if self._generator is not None:
            self._generator.shutdown()
            self._generator = None

    def wake(self, *, reserve_port: Callable[[], int], max_port_attempts: int = 5) -> None:
        if self._generator is not None:
            return
        self.fastvideo_args.master_port = int(reserve_port())
        self._generator = self._start_generator(
            self._runtime,
            self.fastvideo_args,
            reserve_port=reserve_port,
            max_port_attempts=max_port_attempts,
        )

    def shutdown(self) -> None:
        self.sleep()


__all__ = ["FastVideoBackend"]
