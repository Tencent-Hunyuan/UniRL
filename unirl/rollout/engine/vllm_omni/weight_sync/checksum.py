"""Back-compat shim — module moved to the engine-neutral transfer package.

The real implementation lives at
``unirl.distributed.weight_sync.transfer.checksum`` (hoisted so
trainer-side senders and engine-side receivers share one copy without
importing an engine package). This shim keeps the v1 engine's import
paths working until it retires; new code imports the neutral path.
"""

from unirl.distributed.weight_sync.transfer.checksum import *  # noqa: F401,F403
