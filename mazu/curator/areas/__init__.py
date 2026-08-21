from dataclasses import dataclass
from typing import Callable


@dataclass
class Area:
    name: str
    min_interval_days: int
    # signal_fn(root) -> int; 0 means "nothing to do", no API call is worth spending
    # on this pass. Only called when the caller hasn't already forced this area via
    # --area/--full, so an always-inert unconfigured Curator never even opens a store.
    signal_fn: Callable[..., int]
    task_prompt: str
