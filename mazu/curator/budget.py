"""Single-threaded cost cap for a Curator pass -- same "the round that trips the
cap still completes" semantics as mazu.agent.council._SharedCostTracker, minus the
lock: Curator's own loop (mazu/curator/loop.py) runs one area at a time in a plain
for loop, never in parallel worker threads, so there's no concurrent-update race to
guard against here.
"""


class CuratorBudget:
    def __init__(self, max_cost: float | None):
        self._max_cost = max_cost
        self._total = 0.0

    def add_and_check(self, cost: float | None) -> bool:
        """Adds cost (if trackable) and returns whether the budget is now
        exhausted. A None max_cost makes this a permanent no-op."""
        if cost is not None:
            self._total += cost
        return self._max_cost is not None and self._total >= self._max_cost

    def is_exhausted(self) -> bool:
        return self._max_cost is not None and self._total >= self._max_cost

    @property
    def total(self) -> float:
        return self._total
