from dataclasses import dataclass, field
from threading import Lock


@dataclass
class Metrics:
    requests_total: int = 0
    requests_failed: int = 0
    rag_queries_total: int = 0
    rag_refusals_total: int = 0
    rag_grounded_total: int = 0
    retrieval_empty_total: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            if not hasattr(self, name):
                raise ValueError(f"Unknown metric: {name}")

            setattr(self, name, getattr(self, name) + amount)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "requests_failed": self.requests_failed,
                "rag_queries_total": self.rag_queries_total,
                "rag_refusals_total": self.rag_refusals_total,
                "rag_grounded_total": self.rag_grounded_total,
                "retrieval_empty_total": self.retrieval_empty_total,
            }


metrics = Metrics()
