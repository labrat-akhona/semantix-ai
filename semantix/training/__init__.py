"""Training data collection for continuous model improvement."""

from semantix.training.collector import TrainingCollector

_default_collector: TrainingCollector | None = None


def set_default_collector(collector: TrainingCollector | None) -> None:
    """Set the global default training data collector."""
    global _default_collector
    _default_collector = collector


def get_default_collector() -> TrainingCollector | None:
    """Return the global default training data collector, or None."""
    return _default_collector


__all__ = ["TrainingCollector", "set_default_collector", "get_default_collector"]
