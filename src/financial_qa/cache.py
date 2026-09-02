"""Resumable prediction cache backed by a JSONL file.

Predictions are appended as JSON lines keyed by (conversation_id, turn).
On resume, completed predictions are loaded and skipped — saving API cost
when iterating on a partially-completed evaluation run.
"""

import json
import logging
from pathlib import Path

from financial_qa.models.pipeline import PipelineResult

logger = logging.getLogger(__name__)


class PredictionCache:
    """Append-only JSONL cache for pipeline predictions.

    Args:
        path: Path to the JSONL cache file.  Created if it does not exist.

    Example:
        >>> cache = PredictionCache("cache/predictions.jsonl")
        >>> if not cache.has(conv_id, turn):
        ...     result = pipeline.run(...)
        ...     cache.store(result)
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._index: dict[tuple[str, int], PipelineResult] = {}
        self._load()

    def _load(self) -> None:
        """Load existing predictions from the JSONL file into memory."""
        if not self._path.exists():
            return
        with self._path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    result = PipelineResult(**data)
                    self._index[(result.conversation_id, result.turn)] = result
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    logger.warning("cache: skipping malformed line: %s", exc)

    def has(self, conversation_id: str, turn: int) -> bool:
        """Check whether a prediction exists for this conversation turn.

        Args:
            conversation_id: Conversation identifier.
            turn: 1-based turn index.

        Returns:
            True if a cached prediction exists.
        """
        return (conversation_id, turn) in self._index

    def get(self, conversation_id: str, turn: int) -> PipelineResult | None:
        """Retrieve a cached prediction.

        Args:
            conversation_id: Conversation identifier.
            turn: 1-based turn index.

        Returns:
            The cached PipelineResult, or None if not found.
        """
        return self._index.get((conversation_id, turn))

    def store(self, result: PipelineResult) -> None:
        """Append a prediction to the cache.

        Args:
            result: The PipelineResult to persist.
        """
        key = (result.conversation_id, result.turn)
        self._index[key] = result
        with self._path.open("a") as f:
            f.write(result.model_dump_json() + "\n")

    @property
    def size(self) -> int:
        """Number of predictions currently cached."""
        return len(self._index)
