"""Scorer package. Import registry to trigger lazy registration."""

from reward_service.scorers.base import BaseScorer, ScoreItem
from reward_service.scorers.registry import available_scorers, get_scorer_cls

__all__ = ["BaseScorer", "ScoreItem", "available_scorers", "get_scorer_cls"]
