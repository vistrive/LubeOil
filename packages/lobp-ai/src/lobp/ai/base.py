"""Base classes for AI/ML models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from lobp.core.config import settings

logger = structlog.get_logger()


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""

    name: str
    version: str
    created_at: datetime
    trained_samples: int
    accuracy: float
    parameters: dict[str, Any]


class BaseModel(ABC):
    """Abstract base class for all AI/ML models."""

    def __init__(self, model_name: str, version: str = "1.0.0"):
        self.model_name = model_name
        self.version = version
        self.model_path = Path(settings.ai_model_path) / model_name
        self._model: Any = None
        self._metadata: ModelMetadata | None = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def save(self, path: Path | None = None) -> None:
        """Save the model to disk."""
        pass

    @abstractmethod
    def load(self, path: Path | None = None) -> None:
        """Load the model from disk."""
        pass

    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """Get prediction confidence scores."""
        # Default implementation returns constant confidence
        return np.ones(len(X)) * 0.85

    def validate_input(self, X: np.ndarray) -> bool:
        """Validate input data."""
        if X is None or len(X) == 0:
            return False
        if np.isnan(X).any():
            return False
        return True


@dataclass
class PredictionResult:
    """Result of a prediction with confidence intervals."""

    value: float
    lower_bound: float
    upper_bound: float
    confidence: float
    model_name: str
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confidence": self.confidence,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }


@dataclass
class OptimizationResult:
    """Result of a recipe optimization."""

    original_ratios: dict[str, float]
    optimized_ratios: dict[str, float]
    predicted_quality: dict[str, float]
    cost_savings_percent: float
    confidence: float
    adjustments: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_ratios": self.original_ratios,
            "optimized_ratios": self.optimized_ratios,
            "predicted_quality": self.predicted_quality,
            "cost_savings_percent": self.cost_savings_percent,
            "confidence": self.confidence,
            "adjustments": self.adjustments,
        }
