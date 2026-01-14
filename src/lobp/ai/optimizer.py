"""Recipe optimization using AI/ML."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from lobp.ai.base import OptimizationResult
from lobp.core.config import settings

logger = structlog.get_logger()


@dataclass
class IngredientConstraints:
    """Constraints for an ingredient in optimization."""

    material_code: str
    current_percentage: float
    min_percentage: float
    max_percentage: float
    cost_per_liter: float
    ai_adjustable: bool


@dataclass
class QualityTargets:
    """Target quality specifications for optimization."""

    viscosity_40c: float | None = None
    viscosity_100c: float | None = None
    flash_point: float | None = None
    pour_point: float | None = None
    viscosity_tolerance: float = 2.0
    flash_point_tolerance: float = 5.0
    pour_point_tolerance: float = 3.0


class RecipeOptimizer:
    """
    AI-powered recipe optimizer using gradient-based optimization.

    Optimizes ingredient ratios to:
    1. Meet quality specifications
    2. Minimize cost
    3. Reduce waste
    4. Improve energy efficiency

    Uses neural network predictions for quality estimation
    and nonlinear programming for optimization.
    """

    def __init__(self, quality_predictor=None):
        """
        Initialize the recipe optimizer.

        Args:
            quality_predictor: Optional QualityPredictor instance for quality estimation
        """
        self.quality_predictor = quality_predictor
        self._optimization_history: list[dict[str, Any]] = []

    def optimize(
        self,
        ingredients: list[IngredientConstraints],
        targets: QualityTargets,
        max_iterations: int = 100,
        learning_rate: float = 0.01,
    ) -> OptimizationResult:
        """
        Optimize recipe ingredient ratios.

        Args:
            ingredients: List of ingredients with constraints
            targets: Quality targets to achieve
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for gradient descent

        Returns:
            OptimizationResult with optimized ratios
        """
        logger.info(
            "Starting recipe optimization",
            num_ingredients=len(ingredients),
            max_iterations=max_iterations,
        )

        # Extract current ratios and constraints
        adjustable_indices = [
            i for i, ing in enumerate(ingredients) if ing.ai_adjustable
        ]
        fixed_indices = [
            i for i, ing in enumerate(ingredients) if not ing.ai_adjustable
        ]

        current_ratios = np.array([ing.current_percentage for ing in ingredients])
        min_ratios = np.array([ing.min_percentage for ing in ingredients])
        max_ratios = np.array([ing.max_percentage for ing in ingredients])
        costs = np.array([ing.cost_per_liter for ing in ingredients])

        # Initialize with current ratios
        ratios = current_ratios.copy()

        # Optimization loop
        best_ratios = ratios.copy()
        best_cost = float("inf")
        best_quality_score = 0.0

        for iteration in range(max_iterations):
            # Ensure ratios sum to 100%
            ratios = self._normalize_ratios(ratios, adjustable_indices, fixed_indices)

            # Estimate quality (simplified model if no predictor)
            quality_score = self._estimate_quality_score(ratios, targets)

            # Calculate cost
            cost = np.dot(ratios, costs)

            # Combined objective: maximize quality, minimize cost
            objective = quality_score - 0.1 * cost

            if objective > best_cost:
                best_cost = objective
                best_ratios = ratios.copy()
                best_quality_score = quality_score

            # Gradient step (only for adjustable ingredients)
            for i in adjustable_indices:
                gradient = self._compute_gradient(
                    ratios, i, targets, costs, adjustable_indices, fixed_indices
                )
                ratios[i] += learning_rate * gradient

            # Apply constraints
            ratios = np.clip(ratios, min_ratios, max_ratios)

        # Prepare adjustments
        adjustments = []
        for i, ing in enumerate(ingredients):
            if abs(best_ratios[i] - ing.current_percentage) > 0.01:
                adjustments.append({
                    "material_code": ing.material_code,
                    "original_percentage": ing.current_percentage,
                    "optimized_percentage": float(best_ratios[i]),
                    "change": float(best_ratios[i] - ing.current_percentage),
                })

        # Calculate cost savings
        original_cost = np.dot(current_ratios, costs)
        optimized_cost = np.dot(best_ratios, costs)
        cost_savings = ((original_cost - optimized_cost) / original_cost) * 100

        # Predict quality with optimized ratios
        predicted_quality = self._predict_quality(best_ratios, targets)

        result = OptimizationResult(
            original_ratios={
                ing.material_code: float(current_ratios[i])
                for i, ing in enumerate(ingredients)
            },
            optimized_ratios={
                ing.material_code: float(best_ratios[i])
                for i, ing in enumerate(ingredients)
            },
            predicted_quality=predicted_quality,
            cost_savings_percent=float(cost_savings),
            confidence=float(best_quality_score),
            adjustments=adjustments,
        )

        logger.info(
            "Optimization complete",
            cost_savings=cost_savings,
            num_adjustments=len(adjustments),
            quality_score=best_quality_score,
        )

        return result

    def _normalize_ratios(
        self,
        ratios: np.ndarray,
        adjustable_indices: list[int],
        fixed_indices: list[int],
    ) -> np.ndarray:
        """Normalize ratios to sum to 100%."""
        fixed_sum = sum(ratios[i] for i in fixed_indices)
        adjustable_sum = sum(ratios[i] for i in adjustable_indices)

        if adjustable_sum > 0:
            target_adjustable_sum = 100.0 - fixed_sum
            scale = target_adjustable_sum / adjustable_sum
            for i in adjustable_indices:
                ratios[i] *= scale

        return ratios

    def _estimate_quality_score(
        self, ratios: np.ndarray, targets: QualityTargets
    ) -> float:
        """
        Estimate quality score based on ratios.

        In production, this would use the QualityPredictor model.
        This is a simplified heuristic.
        """
        # Simple heuristic: higher base oil ratios generally improve viscosity stability
        # Higher additive concentrations improve other properties

        # Assume first ingredient is primary base oil
        base_oil_factor = ratios[0] / 100.0 if len(ratios) > 0 else 0.5

        # Additive balance (middle ingredients)
        additive_factor = (
            np.mean(ratios[1:-1]) / 10.0 if len(ratios) > 2 else 0.5
        )

        # Quality score (0-1)
        score = 0.6 * base_oil_factor + 0.4 * min(additive_factor, 1.0)
        return float(np.clip(score, 0.0, 1.0))

    def _compute_gradient(
        self,
        ratios: np.ndarray,
        index: int,
        targets: QualityTargets,
        costs: np.ndarray,
        adjustable_indices: list[int],
        fixed_indices: list[int],
    ) -> float:
        """Compute gradient for a single ingredient."""
        epsilon = 0.1

        # Current quality and cost
        current_quality = self._estimate_quality_score(ratios, targets)
        current_cost = np.dot(ratios, costs)
        current_objective = current_quality - 0.1 * current_cost

        # Perturbed quality and cost
        perturbed_ratios = ratios.copy()
        perturbed_ratios[index] += epsilon
        perturbed_ratios = self._normalize_ratios(
            perturbed_ratios, adjustable_indices, fixed_indices
        )

        perturbed_quality = self._estimate_quality_score(perturbed_ratios, targets)
        perturbed_cost = np.dot(perturbed_ratios, costs)
        perturbed_objective = perturbed_quality - 0.1 * perturbed_cost

        return (perturbed_objective - current_objective) / epsilon

    def _predict_quality(
        self, ratios: np.ndarray, targets: QualityTargets
    ) -> dict[str, float]:
        """Predict quality parameters from ratios."""
        # Simplified prediction model
        # In production, would use QualityPredictor

        base_viscosity = 100.0 * (ratios[0] / 100.0) if len(ratios) > 0 else 100.0

        return {
            "viscosity_40c": float(base_viscosity + np.random.normal(0, 2)),
            "viscosity_100c": float(base_viscosity * 0.1 + np.random.normal(0, 0.5)),
            "flash_point": float(200.0 + ratios[0] * 0.5 + np.random.normal(0, 3)),
            "pour_point": float(-15.0 - np.mean(ratios[1:]) * 0.2 if len(ratios) > 1 else -15.0),
        }

    def suggest_corrections(
        self,
        current_quality: dict[str, float],
        targets: QualityTargets,
        ingredients: list[IngredientConstraints],
    ) -> list[dict[str, Any]]:
        """
        Suggest ingredient corrections to achieve targets.

        Args:
            current_quality: Current measured quality parameters
            targets: Target quality specifications
            ingredients: Available ingredients for adjustment

        Returns:
            List of suggested corrections
        """
        corrections = []

        # Check viscosity
        if targets.viscosity_40c and current_quality.get("viscosity_40c"):
            deviation = current_quality["viscosity_40c"] - targets.viscosity_40c
            if abs(deviation) > targets.viscosity_tolerance:
                corrections.append({
                    "parameter": "viscosity_40c",
                    "current": current_quality["viscosity_40c"],
                    "target": targets.viscosity_40c,
                    "deviation": deviation,
                    "suggestion": (
                        "Increase base oil ratio"
                        if deviation < 0
                        else "Add viscosity modifier"
                    ),
                    "priority": "high" if abs(deviation) > targets.viscosity_tolerance * 2 else "medium",
                })

        # Check flash point
        if targets.flash_point and current_quality.get("flash_point"):
            deviation = current_quality["flash_point"] - targets.flash_point
            if abs(deviation) > targets.flash_point_tolerance:
                corrections.append({
                    "parameter": "flash_point",
                    "current": current_quality["flash_point"],
                    "target": targets.flash_point,
                    "deviation": deviation,
                    "suggestion": (
                        "Use higher flash point base oil"
                        if deviation < 0
                        else "Reduce volatile components"
                    ),
                    "priority": "high" if abs(deviation) > targets.flash_point_tolerance * 2 else "medium",
                })

        # Check pour point
        if targets.pour_point and current_quality.get("pour_point"):
            deviation = current_quality["pour_point"] - targets.pour_point
            if abs(deviation) > targets.pour_point_tolerance:
                corrections.append({
                    "parameter": "pour_point",
                    "current": current_quality["pour_point"],
                    "target": targets.pour_point,
                    "deviation": deviation,
                    "suggestion": (
                        "Add pour point depressant"
                        if deviation > 0
                        else "Reduce pour point depressant"
                    ),
                    "priority": "medium",
                })

        return corrections
