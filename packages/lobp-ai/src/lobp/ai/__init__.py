"""
AI/ML Module for LOBP Control System

Provides machine learning capabilities for:
- Recipe optimization using neural networks
- Multi-supplier cost optimization
- Predictive quality control
- Dynamic rerouting and rescheduling
- Adaptive learning from lab feedback
- Energy optimization
- Cross-recipe learning and transfer
"""

from lobp.ai.predictor import QualityPredictor
from lobp.ai.optimizer import RecipeOptimizer, IngredientConstraints, QualityTargets
from lobp.ai.scheduler import BlendScheduler
from lobp.ai.multi_supplier_optimizer import (
    MultiSupplierRecipeOptimizer,
    MultiSupplierIngredient,
    SupplierOption,
    OptimizedRecipeResult,
    OptimizedIngredient,
    PredictedQuality,
)
from lobp.ai.cross_recipe_learning import CrossRecipeLearning
from lobp.ai.soft_sensors import ViscositySoftSensor
from lobp.ai.digital_twin import DigitalTwin
from lobp.ai.dynamic_scheduler import DynamicScheduler
from lobp.ai.nlp_interface import NLPInterface

__all__ = [
    # Core AI
    "QualityPredictor",
    "RecipeOptimizer",
    "BlendScheduler",
    # Multi-supplier optimization
    "MultiSupplierRecipeOptimizer",
    "MultiSupplierIngredient",
    "SupplierOption",
    "OptimizedRecipeResult",
    "OptimizedIngredient",
    "PredictedQuality",
    # Data classes
    "IngredientConstraints",
    "QualityTargets",
    # Advanced features
    "CrossRecipeLearning",
    "ViscositySoftSensor",
    "DigitalTwin",
    "DynamicScheduler",
    "NLPInterface",
]
