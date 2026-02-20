"""
AI Model Versioning and Retraining Management Service.

Implements the model lifecycle management described in Section 11 of the
AI Recipe Optimization document:
- Monthly rapid assessment (1 week effort)
- Quarterly full retraining (2 weeks effort)
- Annual comprehensive review (3-4 weeks effort)
- Performance monitoring and degradation detection
- Model versioning and rollback capabilities

Ensures AI models remain accurate and relevant over time.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from lobp.models.ai_models import ModelStatus, RetrainingFrequency

logger = structlog.get_logger()


class RetrainingTrigger(str, Enum):
    """Reason for triggering model retraining."""

    SCHEDULED_MONTHLY = "scheduled_monthly"
    SCHEDULED_QUARTERLY = "scheduled_quarterly"
    SCHEDULED_ANNUAL = "scheduled_annual"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    MANUAL_REQUEST = "manual_request"
    NEW_PRODUCT = "new_product"
    PROCESS_CHANGE = "process_change"


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a model."""

    model_version_id: str
    evaluation_date: datetime
    period_days: int

    # Prediction accuracy
    total_predictions: int
    verified_predictions: int
    accuracy_percent: float

    # Quality prediction metrics
    viscosity_mae: float  # Mean Absolute Error
    viscosity_rmse: float  # Root Mean Square Error
    tbn_mae: float
    flash_point_mae: float
    pour_point_mae: float

    # Off-spec detection
    true_positives: int  # Correctly predicted off-spec
    true_negatives: int  # Correctly predicted on-spec
    false_positives: int  # Incorrectly predicted off-spec
    false_negatives: int  # Missed off-spec (most critical)
    off_spec_detection_rate: float
    false_alarm_rate: float

    # Confidence calibration
    avg_confidence: float
    confidence_calibration_error: float


@dataclass
class RetrainingRecommendation:
    """Recommendation for model retraining."""

    model_name: str
    model_type: str
    current_version: str
    recommendation: str  # retrain_now, schedule_retrain, monitor, no_action
    trigger: RetrainingTrigger
    urgency: str  # critical, high, medium, low

    # Supporting data
    current_accuracy: float
    target_accuracy: float
    accuracy_trend: str  # improving, stable, degrading
    days_since_last_training: int
    new_batches_since_training: int

    # Recommended action
    recommended_action: str
    estimated_effort_hours: float
    recommended_date: datetime | None = None

    # Additional context
    notes: list[str] = field(default_factory=list)


@dataclass
class ModelVersion:
    """Model version information."""

    version_id: str
    model_name: str
    model_type: str
    version: str
    version_number: int
    status: ModelStatus
    is_production: bool

    # Training info
    training_date: datetime
    training_samples: int
    training_accuracy: float

    # Files
    model_path: str
    weights_path: str | None
    checksum: str

    # Performance
    production_accuracy: float | None = None
    total_predictions: int = 0


class ModelManagementService:
    """
    Service for AI model versioning, monitoring, and retraining scheduling.

    Implements the continuous improvement cycle:
    - Monthly: Quick accuracy assessment
    - Quarterly: Full model retraining
    - Annual: Architecture review and optimization
    """

    def __init__(
        self,
        models_directory: str = "./models",
        accuracy_threshold: float = 90.0,  # Minimum acceptable accuracy
        degradation_threshold: float = 5.0,  # % drop to trigger retraining
    ):
        """
        Initialize model management service.

        Args:
            models_directory: Directory for storing model files
            accuracy_threshold: Minimum accuracy threshold (%)
            degradation_threshold: Accuracy drop threshold for retraining (%)
        """
        self.models_directory = Path(models_directory)
        self.accuracy_threshold = accuracy_threshold
        self.degradation_threshold = degradation_threshold

        # Create models directory if needed
        self.models_directory.mkdir(parents=True, exist_ok=True)

        # Track model versions in memory (would be database in production)
        self._model_versions: dict[str, list[ModelVersion]] = {}
        self._performance_history: dict[str, list[ModelPerformanceMetrics]] = {}

    def register_model_version(
        self,
        model_name: str,
        model_type: str,
        version: str,
        model_path: str,
        training_samples: int,
        training_accuracy: float,
        weights_path: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model_name: Name of the model (e.g., "quality_predictor")
            model_type: Type of model (e.g., "neural_network")
            version: Version string (e.g., "1.0.0")
            model_path: Path to model file
            training_samples: Number of training samples
            training_accuracy: Training accuracy (0-100)
            weights_path: Optional path to weights file
            hyperparameters: Optional hyperparameters dict

        Returns:
            Registered ModelVersion
        """
        # Calculate checksum
        checksum = self._calculate_checksum(model_path)

        # Determine version number
        existing = self._model_versions.get(model_name, [])
        version_number = len(existing) + 1

        model_version = ModelVersion(
            version_id=f"{model_name}-v{version_number}",
            model_name=model_name,
            model_type=model_type,
            version=version,
            version_number=version_number,
            status=ModelStatus.READY,
            is_production=False,
            training_date=datetime.now(timezone.utc),
            training_samples=training_samples,
            training_accuracy=training_accuracy,
            model_path=model_path,
            weights_path=weights_path,
            checksum=checksum,
        )

        # Store version
        if model_name not in self._model_versions:
            self._model_versions[model_name] = []
        self._model_versions[model_name].append(model_version)

        logger.info(
            "Registered new model version",
            model=model_name,
            version=version,
            accuracy=training_accuracy,
        )

        return model_version

    def deploy_model(
        self,
        model_name: str,
        version_id: str,
        deployed_by: str = "system",
    ) -> bool:
        """
        Deploy a model version to production.

        Args:
            model_name: Model name
            version_id: Version ID to deploy
            deployed_by: User deploying the model

        Returns:
            True if deployment successful
        """
        versions = self._model_versions.get(model_name, [])

        # Find the version
        target_version = None
        for v in versions:
            if v.version_id == version_id:
                target_version = v
                break

        if not target_version:
            logger.error("Model version not found", model=model_name, version=version_id)
            return False

        # Demote current production version
        for v in versions:
            if v.is_production:
                v.is_production = False
                v.status = ModelStatus.DEPRECATED
                logger.info("Demoted previous production model", version=v.version_id)

        # Promote new version
        target_version.is_production = True
        target_version.status = ModelStatus.DEPLOYED

        logger.info(
            "Deployed model to production",
            model=model_name,
            version=version_id,
            deployed_by=deployed_by,
        )

        return True

    def record_prediction(
        self,
        model_name: str,
        version_id: str,
        predicted_values: dict[str, float],
        actual_values: dict[str, float] | None = None,
        was_on_spec: bool | None = None,
        predicted_off_spec_risk: float | None = None,
    ) -> None:
        """
        Record a prediction for performance tracking.

        Args:
            model_name: Model name
            version_id: Version that made the prediction
            predicted_values: Predicted quality values
            actual_values: Actual measured values (if available)
            was_on_spec: Whether the batch was actually on-spec
            predicted_off_spec_risk: Predicted off-spec risk (0-100)
        """
        # Find version and increment counter
        versions = self._model_versions.get(model_name, [])
        for v in versions:
            if v.version_id == version_id:
                v.total_predictions += 1
                break

        # Store for later analysis
        # In production, this would go to database
        logger.debug(
            "Recorded prediction",
            model=model_name,
            version=version_id,
            predicted=predicted_values,
            actual=actual_values,
        )

    def evaluate_performance(
        self,
        model_name: str,
        version_id: str,
        predictions: list[dict[str, Any]],
        actuals: list[dict[str, Any]],
        period_days: int = 30,
    ) -> ModelPerformanceMetrics:
        """
        Evaluate model performance over a period.

        Implements the monthly rapid assessment from the document.

        Args:
            model_name: Model name
            version_id: Version to evaluate
            predictions: List of prediction records
            actuals: List of actual measurement records
            period_days: Evaluation period in days

        Returns:
            ModelPerformanceMetrics
        """
        logger.info(
            "Evaluating model performance",
            model=model_name,
            version=version_id,
            period_days=period_days,
            samples=len(predictions),
        )

        # Calculate accuracy metrics
        viscosity_errors = []
        tbn_errors = []
        flash_point_errors = []
        pour_point_errors = []

        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        verified = 0

        for pred, actual in zip(predictions, actuals):
            if actual.get("viscosity_40c"):
                verified += 1
                error = abs(pred.get("viscosity_40c", 0) - actual.get("viscosity_40c", 0))
                viscosity_errors.append(error)

            if actual.get("tbn"):
                error = abs(pred.get("tbn", 0) - actual.get("tbn", 0))
                tbn_errors.append(error)

            if actual.get("flash_point"):
                error = abs(pred.get("flash_point", 0) - actual.get("flash_point", 0))
                flash_point_errors.append(error)

            if actual.get("pour_point"):
                error = abs(pred.get("pour_point", 0) - actual.get("pour_point", 0))
                pour_point_errors.append(error)

            # Off-spec detection
            predicted_off_spec = pred.get("off_spec_risk", 0) > 5.0  # 5% threshold
            actual_off_spec = actual.get("off_spec", False)

            if predicted_off_spec and actual_off_spec:
                true_pos += 1
            elif not predicted_off_spec and not actual_off_spec:
                true_neg += 1
            elif predicted_off_spec and not actual_off_spec:
                false_pos += 1
            else:
                false_neg += 1

        # Calculate metrics
        import statistics

        def safe_mean(values):
            return statistics.mean(values) if values else 0.0

        def safe_rmse(values):
            if not values:
                return 0.0
            return (sum(v**2 for v in values) / len(values)) ** 0.5

        visc_mae = safe_mean(viscosity_errors)
        visc_rmse = safe_rmse(viscosity_errors)

        # Calculate accuracy (within tolerance)
        tolerance = 0.5  # cSt
        accurate_visc = sum(1 for e in viscosity_errors if e <= tolerance)
        accuracy = (accurate_visc / len(viscosity_errors) * 100) if viscosity_errors else 0

        # Off-spec metrics
        total_off_spec = true_pos + false_neg
        off_spec_detection = (true_pos / total_off_spec * 100) if total_off_spec > 0 else 100
        total_on_spec = true_neg + false_pos
        false_alarm = (false_pos / total_on_spec * 100) if total_on_spec > 0 else 0

        metrics = ModelPerformanceMetrics(
            model_version_id=version_id,
            evaluation_date=datetime.now(timezone.utc),
            period_days=period_days,
            total_predictions=len(predictions),
            verified_predictions=verified,
            accuracy_percent=accuracy,
            viscosity_mae=visc_mae,
            viscosity_rmse=visc_rmse,
            tbn_mae=safe_mean(tbn_errors),
            flash_point_mae=safe_mean(flash_point_errors),
            pour_point_mae=safe_mean(pour_point_errors),
            true_positives=true_pos,
            true_negatives=true_neg,
            false_positives=false_pos,
            false_negatives=false_neg,
            off_spec_detection_rate=off_spec_detection,
            false_alarm_rate=false_alarm,
            avg_confidence=85.0,  # Would calculate from predictions
            confidence_calibration_error=0.05,
        )

        # Store in history
        if model_name not in self._performance_history:
            self._performance_history[model_name] = []
        self._performance_history[model_name].append(metrics)

        logger.info(
            "Performance evaluation complete",
            model=model_name,
            accuracy=accuracy,
            off_spec_detection=off_spec_detection,
        )

        return metrics

    def check_retraining_needed(
        self,
        model_name: str,
    ) -> RetrainingRecommendation:
        """
        Check if model retraining is needed.

        Implements the degradation detection logic from the document:
        - Check accuracy against threshold
        - Check for performance degradation trend
        - Check time since last training

        Args:
            model_name: Model to check

        Returns:
            RetrainingRecommendation
        """
        versions = self._model_versions.get(model_name, [])
        history = self._performance_history.get(model_name, [])

        if not versions:
            return RetrainingRecommendation(
                model_name=model_name,
                model_type="unknown",
                current_version="none",
                recommendation="retrain_now",
                trigger=RetrainingTrigger.MANUAL_REQUEST,
                urgency="critical",
                current_accuracy=0,
                target_accuracy=self.accuracy_threshold,
                accuracy_trend="unknown",
                days_since_last_training=999,
                new_batches_since_training=0,
                recommended_action="Train initial model",
                estimated_effort_hours=80,
                notes=["No model exists for this type"],
            )

        # Get current production version
        current = next((v for v in versions if v.is_production), versions[-1])
        days_since = (datetime.now(timezone.utc) - current.training_date).days

        # Get recent performance
        recent_metrics = history[-3:] if len(history) >= 3 else history
        current_accuracy = recent_metrics[-1].accuracy_percent if recent_metrics else current.training_accuracy

        # Determine trend
        if len(recent_metrics) >= 3:
            accuracies = [m.accuracy_percent for m in recent_metrics]
            if accuracies[-1] < accuracies[0] - 2:
                trend = "degrading"
            elif accuracies[-1] > accuracies[0] + 2:
                trend = "improving"
            else:
                trend = "stable"
        else:
            trend = "unknown"

        # Check conditions
        notes = []
        urgency = "low"
        recommendation = "no_action"
        trigger = RetrainingTrigger.SCHEDULED_MONTHLY
        action = "Continue monitoring"
        effort = 0

        # Check accuracy threshold
        if current_accuracy < self.accuracy_threshold:
            urgency = "critical"
            recommendation = "retrain_now"
            trigger = RetrainingTrigger.PERFORMANCE_DEGRADATION
            action = f"Accuracy {current_accuracy:.1f}% below threshold {self.accuracy_threshold}%"
            effort = 80
            notes.append(f"Accuracy below threshold: {current_accuracy:.1f}% < {self.accuracy_threshold}%")

        # Check degradation
        elif trend == "degrading":
            urgency = "high"
            recommendation = "schedule_retrain"
            trigger = RetrainingTrigger.PERFORMANCE_DEGRADATION
            action = "Schedule quarterly retraining immediately"
            effort = 80
            notes.append("Accuracy trend is degrading")

        # Check time-based retraining
        elif days_since > 90:  # Quarterly
            urgency = "medium"
            recommendation = "schedule_retrain"
            trigger = RetrainingTrigger.SCHEDULED_QUARTERLY
            action = "Quarterly retraining due"
            effort = 80
            notes.append(f"Last training was {days_since} days ago (>90 days)")

        elif days_since > 30:  # Monthly assessment
            recommendation = "monitor"
            trigger = RetrainingTrigger.SCHEDULED_MONTHLY
            action = "Perform monthly accuracy assessment"
            effort = 40
            notes.append(f"Monthly assessment due ({days_since} days since training)")

        return RetrainingRecommendation(
            model_name=model_name,
            model_type=current.model_type,
            current_version=current.version,
            recommendation=recommendation,
            trigger=trigger,
            urgency=urgency,
            current_accuracy=current_accuracy,
            target_accuracy=self.accuracy_threshold,
            accuracy_trend=trend,
            days_since_last_training=days_since,
            new_batches_since_training=current.total_predictions,
            recommended_action=action,
            estimated_effort_hours=effort,
            recommended_date=datetime.now(timezone.utc) + timedelta(days=7) if recommendation != "no_action" else None,
            notes=notes,
        )

    def get_retraining_schedule(
        self,
        model_name: str,
    ) -> dict[str, Any]:
        """
        Get recommended retraining schedule for a model.

        Matches the schedule from Section 11 of the document:
        - Monthly: Rapid assessment (1 week effort)
        - Quarterly: Full retraining (2 weeks effort)
        - Annual: Comprehensive review (3-4 weeks effort)

        Args:
            model_name: Model name

        Returns:
            Schedule dictionary
        """
        now = datetime.now(timezone.utc)

        # Calculate next schedule dates
        # Next month start
        next_month = (now.replace(day=1) + timedelta(days=32)).replace(day=1)

        # Next quarter start
        current_quarter = (now.month - 1) // 3
        next_quarter_month = ((current_quarter + 1) % 4) * 3 + 1
        next_quarter_year = now.year if next_quarter_month > now.month else now.year + 1
        next_quarter = datetime(next_quarter_year, next_quarter_month, 1, tzinfo=timezone.utc)

        # Next year start
        next_year = datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)

        return {
            "model_name": model_name,
            "schedule": {
                "monthly_assessment": {
                    "frequency": "monthly",
                    "next_scheduled": next_month.isoformat(),
                    "effort_hours": 40,
                    "description": "Calculate model accuracy on latest 30 batches, compare predictions vs actual lab results",
                    "activities": [
                        "Calculate model accuracy on latest 30 batches",
                        "Compare predictions vs actual lab results",
                        "Flag any systematic prediction errors",
                        "Identify process changes that may require model adjustment",
                    ],
                },
                "quarterly_retraining": {
                    "frequency": "quarterly",
                    "next_scheduled": next_quarter.isoformat(),
                    "effort_hours": 80,
                    "description": "Retrain model using 12 months of accumulated data",
                    "activities": [
                        "Retrain model using 12 months of accumulated data",
                        "Validate on test set (hold back 10%)",
                        "Compare new model performance vs previous version",
                        "Deploy if accuracy improved; otherwise investigate issues",
                        "Document model version and training date",
                    ],
                },
                "annual_review": {
                    "frequency": "annually",
                    "next_scheduled": next_year.isoformat(),
                    "effort_hours": 160,
                    "description": "Comprehensive system audit and performance review",
                    "activities": [
                        "Analyze full year of production data",
                        "Identify seasonal patterns, supplier variations, product changes",
                        "Consider new features or variables to improve predictions",
                        "Assess whether model architecture still optimal",
                        "Plan for next year improvements",
                    ],
                },
            },
            "retraining_costs": {
                "monthly_assessment": "$4,000 (40 hours @ $100/hour)",
                "quarterly_retraining": "$8,000 (80 hours @ $100/hour)",
                "annual_review": "$12,000 (120 hours @ $100/hour)",
                "total_annual": "$24,000-36,000",
            },
        }

    def rollback_model(
        self,
        model_name: str,
        to_version_id: str | None = None,
    ) -> bool:
        """
        Rollback to a previous model version.

        Args:
            model_name: Model name
            to_version_id: Specific version to rollback to (or previous if None)

        Returns:
            True if rollback successful
        """
        versions = self._model_versions.get(model_name, [])

        if len(versions) < 2:
            logger.error("No previous version to rollback to", model=model_name)
            return False

        # Find target version
        if to_version_id:
            target = next((v for v in versions if v.version_id == to_version_id), None)
        else:
            # Rollback to previous production version
            current_idx = next(
                (i for i, v in enumerate(versions) if v.is_production),
                len(versions) - 1
            )
            target = versions[current_idx - 1] if current_idx > 0 else None

        if not target:
            logger.error("Rollback target not found", model=model_name)
            return False

        # Perform rollback
        return self.deploy_model(model_name, target.version_id, deployed_by="rollback")

    def get_model_history(
        self,
        model_name: str,
    ) -> list[dict[str, Any]]:
        """Get version history for a model."""
        versions = self._model_versions.get(model_name, [])

        return [
            {
                "version_id": v.version_id,
                "version": v.version,
                "version_number": v.version_number,
                "status": v.status.value,
                "is_production": v.is_production,
                "training_date": v.training_date.isoformat(),
                "training_samples": v.training_samples,
                "training_accuracy": v.training_accuracy,
                "production_accuracy": v.production_accuracy,
                "total_predictions": v.total_predictions,
            }
            for v in versions
        ]

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of model file."""
        if not os.path.exists(file_path):
            return "file_not_found"

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
