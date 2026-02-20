"""AI model versioning and retraining database models."""

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from lobp.models.base import BaseModel


class ModelStatus(str, enum.Enum):
    """AI model deployment status."""

    TRAINING = "training"
    VALIDATING = "validating"
    READY = "ready"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class RetrainingFrequency(str, enum.Enum):
    """Retraining schedule frequency."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ON_DEMAND = "on_demand"


class AIModelVersion(BaseModel):
    """
    AI model version tracking and management.

    Tracks all versions of AI models used for recipe optimization
    and quality prediction, including performance metrics.
    """

    __tablename__ = "ai_model_versions"

    # Model identification
    model_name: Mapped[str] = mapped_column(String(100), index=True)
    model_type: Mapped[str] = mapped_column(String(50))  # quality_predictor, recipe_optimizer
    version: Mapped[str] = mapped_column(String(50))
    version_number: Mapped[int] = mapped_column(Integer)

    # Status
    status: Mapped[ModelStatus] = mapped_column(
        Enum(ModelStatus), default=ModelStatus.TRAINING
    )
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)
    is_production: Mapped[bool] = mapped_column(Boolean, default=False)

    # Training information
    training_started: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    training_completed: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    training_duration_minutes: Mapped[float | None] = mapped_column(Float)

    # Training data
    training_samples: Mapped[int] = mapped_column(Integer, default=0)
    validation_samples: Mapped[int] = mapped_column(Integer, default=0)
    test_samples: Mapped[int] = mapped_column(Integer, default=0)
    training_data_start_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    training_data_end_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Model parameters (JSON)
    hyperparameters_json: Mapped[str | None] = mapped_column(Text)
    architecture_json: Mapped[str | None] = mapped_column(Text)
    feature_columns_json: Mapped[str | None] = mapped_column(Text)

    # Performance metrics
    training_loss: Mapped[float | None] = mapped_column(Float)
    validation_loss: Mapped[float | None] = mapped_column(Float)
    test_loss: Mapped[float | None] = mapped_column(Float)

    # Quality prediction metrics
    viscosity_mae: Mapped[float | None] = mapped_column(Float)
    viscosity_rmse: Mapped[float | None] = mapped_column(Float)
    viscosity_r2: Mapped[float | None] = mapped_column(Float)
    tbn_mae: Mapped[float | None] = mapped_column(Float)
    flash_point_mae: Mapped[float | None] = mapped_column(Float)
    pour_point_mae: Mapped[float | None] = mapped_column(Float)

    # Overall accuracy
    overall_accuracy: Mapped[float | None] = mapped_column(Float)
    off_spec_detection_rate: Mapped[float | None] = mapped_column(Float)
    false_positive_rate: Mapped[float | None] = mapped_column(Float)
    false_negative_rate: Mapped[float | None] = mapped_column(Float)

    # Production performance (updated over time)
    production_predictions: Mapped[int] = mapped_column(Integer, default=0)
    production_accuracy: Mapped[float | None] = mapped_column(Float)
    avg_confidence_score: Mapped[float | None] = mapped_column(Float)

    # File storage
    model_file_path: Mapped[str | None] = mapped_column(String(500))
    weights_file_path: Mapped[str | None] = mapped_column(String(500))
    scaler_file_path: Mapped[str | None] = mapped_column(String(500))
    model_checksum: Mapped[str | None] = mapped_column(String(64))

    # Deployment tracking
    deployed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deployed_by: Mapped[str | None] = mapped_column(String(100))
    deprecation_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deprecation_reason: Mapped[str | None] = mapped_column(Text)

    # Notes
    description: Mapped[str | None] = mapped_column(Text)
    release_notes: Mapped[str | None] = mapped_column(Text)

    # Relationships
    training_runs: Mapped[list["ModelTrainingRun"]] = relationship(
        "ModelTrainingRun", back_populates="model_version"
    )
    prediction_logs: Mapped[list["ModelPredictionLog"]] = relationship(
        "ModelPredictionLog", back_populates="model_version"
    )


class ModelTrainingRun(BaseModel):
    """
    Individual training run record.

    Tracks each training execution including data used,
    parameters, and results for reproducibility.
    """

    __tablename__ = "model_training_runs"

    model_version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("ai_model_versions.id"), index=True
    )

    # Run identification
    run_id: Mapped[str] = mapped_column(String(100), unique=True)
    run_number: Mapped[int] = mapped_column(Integer)

    # Timing
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[float | None] = mapped_column(Float)

    # Status
    status: Mapped[str] = mapped_column(String(50))  # running, completed, failed
    error_message: Mapped[str | None] = mapped_column(Text)

    # Training configuration
    batch_size: Mapped[int | None] = mapped_column(Integer)
    epochs: Mapped[int | None] = mapped_column(Integer)
    learning_rate: Mapped[float | None] = mapped_column(Float)
    early_stopping_patience: Mapped[int | None] = mapped_column(Integer)

    # Data statistics
    total_samples: Mapped[int] = mapped_column(Integer, default=0)
    training_samples: Mapped[int] = mapped_column(Integer, default=0)
    validation_samples: Mapped[int] = mapped_column(Integer, default=0)
    excluded_samples: Mapped[int] = mapped_column(Integer, default=0)
    exclusion_reasons_json: Mapped[str | None] = mapped_column(Text)

    # Results
    final_training_loss: Mapped[float | None] = mapped_column(Float)
    final_validation_loss: Mapped[float | None] = mapped_column(Float)
    best_epoch: Mapped[int | None] = mapped_column(Integer)
    metrics_json: Mapped[str | None] = mapped_column(Text)

    # Environment
    compute_environment: Mapped[str | None] = mapped_column(String(100))
    gpu_used: Mapped[bool] = mapped_column(Boolean, default=False)
    memory_used_gb: Mapped[float | None] = mapped_column(Float)

    # Trigger
    triggered_by: Mapped[str | None] = mapped_column(String(100))  # scheduler, manual, performance_degradation
    trigger_reason: Mapped[str | None] = mapped_column(Text)

    # Relationships
    model_version: Mapped["AIModelVersion"] = relationship(
        "AIModelVersion", back_populates="training_runs"
    )


class ModelPredictionLog(BaseModel):
    """
    Log of model predictions for accuracy tracking.

    Records predictions made by AI models along with
    actual results for continuous accuracy monitoring.
    """

    __tablename__ = "model_prediction_logs"

    model_version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("ai_model_versions.id"), index=True
    )
    blend_id: Mapped[str | None] = mapped_column(String(36), index=True)

    # Prediction timing
    prediction_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    # Input features (JSON)
    input_features_json: Mapped[str | None] = mapped_column(Text)

    # Predictions
    predicted_viscosity_40c: Mapped[float | None] = mapped_column(Float)
    predicted_viscosity_100c: Mapped[float | None] = mapped_column(Float)
    predicted_tbn: Mapped[float | None] = mapped_column(Float)
    predicted_flash_point: Mapped[float | None] = mapped_column(Float)
    predicted_pour_point: Mapped[float | None] = mapped_column(Float)
    predicted_off_spec_risk: Mapped[float | None] = mapped_column(Float)
    confidence_score: Mapped[float | None] = mapped_column(Float)

    # Actual results (filled in after lab analysis)
    actual_viscosity_40c: Mapped[float | None] = mapped_column(Float)
    actual_viscosity_100c: Mapped[float | None] = mapped_column(Float)
    actual_tbn: Mapped[float | None] = mapped_column(Float)
    actual_flash_point: Mapped[float | None] = mapped_column(Float)
    actual_pour_point: Mapped[float | None] = mapped_column(Float)
    actual_on_spec: Mapped[bool | None] = mapped_column(Boolean)

    # Accuracy metrics (computed after actual results available)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verification_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    viscosity_40c_error: Mapped[float | None] = mapped_column(Float)
    viscosity_40c_error_percent: Mapped[float | None] = mapped_column(Float)
    tbn_error: Mapped[float | None] = mapped_column(Float)
    off_spec_prediction_correct: Mapped[bool | None] = mapped_column(Boolean)

    # Relationships
    model_version: Mapped["AIModelVersion"] = relationship(
        "AIModelVersion", back_populates="prediction_logs"
    )


class RetrainingSchedule(BaseModel):
    """
    Scheduled retraining configuration.

    Defines when and how models should be retrained,
    following the document's monthly/quarterly requirements.
    """

    __tablename__ = "retraining_schedules"

    # Model identification
    model_name: Mapped[str] = mapped_column(String(100), index=True)
    model_type: Mapped[str] = mapped_column(String(50))

    # Schedule configuration
    frequency: Mapped[RetrainingFrequency] = mapped_column(
        Enum(RetrainingFrequency), default=RetrainingFrequency.MONTHLY
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Timing
    next_scheduled_run: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_run_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_run_status: Mapped[str | None] = mapped_column(String(50))
    last_run_model_version_id: Mapped[str | None] = mapped_column(String(36))

    # Data requirements
    min_training_samples: Mapped[int] = mapped_column(Integer, default=500)
    max_training_age_days: Mapped[int] = mapped_column(Integer, default=365)
    include_new_data_only: Mapped[bool] = mapped_column(Boolean, default=False)

    # Performance triggers
    accuracy_threshold: Mapped[float | None] = mapped_column(Float)  # Trigger if below
    degradation_trigger: Mapped[bool] = mapped_column(Boolean, default=True)
    degradation_threshold_percent: Mapped[float] = mapped_column(Float, default=5.0)

    # Training parameters (JSON)
    training_config_json: Mapped[str | None] = mapped_column(Text)

    # Notification
    notify_on_completion: Mapped[bool] = mapped_column(Boolean, default=True)
    notification_emails: Mapped[str | None] = mapped_column(Text)

    # Notes
    description: Mapped[str | None] = mapped_column(Text)
