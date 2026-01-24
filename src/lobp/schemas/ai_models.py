"""Pydantic schemas for AI Model versioning and management."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from lobp.models.ai_models import ModelStatus, RetrainingFrequency


class AIModelVersionBase(BaseModel):
    """Base schema for AI model versions."""

    model_name: str = Field(..., max_length=100)
    model_type: str = Field(..., max_length=50)  # quality_predictor, recipe_optimizer
    version: str = Field(..., max_length=50)
    description: str | None = None


class AIModelVersionCreate(AIModelVersionBase):
    """Schema for creating a new model version."""

    hyperparameters_json: str | None = None
    architecture_json: str | None = None
    feature_columns_json: str | None = None


class AIModelVersionUpdate(BaseModel):
    """Schema for updating model version."""

    status: ModelStatus | None = None
    is_current: bool | None = None
    is_production: bool | None = None

    # Performance metrics
    training_loss: float | None = None
    validation_loss: float | None = None
    test_loss: float | None = None
    overall_accuracy: float | None = None
    viscosity_mae: float | None = None
    viscosity_rmse: float | None = None
    tbn_mae: float | None = None

    # Deployment
    deployed_by: str | None = None
    deprecation_reason: str | None = None
    release_notes: str | None = None


class AIModelVersionResponse(AIModelVersionBase):
    """Response schema for AI model versions."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    version_number: int
    status: ModelStatus
    is_current: bool
    is_production: bool

    # Training info
    training_started: datetime | None
    training_completed: datetime | None
    training_duration_minutes: float | None
    training_samples: int
    validation_samples: int
    test_samples: int
    training_data_start_date: datetime | None
    training_data_end_date: datetime | None

    # Performance
    training_loss: float | None
    validation_loss: float | None
    test_loss: float | None
    overall_accuracy: float | None
    viscosity_mae: float | None
    viscosity_rmse: float | None
    viscosity_r2: float | None
    tbn_mae: float | None
    flash_point_mae: float | None
    pour_point_mae: float | None
    off_spec_detection_rate: float | None
    false_positive_rate: float | None
    false_negative_rate: float | None

    # Production stats
    production_predictions: int
    production_accuracy: float | None
    avg_confidence_score: float | None

    # Files
    model_file_path: str | None
    weights_file_path: str | None
    model_checksum: str | None

    # Deployment
    deployed_at: datetime | None
    deployed_by: str | None
    deprecation_date: datetime | None
    deprecation_reason: str | None
    release_notes: str | None

    created_at: datetime
    updated_at: datetime


class ModelTrainingRunBase(BaseModel):
    """Base schema for training runs."""

    run_id: str = Field(..., max_length=100)
    run_number: int = Field(..., ge=1)

    # Configuration
    batch_size: int | None = Field(None, ge=1)
    epochs: int | None = Field(None, ge=1)
    learning_rate: float | None = Field(None, gt=0)
    early_stopping_patience: int | None = Field(None, ge=1)


class ModelTrainingRunCreate(ModelTrainingRunBase):
    """Schema for creating a training run."""

    model_version_id: str
    triggered_by: str | None = None
    trigger_reason: str | None = None


class ModelTrainingRunUpdate(BaseModel):
    """Schema for updating training run."""

    status: str | None = None
    error_message: str | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None

    # Results
    final_training_loss: float | None = None
    final_validation_loss: float | None = None
    best_epoch: int | None = None
    metrics_json: str | None = None


class ModelTrainingRunResponse(ModelTrainingRunBase):
    """Response schema for training runs."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    model_version_id: str

    # Timing
    started_at: datetime
    completed_at: datetime | None
    duration_seconds: float | None

    # Status
    status: str
    error_message: str | None

    # Data
    total_samples: int
    training_samples: int
    validation_samples: int
    excluded_samples: int

    # Results
    final_training_loss: float | None
    final_validation_loss: float | None
    best_epoch: int | None

    # Trigger
    triggered_by: str | None
    trigger_reason: str | None

    created_at: datetime


class ModelPredictionLogBase(BaseModel):
    """Base schema for prediction logs."""

    blend_id: str | None = None
    prediction_time: datetime

    # Predictions
    predicted_viscosity_40c: float | None = None
    predicted_viscosity_100c: float | None = None
    predicted_tbn: float | None = None
    predicted_flash_point: float | None = None
    predicted_pour_point: float | None = None
    predicted_off_spec_risk: float | None = Field(None, ge=0, le=100)
    confidence_score: float | None = Field(None, ge=0, le=100)


class ModelPredictionLogCreate(ModelPredictionLogBase):
    """Schema for creating prediction log."""

    model_version_id: str
    input_features_json: str | None = None


class ModelPredictionLogVerify(BaseModel):
    """Schema for verifying predictions with actual results."""

    actual_viscosity_40c: float | None = None
    actual_viscosity_100c: float | None = None
    actual_tbn: float | None = None
    actual_flash_point: float | None = None
    actual_pour_point: float | None = None
    actual_on_spec: bool | None = None


class ModelPredictionLogResponse(ModelPredictionLogBase):
    """Response schema for prediction logs."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    model_version_id: str

    # Actuals
    actual_viscosity_40c: float | None
    actual_viscosity_100c: float | None
    actual_tbn: float | None
    actual_flash_point: float | None
    actual_pour_point: float | None
    actual_on_spec: bool | None

    # Verification
    verified: bool
    verification_time: datetime | None
    viscosity_40c_error: float | None
    viscosity_40c_error_percent: float | None
    tbn_error: float | None
    off_spec_prediction_correct: bool | None

    created_at: datetime


class RetrainingScheduleBase(BaseModel):
    """Base schema for retraining schedules."""

    model_name: str = Field(..., max_length=100)
    model_type: str = Field(..., max_length=50)
    frequency: RetrainingFrequency = RetrainingFrequency.MONTHLY
    is_active: bool = True

    # Data requirements
    min_training_samples: int = Field(default=500, ge=100)
    max_training_age_days: int = Field(default=365, ge=30)
    include_new_data_only: bool = False

    # Performance triggers
    accuracy_threshold: float | None = Field(None, ge=0, le=100)
    degradation_trigger: bool = True
    degradation_threshold_percent: float = Field(default=5.0, ge=0, le=50)

    # Notification
    notify_on_completion: bool = True
    notification_emails: str | None = None

    description: str | None = None


class RetrainingScheduleCreate(RetrainingScheduleBase):
    """Schema for creating retraining schedule."""

    training_config_json: str | None = None
    next_scheduled_run: datetime | None = None


class RetrainingScheduleUpdate(BaseModel):
    """Schema for updating retraining schedule."""

    frequency: RetrainingFrequency | None = None
    is_active: bool | None = None
    next_scheduled_run: datetime | None = None
    min_training_samples: int | None = None
    max_training_age_days: int | None = None
    accuracy_threshold: float | None = None
    degradation_trigger: bool | None = None
    notification_emails: str | None = None


class RetrainingScheduleResponse(RetrainingScheduleBase):
    """Response schema for retraining schedules."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    next_scheduled_run: datetime | None
    last_run_time: datetime | None
    last_run_status: str | None
    last_run_model_version_id: str | None
    training_config_json: str | None
    created_at: datetime
    updated_at: datetime


class ModelPerformanceSummary(BaseModel):
    """Summary of model performance metrics."""

    model_name: str
    model_type: str
    current_version: str
    is_production: bool

    # Accuracy
    overall_accuracy: float | None
    viscosity_mae: float | None
    tbn_mae: float | None
    off_spec_detection_rate: float | None

    # Production stats
    total_predictions: int
    verified_predictions: int
    avg_confidence: float | None
    production_accuracy: float | None

    # Trend (compared to previous period)
    accuracy_trend: str | None  # improving, stable, degrading
    accuracy_change_percent: float | None

    # Recommendations
    needs_retraining: bool
    retraining_reason: str | None
    next_scheduled_retraining: datetime | None
