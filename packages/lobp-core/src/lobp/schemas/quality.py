"""Pydantic schemas for Quality models."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from lobp.models.quality import MeasurementSource, QualityStatus


class QualityMeasurementBase(BaseModel):
    """Base schema for quality measurements."""

    blend_id: str | None = None
    tank_tag: str | None = Field(None, max_length=50)
    sample_id: str | None = Field(None, max_length=100)
    measurement_time: datetime
    source: MeasurementSource
    analyzer_tag: str | None = Field(None, max_length=50)

    # Quality parameters
    viscosity_40c: float | None = Field(None, gt=0)
    viscosity_100c: float | None = Field(None, gt=0)
    viscosity_index: float | None = None
    flash_point: float | None = None
    pour_point: float | None = None
    density_15c: float | None = Field(None, gt=0)
    tbn: float | None = Field(None, ge=0)
    tan: float | None = Field(None, ge=0)
    water_content_ppm: float | None = Field(None, ge=0)
    sulfur_content_ppm: float | None = Field(None, ge=0)
    color: float | None = Field(None, ge=0)
    appearance: str | None = Field(None, max_length=100)


class QualityMeasurementCreate(QualityMeasurementBase):
    """Schema for creating quality measurements."""

    pass


class QualityMeasurementUpdate(BaseModel):
    """Schema for updating quality measurements."""

    status: QualityStatus | None = None
    is_final: bool | None = None
    certified: bool | None = None
    certified_by: str | None = Field(None, max_length=100)
    notes: str | None = None


class QualityMeasurementResponse(QualityMeasurementBase):
    """Response schema for quality measurements."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    status: QualityStatus
    is_final: bool
    certified: bool
    certified_by: str | None
    certification_date: datetime | None
    notes: str | None
    deviations: str | None
    created_at: datetime
    updated_at: datetime


class QualityPredictionBase(BaseModel):
    """Base schema for quality predictions."""

    blend_id: str
    model_version: str = Field(..., max_length=50)
    model_name: str = Field(..., max_length=100)

    # Predictions
    predicted_viscosity_40c: float | None = None
    predicted_viscosity_100c: float | None = None
    predicted_flash_point: float | None = None
    predicted_pour_point: float | None = None
    predicted_density: float | None = None
    predicted_tbn: float | None = None

    # Risk assessment
    off_spec_risk_percent: float = Field(..., ge=0, le=100)
    overall_confidence: float = Field(..., ge=0, le=100)


class QualityPredictionCreate(QualityPredictionBase):
    """Schema for creating quality predictions."""

    prediction_time: datetime
    risk_factors_json: str | None = None
    recommendations_json: str | None = None


class QualityPredictionResponse(QualityPredictionBase):
    """Response schema for quality predictions."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    prediction_time: datetime
    risk_factors_json: str | None
    recommendations_json: str | None
    auto_correction_applied: bool
    verified: bool
    prediction_accuracy: float | None
    created_at: datetime
    updated_at: datetime


class QualityComparisonResponse(BaseModel):
    """Schema for comparing predicted vs actual quality."""

    blend_id: str
    batch_number: str

    # Predictions
    predicted_viscosity_40c: float | None
    predicted_flash_point: float | None
    predicted_pour_point: float | None

    # Actuals
    actual_viscosity_40c: float | None
    actual_flash_point: float | None
    actual_pour_point: float | None

    # Deviations
    viscosity_deviation_percent: float | None
    flash_point_deviation: float | None
    pour_point_deviation: float | None

    # Overall
    prediction_accuracy: float | None
    status: QualityStatus
