"""Quality measurement and prediction database models."""

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from lobp.models.base import BaseModel

if TYPE_CHECKING:
    from lobp.models.blend import Blend


class MeasurementSource(str, enum.Enum):
    """Source of quality measurement."""

    INLINE_ANALYZER = "inline_analyzer"
    LAB_ANALYSIS = "lab_analysis"
    SOFT_SENSOR = "soft_sensor"
    MANUAL_ENTRY = "manual_entry"


class QualityStatus(str, enum.Enum):
    """Quality assessment result."""

    ON_SPEC = "on_spec"
    MARGINAL = "marginal"
    OFF_SPEC = "off_spec"
    PENDING = "pending"


class QualityMeasurement(BaseModel):
    """
    Quality measurement record from lab or inline analyzers.

    Stores actual measured quality parameters for a blend
    from various sources (lab, inline analyzer, soft sensor).
    """

    __tablename__ = "quality_measurements"

    blend_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("blends.id"), index=True
    )
    tank_tag: Mapped[str | None] = mapped_column(String(50))
    sample_id: Mapped[str | None] = mapped_column(String(100), index=True)

    # Measurement metadata
    measurement_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source: Mapped[MeasurementSource] = mapped_column(Enum(MeasurementSource))
    analyzer_tag: Mapped[str | None] = mapped_column(String(50))

    # Quality parameters
    viscosity_40c: Mapped[float | None] = mapped_column(Float)
    viscosity_100c: Mapped[float | None] = mapped_column(Float)
    viscosity_index: Mapped[float | None] = mapped_column(Float)
    flash_point: Mapped[float | None] = mapped_column(Float)
    pour_point: Mapped[float | None] = mapped_column(Float)
    density_15c: Mapped[float | None] = mapped_column(Float)
    tbn: Mapped[float | None] = mapped_column(Float)
    tan: Mapped[float | None] = mapped_column(Float)
    water_content_ppm: Mapped[float | None] = mapped_column(Float)
    sulfur_content_ppm: Mapped[float | None] = mapped_column(Float)
    foam_test_ml: Mapped[float | None] = mapped_column(Float)
    oxidation_stability_hours: Mapped[float | None] = mapped_column(Float)
    color: Mapped[float | None] = mapped_column(Float)
    appearance: Mapped[str | None] = mapped_column(String(100))

    # Assessment
    status: Mapped[QualityStatus] = mapped_column(
        Enum(QualityStatus), default=QualityStatus.PENDING
    )
    is_final: Mapped[bool] = mapped_column(Boolean, default=False)
    certified: Mapped[bool] = mapped_column(Boolean, default=False)
    certified_by: Mapped[str | None] = mapped_column(String(100))
    certification_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Deviations and notes
    deviations: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)

    # Relationships
    blend: Mapped["Blend | None"] = relationship(
        "Blend", back_populates="quality_measurements"
    )


class QualityPrediction(BaseModel):
    """
    AI-generated quality predictions for a blend.

    Stores predicted quality parameters before blending completes,
    along with confidence scores and risk assessments.
    """

    __tablename__ = "quality_predictions"

    blend_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("blends.id"), index=True
    )

    # Model information
    model_version: Mapped[str] = mapped_column(String(50))
    model_name: Mapped[str] = mapped_column(String(100))
    prediction_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    # Predicted values
    predicted_viscosity_40c: Mapped[float | None] = mapped_column(Float)
    predicted_viscosity_100c: Mapped[float | None] = mapped_column(Float)
    predicted_viscosity_index: Mapped[float | None] = mapped_column(Float)
    predicted_flash_point: Mapped[float | None] = mapped_column(Float)
    predicted_pour_point: Mapped[float | None] = mapped_column(Float)
    predicted_density: Mapped[float | None] = mapped_column(Float)
    predicted_tbn: Mapped[float | None] = mapped_column(Float)

    # Confidence intervals (95%)
    viscosity_40c_lower: Mapped[float | None] = mapped_column(Float)
    viscosity_40c_upper: Mapped[float | None] = mapped_column(Float)
    viscosity_100c_lower: Mapped[float | None] = mapped_column(Float)
    viscosity_100c_upper: Mapped[float | None] = mapped_column(Float)
    flash_point_lower: Mapped[float | None] = mapped_column(Float)
    flash_point_upper: Mapped[float | None] = mapped_column(Float)
    pour_point_lower: Mapped[float | None] = mapped_column(Float)
    pour_point_upper: Mapped[float | None] = mapped_column(Float)

    # Risk assessment
    off_spec_risk_percent: Mapped[float] = mapped_column(Float)
    overall_confidence: Mapped[float] = mapped_column(Float)
    historical_batches_used: Mapped[int] = mapped_column(Float, default=0)

    # Risk factors and recommendations (JSON)
    risk_factors_json: Mapped[str | None] = mapped_column(Text)
    recommendations_json: Mapped[str | None] = mapped_column(Text)

    # Verification
    auto_correction_applied: Mapped[bool] = mapped_column(Boolean, default=False)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verification_measurement_id: Mapped[str | None] = mapped_column(String(36))
    prediction_accuracy: Mapped[float | None] = mapped_column(Float)

    # Relationships
    blend: Mapped["Blend"] = relationship(
        "Blend", back_populates="quality_predictions"
    )
