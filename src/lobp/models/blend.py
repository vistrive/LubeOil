"""Blend and BlendIngredient database models."""

import enum
from datetime import datetime
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from lobp.models.recipe import Recipe
    from lobp.models.quality import QualityMeasurement, QualityPrediction
    from lobp.models.batch_history import BatchHistory


class BlendStatus(str, enum.Enum):
    """Blend operation status."""

    DRAFT = "draft"
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    MIXING = "mixing"
    COOLING = "cooling"
    SAMPLING = "sampling"
    LAB_ANALYSIS = "lab_analysis"
    QUALITY_HOLD = "quality_hold"
    COMPLETED = "completed"
    TRANSFERRED = "transferred"
    CANCELLED = "cancelled"
    FAILED = "failed"


class BlendPriority(str, enum.Enum):
    """Blend scheduling priority."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Blend(BaseModel):
    """
    Blend operation tracking for lubricant production.

    Records all aspects of a single batch production run,
    including materials, timing, quality, and AI optimization results.
    """

    __tablename__ = "blends"

    # Identification
    batch_number: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    recipe_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("recipes.id"), index=True
    )

    # Volume tracking
    target_volume_liters: Mapped[float] = mapped_column(Float)
    actual_volume_liters: Mapped[float] = mapped_column(Float, default=0.0)

    # Status and priority
    status: Mapped[BlendStatus] = mapped_column(
        Enum(BlendStatus), default=BlendStatus.DRAFT, index=True
    )
    priority: Mapped[BlendPriority] = mapped_column(
        Enum(BlendPriority), default=BlendPriority.NORMAL
    )

    # Tank assignments
    blend_tank_tag: Mapped[str | None] = mapped_column(String(50))
    destination_tank_tag: Mapped[str | None] = mapped_column(String(50))

    # Scheduling
    scheduled_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    scheduled_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    actual_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    actual_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Progress tracking
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    progress_percent: Mapped[float] = mapped_column(Float, default=0.0)

    # Mixing parameters (actual values)
    mixing_speed_rpm: Mapped[float | None] = mapped_column(Float)
    mixing_temperature_celsius: Mapped[float | None] = mapped_column(Float)
    mixing_duration_minutes: Mapped[int | None] = mapped_column(Integer)

    # Quality tracking
    off_spec_risk_percent: Mapped[float] = mapped_column(Float, default=0.0)
    quality_approved: Mapped[bool] = mapped_column(Boolean, default=False)
    lab_sample_taken: Mapped[bool] = mapped_column(Boolean, default=False)
    lab_sample_id: Mapped[str | None] = mapped_column(String(100))

    # AI optimization tracking
    ai_optimized: Mapped[bool] = mapped_column(Boolean, default=False)
    ai_confidence_score: Mapped[float | None] = mapped_column(Float)
    ai_cost_savings_percent: Mapped[float | None] = mapped_column(Float)
    ai_recommendations_json: Mapped[str | None] = mapped_column(Text)
    ai_model_version: Mapped[str | None] = mapped_column(String(50))

    # Energy tracking
    energy_consumed_kwh: Mapped[float] = mapped_column(Float, default=0.0)

    # Cost tracking
    material_cost: Mapped[float] = mapped_column(Float, default=0.0)
    energy_cost: Mapped[float] = mapped_column(Float, default=0.0)
    labor_cost: Mapped[float] = mapped_column(Float, default=0.0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    cost_per_liter: Mapped[float] = mapped_column(Float, default=0.0)

    # User tracking
    created_by: Mapped[str | None] = mapped_column(String(100))
    approved_by: Mapped[str | None] = mapped_column(String(100))
    notes: Mapped[str | None] = mapped_column(Text)
    hold_reason: Mapped[str | None] = mapped_column(Text)

    # Relationships
    recipe: Mapped["Recipe"] = relationship("Recipe", back_populates="blends")
    ingredients: Mapped[list["BlendIngredient"]] = relationship(
        "BlendIngredient",
        back_populates="blend",
        cascade="all, delete-orphan",
        order_by="BlendIngredient.sequence_order",
    )
    quality_measurements: Mapped[list["QualityMeasurement"]] = relationship(
        "QualityMeasurement", back_populates="blend"
    )
    quality_predictions: Mapped[list["QualityPrediction"]] = relationship(
        "QualityPrediction", back_populates="blend"
    )
    batch_history: Mapped["BatchHistory | None"] = relationship(
        "BatchHistory", back_populates="blend", uselist=False
    )


class BlendIngredient(BaseModel):
    """
    Individual ingredient addition in a blend operation.

    Tracks planned vs actual quantities, timing, and source materials
    for each component added during blending.
    """

    __tablename__ = "blend_ingredients"

    blend_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("blends.id", ondelete="CASCADE"), index=True
    )

    # Material identification
    material_code: Mapped[str] = mapped_column(String(50), index=True)
    material_name: Mapped[str] = mapped_column(String(200))
    material_lot_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("material_lots.id")
    )

    # Volume tracking
    target_volume_liters: Mapped[float] = mapped_column(Float)
    actual_volume_liters: Mapped[float] = mapped_column(Float, default=0.0)
    target_percentage: Mapped[float] = mapped_column(Float)
    actual_percentage: Mapped[float] = mapped_column(Float, default=0.0)

    # Raw material properties at time of blending
    raw_material_viscosity_cst: Mapped[float | None] = mapped_column(Float)
    raw_material_tbn: Mapped[float | None] = mapped_column(Float)
    raw_material_density: Mapped[float | None] = mapped_column(Float)

    # Sequencing
    sequence_order: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(50), default="pending")

    # Transfer tracking
    source_tank_tag: Mapped[str | None] = mapped_column(String(50))
    pump_tag: Mapped[str | None] = mapped_column(String(50))
    transfer_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    transfer_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    flow_rate_lpm: Mapped[float | None] = mapped_column(Float)

    # Cost tracking
    unit_cost: Mapped[float | None] = mapped_column(Float)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)

    # AI tracking
    ai_adjusted: Mapped[bool] = mapped_column(Boolean, default=False)
    ai_original_percentage: Mapped[float | None] = mapped_column(Float)
    deviation_percent: Mapped[float] = mapped_column(Float, default=0.0)

    # Relationships
    blend: Mapped["Blend"] = relationship("Blend", back_populates="ingredients")
    material_lot: Mapped["MaterialLot | None"] = relationship("MaterialLot")
