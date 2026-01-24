"""Batch history models for AI training and analysis."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from lobp.models.base import BaseModel

if TYPE_CHECKING:
    from lobp.models.blend import Blend


class BatchHistory(BaseModel):
    """
    Comprehensive batch history for AI model training.

    Consolidates all data from a completed blend into a single record
    optimized for machine learning and historical analysis.
    Format aligned with AI Recipe Optimization document requirements.
    """

    __tablename__ = "batch_history"

    # Link to source blend
    blend_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("blends.id"), unique=True, index=True
    )

    # Batch identification (matches document format)
    batch_id: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    blend_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    recipe_code: Mapped[str] = mapped_column(String(50), index=True)
    product_code: Mapped[str | None] = mapped_column(String(50), index=True)

    # Blend volume
    blend_volume_liters: Mapped[float] = mapped_column(Float)

    # Raw material properties (primary base oil)
    base_oil_type: Mapped[str | None] = mapped_column(String(100))
    base_oil_supplier: Mapped[str | None] = mapped_column(String(100))
    base_oil_lot_number: Mapped[str | None] = mapped_column(String(100))
    base_oil_viscosity_cst: Mapped[float | None] = mapped_column(Float)
    base_oil_tbn: Mapped[float | None] = mapped_column(Float)
    base_oil_density: Mapped[float | None] = mapped_column(Float)
    base_oil_percentage: Mapped[float | None] = mapped_column(Float)

    # Additive 1 properties
    additive1_type: Mapped[str | None] = mapped_column(String(100))
    additive1_code: Mapped[str | None] = mapped_column(String(50))
    additive1_qty_wt_percent: Mapped[float | None] = mapped_column(Float)
    additive1_lot_number: Mapped[str | None] = mapped_column(String(100))

    # Additive 2 properties
    additive2_type: Mapped[str | None] = mapped_column(String(100))
    additive2_code: Mapped[str | None] = mapped_column(String(50))
    additive2_qty_wt_percent: Mapped[float | None] = mapped_column(Float)
    additive2_lot_number: Mapped[str | None] = mapped_column(String(100))

    # Additive 3 properties
    additive3_type: Mapped[str | None] = mapped_column(String(100))
    additive3_code: Mapped[str | None] = mapped_column(String(50))
    additive3_qty_wt_percent: Mapped[float | None] = mapped_column(Float)
    additive3_lot_number: Mapped[str | None] = mapped_column(String(100))

    # Additive 4 properties
    additive4_type: Mapped[str | None] = mapped_column(String(100))
    additive4_code: Mapped[str | None] = mapped_column(String(50))
    additive4_qty_wt_percent: Mapped[float | None] = mapped_column(Float)
    additive4_lot_number: Mapped[str | None] = mapped_column(String(100))

    # Extended additives (JSON for flexibility)
    additional_additives_json: Mapped[str | None] = mapped_column(Text)

    # Process parameters
    temperature_blending_c: Mapped[float | None] = mapped_column(Float)
    mixing_speed_rpm: Mapped[float | None] = mapped_column(Float)
    mixing_duration_minutes: Mapped[int | None] = mapped_column(Integer)
    ambient_temperature_c: Mapped[float | None] = mapped_column(Float)
    humidity_percent: Mapped[float | None] = mapped_column(Float)

    # Final blend quality results (from lab)
    viscosity_40c_cst: Mapped[float | None] = mapped_column(Float)
    viscosity_100c_cst: Mapped[float | None] = mapped_column(Float)
    viscosity_index: Mapped[float | None] = mapped_column(Float)
    tbn_mgkoh: Mapped[float | None] = mapped_column(Float)
    pour_point_c: Mapped[float | None] = mapped_column(Float)
    flash_point_c: Mapped[float | None] = mapped_column(Float)
    water_content_ppm: Mapped[float | None] = mapped_column(Float)
    foam_test_ml: Mapped[float | None] = mapped_column(Float)
    oxidation_stability_hours: Mapped[float | None] = mapped_column(Float)
    density_15c: Mapped[float | None] = mapped_column(Float)

    # Quality assessment
    off_spec_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    specification_met: Mapped[str] = mapped_column(String(10), default="PASS")
    quality_status: Mapped[str | None] = mapped_column(String(20))
    quality_notes: Mapped[str | None] = mapped_column(Text)

    # Target specifications (for comparison)
    target_viscosity_40c: Mapped[float | None] = mapped_column(Float)
    target_viscosity_100c: Mapped[float | None] = mapped_column(Float)
    target_tbn: Mapped[float | None] = mapped_column(Float)
    target_pour_point: Mapped[float | None] = mapped_column(Float)
    target_flash_point: Mapped[float | None] = mapped_column(Float)

    # Tolerances
    viscosity_tolerance: Mapped[float | None] = mapped_column(Float)
    tbn_tolerance: Mapped[float | None] = mapped_column(Float)
    pour_point_tolerance: Mapped[float | None] = mapped_column(Float)
    flash_point_tolerance: Mapped[float | None] = mapped_column(Float)

    # Cost data
    total_material_cost: Mapped[float | None] = mapped_column(Float)
    cost_per_liter: Mapped[float | None] = mapped_column(Float)
    energy_consumed_kwh: Mapped[float | None] = mapped_column(Float)
    energy_cost: Mapped[float | None] = mapped_column(Float)
    labor_cost: Mapped[float | None] = mapped_column(Float)
    total_batch_cost: Mapped[float | None] = mapped_column(Float)

    # AI optimization data
    ai_optimized: Mapped[bool] = mapped_column(Boolean, default=False)
    ai_model_version: Mapped[str | None] = mapped_column(String(50))
    ai_predicted_viscosity_40c: Mapped[float | None] = mapped_column(Float)
    ai_predicted_tbn: Mapped[float | None] = mapped_column(Float)
    ai_confidence: Mapped[float | None] = mapped_column(Float)
    ai_cost_savings_percent: Mapped[float | None] = mapped_column(Float)

    # Data quality flags
    data_complete: Mapped[bool] = mapped_column(Boolean, default=True)
    data_validated: Mapped[bool] = mapped_column(Boolean, default=False)
    validation_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    validated_by: Mapped[str | None] = mapped_column(String(100))
    validation_notes: Mapped[str | None] = mapped_column(Text)

    # Used for ML training
    used_for_training: Mapped[bool] = mapped_column(Boolean, default=False)
    training_run_id: Mapped[str | None] = mapped_column(String(36))
    excluded_reason: Mapped[str | None] = mapped_column(Text)

    # Relationships
    blend: Mapped["Blend"] = relationship("Blend", back_populates="batch_history")
    raw_material_data: Mapped[list["RawMaterialBatchData"]] = relationship(
        "RawMaterialBatchData",
        back_populates="batch_history",
        cascade="all, delete-orphan",
    )
    quality_results: Mapped[list["BlendQualityResult"]] = relationship(
        "BlendQualityResult",
        back_populates="batch_history",
        cascade="all, delete-orphan",
    )


class RawMaterialBatchData(BaseModel):
    """
    Detailed raw material data for each ingredient in a batch.

    Stores the actual properties of each raw material lot
    used in a blend for precise ML training.
    """

    __tablename__ = "raw_material_batch_data"

    batch_history_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("batch_history.id", ondelete="CASCADE"), index=True
    )

    # Material identification
    material_code: Mapped[str] = mapped_column(String(50))
    material_name: Mapped[str] = mapped_column(String(200))
    material_type: Mapped[str] = mapped_column(String(100))
    lot_number: Mapped[str | None] = mapped_column(String(100))
    supplier_code: Mapped[str | None] = mapped_column(String(50))
    supplier_name: Mapped[str | None] = mapped_column(String(200))

    # Quantity used
    quantity_wt_percent: Mapped[float] = mapped_column(Float)
    quantity_liters: Mapped[float | None] = mapped_column(Float)
    quantity_kg: Mapped[float | None] = mapped_column(Float)

    # Actual lot properties at time of use
    viscosity_40c_cst: Mapped[float | None] = mapped_column(Float)
    viscosity_100c_cst: Mapped[float | None] = mapped_column(Float)
    viscosity_index: Mapped[float | None] = mapped_column(Float)
    density_15c: Mapped[float | None] = mapped_column(Float)
    tbn: Mapped[float | None] = mapped_column(Float)
    tan: Mapped[float | None] = mapped_column(Float)
    pour_point_c: Mapped[float | None] = mapped_column(Float)
    flash_point_c: Mapped[float | None] = mapped_column(Float)
    water_content_ppm: Mapped[float | None] = mapped_column(Float)

    # Cost at time of use
    unit_cost: Mapped[float | None] = mapped_column(Float)
    total_cost: Mapped[float | None] = mapped_column(Float)

    # Addition sequence
    addition_order: Mapped[int] = mapped_column(Integer, default=1)
    addition_temperature_c: Mapped[float | None] = mapped_column(Float)

    # Relationships
    batch_history: Mapped["BatchHistory"] = relationship(
        "BatchHistory", back_populates="raw_material_data"
    )


class BlendQualityResult(BaseModel):
    """
    Quality test results over time for a batch.

    Stores multiple quality measurements taken during and after
    blending for tracking quality progression.
    """

    __tablename__ = "blend_quality_results"

    batch_history_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("batch_history.id", ondelete="CASCADE"), index=True
    )

    # Measurement identification
    measurement_type: Mapped[str] = mapped_column(String(50))  # inline, lab_interim, lab_final
    measurement_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    sample_id: Mapped[str | None] = mapped_column(String(100))

    # Blend progress at measurement
    blend_progress_percent: Mapped[float | None] = mapped_column(Float)
    current_volume_liters: Mapped[float | None] = mapped_column(Float)

    # Quality parameters
    viscosity_40c_cst: Mapped[float | None] = mapped_column(Float)
    viscosity_100c_cst: Mapped[float | None] = mapped_column(Float)
    viscosity_index: Mapped[float | None] = mapped_column(Float)
    tbn_mgkoh: Mapped[float | None] = mapped_column(Float)
    pour_point_c: Mapped[float | None] = mapped_column(Float)
    flash_point_c: Mapped[float | None] = mapped_column(Float)
    water_content_ppm: Mapped[float | None] = mapped_column(Float)
    foam_test_ml: Mapped[float | None] = mapped_column(Float)
    density_15c: Mapped[float | None] = mapped_column(Float)

    # Assessment at this point
    on_spec: Mapped[bool | None] = mapped_column(Boolean)
    notes: Mapped[str | None] = mapped_column(Text)

    # Relationships
    batch_history: Mapped["BatchHistory"] = relationship(
        "BatchHistory", back_populates="quality_results"
    )
