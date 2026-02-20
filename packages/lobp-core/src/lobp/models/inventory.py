"""Material and MaterialLot database models for inventory management."""

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
    from lobp.models.supplier import Supplier


class MaterialCategory(str, enum.Enum):
    """Material classification category."""

    BASE_OIL_PARAFFINIC = "base_oil_paraffinic"
    BASE_OIL_NAPHTHENIC = "base_oil_naphthenic"
    BASE_OIL_SYNTHETIC = "base_oil_synthetic"
    ADDITIVE_PACKAGE = "additive_package"
    DETERGENT = "detergent"
    DISPERSANT = "dispersant"
    ANTIOXIDANT = "antioxidant"
    ANTI_WEAR = "anti_wear"
    VISCOSITY_MODIFIER = "viscosity_modifier"
    POUR_POINT_DEPRESSANT = "pour_point_depressant"
    FOAM_INHIBITOR = "foam_inhibitor"
    CORROSION_INHIBITOR = "corrosion_inhibitor"
    OTHER = "other"


class Material(BaseModel):
    """
    Material master data for raw materials and additives.

    Contains standard properties, costs, and supplier information
    for materials used in lubricant blending.
    """

    __tablename__ = "materials"

    # Identification
    code: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    category: Mapped[MaterialCategory] = mapped_column(Enum(MaterialCategory))

    # Classification
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_hazardous: Mapped[bool] = mapped_column(Boolean, default=False)
    hazard_class: Mapped[str | None] = mapped_column(String(50))

    # Standard properties (typical values)
    standard_viscosity_40c: Mapped[float | None] = mapped_column(Float)
    standard_viscosity_100c: Mapped[float | None] = mapped_column(Float)
    standard_viscosity_index: Mapped[float | None] = mapped_column(Float)
    standard_density_15c: Mapped[float | None] = mapped_column(Float)
    standard_flash_point: Mapped[float | None] = mapped_column(Float)
    standard_pour_point: Mapped[float | None] = mapped_column(Float)
    standard_tbn: Mapped[float | None] = mapped_column(Float)
    standard_tan: Mapped[float | None] = mapped_column(Float)

    # Property ranges (min/max acceptable)
    viscosity_40c_min: Mapped[float | None] = mapped_column(Float)
    viscosity_40c_max: Mapped[float | None] = mapped_column(Float)
    viscosity_100c_min: Mapped[float | None] = mapped_column(Float)
    viscosity_100c_max: Mapped[float | None] = mapped_column(Float)
    density_min: Mapped[float | None] = mapped_column(Float)
    density_max: Mapped[float | None] = mapped_column(Float)

    # Storage requirements
    storage_temperature_min: Mapped[float | None] = mapped_column(Float)
    storage_temperature_max: Mapped[float | None] = mapped_column(Float)
    shelf_life_days: Mapped[int | None] = mapped_column(Integer)

    # Cost tracking
    standard_cost_per_liter: Mapped[float | None] = mapped_column(Float)
    standard_cost_per_kg: Mapped[float | None] = mapped_column(Float)
    last_purchase_cost: Mapped[float | None] = mapped_column(Float)
    last_purchase_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Unit of measure
    primary_uom: Mapped[str] = mapped_column(String(20), default="liter")
    density_for_conversion: Mapped[float | None] = mapped_column(Float)

    # Default supplier
    default_supplier_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("suppliers.id")
    )

    # Relationships
    default_supplier: Mapped["Supplier | None"] = relationship("Supplier")
    lots: Mapped[list["MaterialLot"]] = relationship(
        "MaterialLot", back_populates="material"
    )


class MaterialLot(BaseModel):
    """
    Material lot/batch tracking for FIFO inventory management.

    Tracks individual lots with specific properties, supplier info,
    and quality certifications for each receipt of material.
    """

    __tablename__ = "material_lots"

    material_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("materials.id"), index=True
    )
    supplier_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("suppliers.id")
    )

    # Lot identification
    lot_number: Mapped[str] = mapped_column(String(100), index=True)
    supplier_lot_number: Mapped[str | None] = mapped_column(String(100))
    purchase_order: Mapped[str | None] = mapped_column(String(100))

    # Receipt information
    receipt_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    received_quantity: Mapped[float] = mapped_column(Float)
    current_quantity: Mapped[float] = mapped_column(Float)
    reserved_quantity: Mapped[float] = mapped_column(Float, default=0.0)
    uom: Mapped[str] = mapped_column(String(20), default="liter")

    # Actual lot properties (from COA)
    lot_viscosity_40c: Mapped[float | None] = mapped_column(Float)
    lot_viscosity_100c: Mapped[float | None] = mapped_column(Float)
    lot_viscosity_index: Mapped[float | None] = mapped_column(Float)
    lot_density_15c: Mapped[float | None] = mapped_column(Float)
    lot_flash_point: Mapped[float | None] = mapped_column(Float)
    lot_pour_point: Mapped[float | None] = mapped_column(Float)
    lot_tbn: Mapped[float | None] = mapped_column(Float)
    lot_tan: Mapped[float | None] = mapped_column(Float)
    lot_water_content_ppm: Mapped[float | None] = mapped_column(Float)
    lot_sulfur_content_ppm: Mapped[float | None] = mapped_column(Float)

    # Quality certification
    coa_number: Mapped[str | None] = mapped_column(String(100))
    coa_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    quality_grade: Mapped[str | None] = mapped_column(String(50))
    quality_approved: Mapped[bool] = mapped_column(Boolean, default=False)
    approved_by: Mapped[str | None] = mapped_column(String(100))
    approval_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Cost tracking
    unit_cost: Mapped[float | None] = mapped_column(Float)
    total_cost: Mapped[float | None] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(3), default="USD")

    # Storage location
    tank_tag: Mapped[str | None] = mapped_column(String(50))
    warehouse_location: Mapped[str | None] = mapped_column(String(100))

    # Expiry tracking
    manufacture_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expiry_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_expired: Mapped[bool] = mapped_column(Boolean, default=False)

    # Status
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)
    is_quarantined: Mapped[bool] = mapped_column(Boolean, default=False)
    quarantine_reason: Mapped[str | None] = mapped_column(Text)

    # Relationships
    material: Mapped["Material"] = relationship("Material", back_populates="lots")
    supplier: Mapped["Supplier | None"] = relationship("Supplier")
