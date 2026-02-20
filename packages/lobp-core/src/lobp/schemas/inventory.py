"""Pydantic schemas for Inventory models (Material and MaterialLot)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from lobp.models.inventory import MaterialCategory


class MaterialBase(BaseModel):
    """Base schema for materials."""

    code: str = Field(..., min_length=1, max_length=50)
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    category: MaterialCategory

    # Classification
    is_active: bool = True
    is_hazardous: bool = False
    hazard_class: str | None = Field(None, max_length=50)

    # Standard properties
    standard_viscosity_40c: float | None = Field(None, gt=0)
    standard_viscosity_100c: float | None = Field(None, gt=0)
    standard_viscosity_index: float | None = None
    standard_density_15c: float | None = Field(None, gt=0)
    standard_flash_point: float | None = None
    standard_pour_point: float | None = None
    standard_tbn: float | None = Field(None, ge=0)
    standard_tan: float | None = Field(None, ge=0)

    # Property ranges
    viscosity_40c_min: float | None = None
    viscosity_40c_max: float | None = None
    viscosity_100c_min: float | None = None
    viscosity_100c_max: float | None = None
    density_min: float | None = None
    density_max: float | None = None

    # Storage
    storage_temperature_min: float | None = None
    storage_temperature_max: float | None = None
    shelf_life_days: int | None = Field(None, ge=0)

    # Cost
    standard_cost_per_liter: float | None = Field(None, ge=0)
    standard_cost_per_kg: float | None = Field(None, ge=0)

    # Unit of measure
    primary_uom: str = Field(default="liter", max_length=20)
    density_for_conversion: float | None = Field(None, gt=0)


class MaterialCreate(MaterialBase):
    """Schema for creating a material."""

    default_supplier_id: str | None = None


class MaterialUpdate(BaseModel):
    """Schema for updating a material."""

    name: str | None = Field(None, max_length=200)
    description: str | None = None
    category: MaterialCategory | None = None
    is_active: bool | None = None
    is_hazardous: bool | None = None

    standard_viscosity_40c: float | None = None
    standard_viscosity_100c: float | None = None
    standard_density_15c: float | None = None
    standard_flash_point: float | None = None
    standard_pour_point: float | None = None
    standard_tbn: float | None = None

    standard_cost_per_liter: float | None = None
    standard_cost_per_kg: float | None = None
    last_purchase_cost: float | None = None

    default_supplier_id: str | None = None


class MaterialResponse(MaterialBase):
    """Response schema for materials."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    last_purchase_cost: float | None
    last_purchase_date: datetime | None
    default_supplier_id: str | None
    created_at: datetime
    updated_at: datetime


class MaterialLotBase(BaseModel):
    """Base schema for material lots."""

    lot_number: str = Field(..., max_length=100)
    supplier_lot_number: str | None = Field(None, max_length=100)
    purchase_order: str | None = Field(None, max_length=100)

    # Receipt
    receipt_date: datetime
    received_quantity: float = Field(..., gt=0)
    current_quantity: float = Field(..., ge=0)
    reserved_quantity: float = Field(default=0.0, ge=0)
    uom: str = Field(default="liter", max_length=20)

    # Actual lot properties (from COA - Certificate of Analysis)
    lot_viscosity_40c: float | None = Field(None, gt=0)
    lot_viscosity_100c: float | None = Field(None, gt=0)
    lot_viscosity_index: float | None = None
    lot_density_15c: float | None = Field(None, gt=0)
    lot_flash_point: float | None = None
    lot_pour_point: float | None = None
    lot_tbn: float | None = Field(None, ge=0)
    lot_tan: float | None = Field(None, ge=0)
    lot_water_content_ppm: float | None = Field(None, ge=0)
    lot_sulfur_content_ppm: float | None = Field(None, ge=0)

    # Quality certification
    coa_number: str | None = Field(None, max_length=100)
    coa_date: datetime | None = None
    quality_grade: str | None = Field(None, max_length=50)
    quality_approved: bool = False

    # Cost
    unit_cost: float | None = Field(None, ge=0)
    total_cost: float | None = Field(None, ge=0)
    currency: str = Field(default="USD", max_length=3)

    # Storage
    tank_tag: str | None = Field(None, max_length=50)
    warehouse_location: str | None = Field(None, max_length=100)

    # Expiry
    manufacture_date: datetime | None = None
    expiry_date: datetime | None = None


class MaterialLotCreate(MaterialLotBase):
    """Schema for creating a material lot."""

    material_id: str
    supplier_id: str | None = None


class MaterialLotUpdate(BaseModel):
    """Schema for updating a material lot."""

    current_quantity: float | None = Field(None, ge=0)
    reserved_quantity: float | None = Field(None, ge=0)

    # Quality
    lot_viscosity_40c: float | None = None
    lot_viscosity_100c: float | None = None
    lot_density_15c: float | None = None
    lot_tbn: float | None = None
    quality_approved: bool | None = None
    approved_by: str | None = None

    # Storage
    tank_tag: str | None = None
    warehouse_location: str | None = None

    # Status
    is_available: bool | None = None
    is_quarantined: bool | None = None
    quarantine_reason: str | None = None
    is_expired: bool | None = None


class MaterialLotResponse(MaterialLotBase):
    """Response schema for material lots."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    material_id: str
    supplier_id: str | None

    # Approval
    approved_by: str | None
    approval_date: datetime | None

    # Status
    is_available: bool
    is_quarantined: bool
    quarantine_reason: str | None
    is_expired: bool

    created_at: datetime
    updated_at: datetime


class MaterialWithLots(MaterialResponse):
    """Material with its available lots."""

    available_lots: list[MaterialLotResponse] = []
    total_available_quantity: float = 0.0
    oldest_lot_date: datetime | None = None


class LotSelectionRequest(BaseModel):
    """Request for selecting lots for blending (FIFO)."""

    material_code: str
    required_quantity: float = Field(..., gt=0)
    uom: str = Field(default="liter")
    prefer_tank: str | None = None  # Prefer lots in specific tank


class LotSelectionResponse(BaseModel):
    """Response with selected lots for blending."""

    material_code: str
    required_quantity: float
    selected_lots: list["SelectedLot"]
    total_selected: float
    can_fulfill: bool
    shortage_quantity: float | None = None


class SelectedLot(BaseModel):
    """Individual lot selection."""

    lot_id: str
    lot_number: str
    quantity_to_use: float
    tank_tag: str | None
    lot_viscosity_40c: float | None
    lot_tbn: float | None
    unit_cost: float | None
    expiry_date: datetime | None
