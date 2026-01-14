"""Pydantic schemas for Tank models."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from lobp.models.tank import TankStatus, TankType


class TankBase(BaseModel):
    """Base schema for tanks."""

    tag: str = Field(..., min_length=1, max_length=50)
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    location: str | None = Field(None, max_length=100)
    tank_type: TankType

    # Physical dimensions
    capacity_liters: float = Field(..., gt=0)
    diameter_meters: float | None = Field(None, gt=0)
    height_meters: float | None = Field(None, gt=0)

    # Level thresholds
    high_high_level: float = Field(default=95.0, ge=0, le=100)
    high_level: float = Field(default=90.0, ge=0, le=100)
    low_level: float = Field(default=20.0, ge=0, le=100)
    low_low_level: float = Field(default=10.0, ge=0, le=100)

    # Temperature limits
    max_temperature: float | None = None
    min_temperature: float | None = None

    # Equipment flags
    has_heating: bool = False
    has_agitator: bool = False
    has_level_indicator: bool = True
    has_temperature_indicator: bool = True

    # Segregation
    segregation_group: str | None = Field(None, max_length=50)


class TankCreate(TankBase):
    """Schema for creating a tank."""

    pass


class TankUpdate(BaseModel):
    """Schema for updating a tank."""

    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = None
    location: str | None = Field(None, max_length=100)
    status: TankStatus | None = None

    high_high_level: float | None = Field(None, ge=0, le=100)
    high_level: float | None = Field(None, ge=0, le=100)
    low_level: float | None = Field(None, ge=0, le=100)
    low_low_level: float | None = Field(None, ge=0, le=100)

    max_temperature: float | None = None
    min_temperature: float | None = None

    has_heating: bool | None = None
    has_agitator: bool | None = None

    segregation_group: str | None = Field(None, max_length=50)

    # Inventory management
    min_stock_level: float | None = Field(None, ge=0)
    reorder_point: float | None = Field(None, ge=0)


class TankLevelUpdate(BaseModel):
    """Schema for updating tank level (from DCS)."""

    current_level_liters: float = Field(..., ge=0)
    current_level_percent: float = Field(..., ge=0, le=100)
    current_temperature_celsius: float | None = None
    current_pressure_bar: float | None = None


class TankContentsUpdate(BaseModel):
    """Schema for updating tank contents."""

    material_code: str | None = Field(None, max_length=50)
    material_name: str | None = Field(None, max_length=200)
    batch_number: str | None = Field(None, max_length=100)
    current_viscosity: float | None = None
    current_density: float | None = None


class TankResponse(TankBase):
    """Response schema for tanks."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    status: TankStatus

    # Current state
    current_level_liters: float
    current_level_percent: float
    current_temperature_celsius: float | None
    current_pressure_bar: float | None

    # Current contents
    material_code: str | None
    material_name: str | None
    batch_number: str | None
    current_viscosity: float | None
    current_density: float | None

    # Inventory
    min_stock_level: float | None
    reorder_point: float | None
    heel_volume_liters: float

    # Computed properties
    available_capacity: float
    usable_volume: float
    is_high_level: bool
    is_low_level: bool

    created_at: datetime
    updated_at: datetime
