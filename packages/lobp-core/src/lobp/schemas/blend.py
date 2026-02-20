"""Pydantic schemas for Blend models."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from lobp.models.blend import BlendPriority, BlendStatus


class BlendIngredientBase(BaseModel):
    """Base schema for blend ingredients."""

    material_code: str = Field(..., min_length=1, max_length=50)
    material_name: str = Field(..., min_length=1, max_length=200)
    target_volume_liters: float = Field(..., gt=0)
    target_percentage: float = Field(..., ge=0, le=100)
    source_tank_tag: str | None = Field(None, max_length=50)
    sequence_order: int = Field(default=1, ge=1)


class BlendIngredientCreate(BlendIngredientBase):
    """Schema for creating blend ingredients."""

    pass


class BlendIngredientResponse(BlendIngredientBase):
    """Response schema for blend ingredients."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    blend_id: str
    actual_volume_liters: float
    actual_percentage: float
    status: str
    transfer_start: datetime | None
    transfer_end: datetime | None
    flow_rate_lpm: float | None
    pump_tag: str | None
    ai_adjusted: bool
    deviation_percent: float
    created_at: datetime
    updated_at: datetime


class BlendBase(BaseModel):
    """Base schema for blends."""

    recipe_id: str
    target_volume_liters: float = Field(..., gt=0)
    priority: BlendPriority = BlendPriority.NORMAL
    blend_tank_tag: str | None = Field(None, max_length=50)
    destination_tank_tag: str | None = Field(None, max_length=50)
    scheduled_start: datetime | None = None
    scheduled_end: datetime | None = None
    notes: str | None = None


class BlendCreate(BlendBase):
    """Schema for creating a blend."""

    ingredients: list[BlendIngredientCreate] = Field(default_factory=list)


class BlendUpdate(BaseModel):
    """Schema for updating a blend."""

    priority: BlendPriority | None = None
    blend_tank_tag: str | None = Field(None, max_length=50)
    destination_tank_tag: str | None = Field(None, max_length=50)
    scheduled_start: datetime | None = None
    scheduled_end: datetime | None = None
    notes: str | None = None
    hold_reason: str | None = None


class BlendProgressUpdate(BaseModel):
    """Schema for updating blend progress (from DCS)."""

    current_step: int = Field(..., ge=0)
    progress_percent: float = Field(..., ge=0, le=100)
    actual_volume_liters: float = Field(..., ge=0)
    mixing_speed_rpm: float | None = None
    mixing_temperature_celsius: float | None = None
    energy_consumed_kwh: float | None = Field(None, ge=0)


class BlendStatusUpdate(BaseModel):
    """Schema for status updates."""

    status: BlendStatus
    notes: str | None = None


class BlendResponse(BlendBase):
    """Response schema for blends."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    batch_number: str
    status: BlendStatus
    actual_volume_liters: float
    current_step: int
    total_steps: int
    progress_percent: float

    # Timing
    actual_start: datetime | None
    actual_end: datetime | None

    # Mixing
    mixing_speed_rpm: float | None
    mixing_temperature_celsius: float | None
    mixing_duration_minutes: int | None

    # Quality
    off_spec_risk_percent: float
    quality_approved: bool
    lab_sample_taken: bool

    # AI
    ai_optimized: bool
    ai_confidence_score: float | None

    # Cost/Energy
    energy_consumed_kwh: float
    material_cost: float

    # Metadata
    created_by: str | None
    approved_by: str | None
    hold_reason: str | None

    # Related
    ingredients: list[BlendIngredientResponse] = Field(default_factory=list)

    created_at: datetime
    updated_at: datetime


class BlendQueueItem(BaseModel):
    """Schema for blend queue display."""

    id: str
    batch_number: str
    recipe_code: str
    recipe_name: str
    target_volume_liters: float
    status: BlendStatus
    priority: BlendPriority
    scheduled_start: datetime | None
    progress_percent: float
    blend_tank_tag: str | None
