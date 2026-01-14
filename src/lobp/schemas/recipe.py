"""Pydantic schemas for Recipe models."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from lobp.models.recipe import IngredientType, RecipeStatus


class RecipeIngredientBase(BaseModel):
    """Base schema for recipe ingredients."""

    material_code: str = Field(..., min_length=1, max_length=50)
    material_name: str = Field(..., min_length=1, max_length=200)
    ingredient_type: IngredientType
    target_percentage: float = Field(..., ge=0, le=100)
    min_percentage: float | None = Field(None, ge=0, le=100)
    max_percentage: float | None = Field(None, ge=0, le=100)
    addition_order: int = Field(default=1, ge=1)
    requires_heating: bool = False
    heating_temperature: float | None = None
    cost_per_liter: float | None = Field(None, ge=0)
    ai_adjustable: bool = True


class RecipeIngredientCreate(RecipeIngredientBase):
    """Schema for creating a recipe ingredient."""

    pass


class RecipeIngredientUpdate(BaseModel):
    """Schema for updating a recipe ingredient."""

    target_percentage: float | None = Field(None, ge=0, le=100)
    min_percentage: float | None = Field(None, ge=0, le=100)
    max_percentage: float | None = Field(None, ge=0, le=100)
    addition_order: int | None = Field(None, ge=1)
    requires_heating: bool | None = None
    heating_temperature: float | None = None
    cost_per_liter: float | None = Field(None, ge=0)
    ai_adjustable: bool | None = None


class RecipeIngredientResponse(RecipeIngredientBase):
    """Response schema for recipe ingredients."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    recipe_id: str
    created_at: datetime
    updated_at: datetime


class RecipeBase(BaseModel):
    """Base schema for recipes."""

    code: str = Field(..., min_length=1, max_length=50)
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None

    # Target quality specifications
    target_viscosity_40c: float | None = Field(None, gt=0)
    target_viscosity_100c: float | None = Field(None, gt=0)
    target_viscosity_index: float | None = None
    target_flash_point: float | None = None
    target_pour_point: float | None = None
    target_density: float | None = Field(None, gt=0)
    target_tbn: float | None = Field(None, ge=0)
    target_tan: float | None = Field(None, ge=0)

    # Tolerance ranges
    viscosity_tolerance: float = Field(default=2.0, ge=0, le=10)
    flash_point_tolerance: float = Field(default=5.0, ge=0, le=20)
    pour_point_tolerance: float = Field(default=3.0, ge=0, le=10)

    # Batch size constraints
    min_batch_size_liters: float | None = Field(None, gt=0)
    max_batch_size_liters: float | None = Field(None, gt=0)
    standard_batch_size_liters: float | None = Field(None, gt=0)

    # Production parameters
    mixing_time_minutes: int | None = Field(None, ge=0)
    mixing_temperature_celsius: float | None = None
    cooling_required: bool = False

    # AI optimization
    ai_optimization_enabled: bool = True


class RecipeCreate(RecipeBase):
    """Schema for creating a recipe."""

    ingredients: list[RecipeIngredientCreate] = Field(default_factory=list)


class RecipeUpdate(BaseModel):
    """Schema for updating a recipe."""

    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = None
    status: RecipeStatus | None = None

    target_viscosity_40c: float | None = Field(None, gt=0)
    target_viscosity_100c: float | None = Field(None, gt=0)
    target_viscosity_index: float | None = None
    target_flash_point: float | None = None
    target_pour_point: float | None = None
    target_density: float | None = Field(None, gt=0)
    target_tbn: float | None = Field(None, ge=0)
    target_tan: float | None = Field(None, ge=0)

    viscosity_tolerance: float | None = Field(None, ge=0, le=10)
    flash_point_tolerance: float | None = Field(None, ge=0, le=20)
    pour_point_tolerance: float | None = Field(None, ge=0, le=10)

    min_batch_size_liters: float | None = Field(None, gt=0)
    max_batch_size_liters: float | None = Field(None, gt=0)
    standard_batch_size_liters: float | None = Field(None, gt=0)

    mixing_time_minutes: int | None = Field(None, ge=0)
    mixing_temperature_celsius: float | None = None
    cooling_required: bool | None = None

    ai_optimization_enabled: bool | None = None


class RecipeResponse(RecipeBase):
    """Response schema for recipes."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    version: int
    status: RecipeStatus
    ingredients: list[RecipeIngredientResponse] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
