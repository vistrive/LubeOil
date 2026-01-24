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
    requires_premix: bool = False
    premix_time_minutes: int | None = None
    cost_per_liter: float | None = Field(None, ge=0)
    cost_per_kg: float | None = Field(None, ge=0)
    ai_adjustable: bool = True
    ai_adjustment_priority: int = Field(default=1, ge=1, le=10)


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

    # Product classification
    product_code: str | None = Field(None, max_length=50)
    product_family: str | None = Field(None, max_length=100)
    iso_grade: str | None = Field(None, max_length=20)

    # Target quality specifications (matches document format)
    target_viscosity_40c: float | None = Field(None, gt=0, description="Target viscosity @ 40°C in cSt")
    target_viscosity_100c: float | None = Field(None, gt=0, description="Target viscosity @ 100°C in cSt")
    target_viscosity_index: float | None = Field(None, description="Target viscosity index")
    target_flash_point: float | None = Field(None, description="Target flash point in °C")
    target_pour_point: float | None = Field(None, description="Target pour point in °C")
    target_density: float | None = Field(None, gt=0, description="Target density @ 15°C")
    target_tbn: float | None = Field(None, ge=0, description="Target TBN in mgKOH/g")
    target_tan: float | None = Field(None, ge=0, description="Target TAN in mgKOH/g")
    target_water_content_ppm: float | None = Field(None, ge=0, description="Max water content in ppm")
    target_foam_test_ml: float | None = Field(None, ge=0, description="Max foam test in mL")

    # Min/Max tolerance ranges (matches document format)
    viscosity_40c_min: float | None = Field(None, description="Min viscosity @ 40°C")
    viscosity_40c_max: float | None = Field(None, description="Max viscosity @ 40°C")
    viscosity_100c_min: float | None = Field(None, description="Min viscosity @ 100°C")
    viscosity_100c_max: float | None = Field(None, description="Max viscosity @ 100°C")
    viscosity_tolerance: float = Field(default=2.0, ge=0, le=10, description="Viscosity tolerance %")
    flash_point_min: float | None = Field(None, description="Min flash point °C")
    flash_point_tolerance: float = Field(default=5.0, ge=0, le=20, description="Flash point tolerance °C")
    pour_point_max: float | None = Field(None, description="Max pour point °C")
    pour_point_tolerance: float = Field(default=3.0, ge=0, le=10, description="Pour point tolerance °C")
    tbn_min: float | None = Field(None, description="Min TBN")
    tbn_max: float | None = Field(None, description="Max TBN")
    tbn_tolerance: float = Field(default=0.4, ge=0, description="TBN tolerance")
    water_content_max_ppm: float | None = Field(None, description="Max water content ppm")
    foam_test_max_ml: float | None = Field(None, description="Max foam test mL")

    # Batch size constraints
    min_batch_size_liters: float | None = Field(None, gt=0)
    max_batch_size_liters: float | None = Field(None, gt=0)
    standard_batch_size_liters: float | None = Field(None, gt=0)

    # Production parameters
    mixing_time_minutes: int | None = Field(None, ge=0)
    mixing_temperature_celsius: float | None = None
    mixing_speed_rpm: float | None = Field(None, ge=0)
    cooling_required: bool = False
    cooling_target_celsius: float | None = None

    # AI optimization
    ai_optimization_enabled: bool = True
    ai_cost_optimization_weight: float = Field(default=0.3, ge=0, le=1, description="Weight for cost optimization (0-1)")
    ai_quality_optimization_weight: float = Field(default=0.7, ge=0, le=1, description="Weight for quality optimization (0-1)")


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
