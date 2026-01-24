"""Recipe and RecipeIngredient database models."""

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
    from lobp.models.blend import Blend


class RecipeStatus(str, enum.Enum):
    """Recipe lifecycle status."""

    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    RETIRED = "retired"
    ARCHIVED = "archived"


class IngredientType(str, enum.Enum):
    """Type classification for recipe ingredients."""

    BASE_OIL = "base_oil"
    ADDITIVE = "additive"
    VISCOSITY_MODIFIER = "viscosity_modifier"
    POUR_POINT_DEPRESSANT = "pour_point_depressant"
    DETERGENT = "detergent"
    DISPERSANT = "dispersant"
    ANTI_WEAR = "anti_wear"
    ANTIOXIDANT = "antioxidant"
    CORROSION_INHIBITOR = "corrosion_inhibitor"
    FOAM_INHIBITOR = "foam_inhibitor"
    OTHER = "other"


class Recipe(BaseModel):
    """
    Recipe master data for lubricant blending.

    Contains target specifications, tolerances, and production parameters
    for creating a specific lubricant product.
    """

    __tablename__ = "recipes"

    # Identification
    code: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[RecipeStatus] = mapped_column(
        Enum(RecipeStatus), default=RecipeStatus.DRAFT
    )

    # Product classification
    product_code: Mapped[str | None] = mapped_column(String(50), index=True)
    product_family: Mapped[str | None] = mapped_column(String(100))
    iso_grade: Mapped[str | None] = mapped_column(String(20))

    # Target quality specifications
    target_viscosity_40c: Mapped[float | None] = mapped_column(Float)
    target_viscosity_100c: Mapped[float | None] = mapped_column(Float)
    target_viscosity_index: Mapped[float | None] = mapped_column(Float)
    target_flash_point: Mapped[float | None] = mapped_column(Float)
    target_pour_point: Mapped[float | None] = mapped_column(Float)
    target_density: Mapped[float | None] = mapped_column(Float)
    target_tbn: Mapped[float | None] = mapped_column(Float)
    target_tan: Mapped[float | None] = mapped_column(Float)
    target_water_content_ppm: Mapped[float | None] = mapped_column(Float)
    target_foam_test_ml: Mapped[float | None] = mapped_column(Float)

    # Tolerance ranges (percentage or absolute)
    viscosity_40c_min: Mapped[float | None] = mapped_column(Float)
    viscosity_40c_max: Mapped[float | None] = mapped_column(Float)
    viscosity_100c_min: Mapped[float | None] = mapped_column(Float)
    viscosity_100c_max: Mapped[float | None] = mapped_column(Float)
    viscosity_tolerance: Mapped[float] = mapped_column(Float, default=2.0)
    flash_point_min: Mapped[float | None] = mapped_column(Float)
    flash_point_tolerance: Mapped[float] = mapped_column(Float, default=5.0)
    pour_point_max: Mapped[float | None] = mapped_column(Float)
    pour_point_tolerance: Mapped[float] = mapped_column(Float, default=3.0)
    tbn_min: Mapped[float | None] = mapped_column(Float)
    tbn_max: Mapped[float | None] = mapped_column(Float)
    water_content_max_ppm: Mapped[float | None] = mapped_column(Float)
    foam_test_max_ml: Mapped[float | None] = mapped_column(Float)

    # Batch size constraints
    min_batch_size_liters: Mapped[float | None] = mapped_column(Float)
    max_batch_size_liters: Mapped[float | None] = mapped_column(Float)
    standard_batch_size_liters: Mapped[float | None] = mapped_column(Float)

    # Production parameters
    mixing_time_minutes: Mapped[int | None] = mapped_column(Integer)
    mixing_temperature_celsius: Mapped[float | None] = mapped_column(Float)
    mixing_speed_rpm: Mapped[float | None] = mapped_column(Float)
    cooling_required: Mapped[bool] = mapped_column(Boolean, default=False)
    cooling_target_celsius: Mapped[float | None] = mapped_column(Float)

    # AI optimization settings
    ai_optimization_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    ai_cost_optimization_weight: Mapped[float] = mapped_column(Float, default=0.3)
    ai_quality_optimization_weight: Mapped[float] = mapped_column(Float, default=0.7)

    # Approval tracking
    approved_by: Mapped[str | None] = mapped_column(String(100))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    ingredients: Mapped[list["RecipeIngredient"]] = relationship(
        "RecipeIngredient",
        back_populates="recipe",
        cascade="all, delete-orphan",
        order_by="RecipeIngredient.addition_order",
    )
    blends: Mapped[list["Blend"]] = relationship(
        "Blend", back_populates="recipe"
    )


class RecipeIngredient(BaseModel):
    """
    Individual ingredient specification within a recipe.

    Defines material requirements, percentages, and constraints
    for each component in the blend formula.
    """

    __tablename__ = "recipe_ingredients"

    recipe_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("recipes.id", ondelete="CASCADE"), index=True
    )

    # Material identification
    material_code: Mapped[str] = mapped_column(String(50), index=True)
    material_name: Mapped[str] = mapped_column(String(200))
    ingredient_type: Mapped[IngredientType] = mapped_column(Enum(IngredientType))

    # Percentage specifications
    target_percentage: Mapped[float] = mapped_column(Float)
    min_percentage: Mapped[float | None] = mapped_column(Float)
    max_percentage: Mapped[float | None] = mapped_column(Float)

    # Processing parameters
    addition_order: Mapped[int] = mapped_column(Integer, default=1)
    requires_heating: Mapped[bool] = mapped_column(Boolean, default=False)
    heating_temperature: Mapped[float | None] = mapped_column(Float)
    requires_premix: Mapped[bool] = mapped_column(Boolean, default=False)
    premix_time_minutes: Mapped[int | None] = mapped_column(Integer)

    # Cost tracking
    cost_per_liter: Mapped[float | None] = mapped_column(Float)
    cost_per_kg: Mapped[float | None] = mapped_column(Float)

    # AI optimization
    ai_adjustable: Mapped[bool] = mapped_column(Boolean, default=True)
    ai_adjustment_priority: Mapped[int] = mapped_column(Integer, default=1)

    # Relationships
    recipe: Mapped["Recipe"] = relationship("Recipe", back_populates="ingredients")
