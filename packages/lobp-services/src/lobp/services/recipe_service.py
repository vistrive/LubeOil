"""Recipe service for managing blend formulations."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from lobp.models.recipe import Recipe, RecipeIngredient, RecipeStatus
from lobp.schemas.recipe import RecipeCreate, RecipeUpdate


class RecipeService:
    """Service for recipe management operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        status: RecipeStatus | None = None,
    ) -> list[Recipe]:
        """Get all recipes with optional filtering."""
        query = select(Recipe).options(selectinload(Recipe.ingredients))

        if status:
            query = query.where(Recipe.status == status)

        query = query.offset(skip).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_by_id(self, recipe_id: str) -> Recipe | None:
        """Get a recipe by ID."""
        query = (
            select(Recipe)
            .options(selectinload(Recipe.ingredients))
            .where(Recipe.id == recipe_id)
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_code(self, code: str) -> Recipe | None:
        """Get a recipe by code."""
        query = (
            select(Recipe)
            .options(selectinload(Recipe.ingredients))
            .where(Recipe.code == code)
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def create(self, recipe_data: RecipeCreate) -> Recipe:
        """Create a new recipe with ingredients."""
        # Create recipe
        recipe = Recipe(
            code=recipe_data.code,
            name=recipe_data.name,
            description=recipe_data.description,
            target_viscosity_40c=recipe_data.target_viscosity_40c,
            target_viscosity_100c=recipe_data.target_viscosity_100c,
            target_viscosity_index=recipe_data.target_viscosity_index,
            target_flash_point=recipe_data.target_flash_point,
            target_pour_point=recipe_data.target_pour_point,
            target_density=recipe_data.target_density,
            target_tbn=recipe_data.target_tbn,
            target_tan=recipe_data.target_tan,
            viscosity_tolerance=recipe_data.viscosity_tolerance,
            flash_point_tolerance=recipe_data.flash_point_tolerance,
            pour_point_tolerance=recipe_data.pour_point_tolerance,
            min_batch_size_liters=recipe_data.min_batch_size_liters,
            max_batch_size_liters=recipe_data.max_batch_size_liters,
            standard_batch_size_liters=recipe_data.standard_batch_size_liters,
            mixing_time_minutes=recipe_data.mixing_time_minutes,
            mixing_temperature_celsius=recipe_data.mixing_temperature_celsius,
            cooling_required=recipe_data.cooling_required,
            ai_optimization_enabled=recipe_data.ai_optimization_enabled,
        )

        self.db.add(recipe)
        await self.db.flush()

        # Add ingredients
        for ing_data in recipe_data.ingredients:
            ingredient = RecipeIngredient(
                recipe_id=recipe.id,
                material_code=ing_data.material_code,
                material_name=ing_data.material_name,
                ingredient_type=ing_data.ingredient_type,
                target_percentage=ing_data.target_percentage,
                min_percentage=ing_data.min_percentage,
                max_percentage=ing_data.max_percentage,
                addition_order=ing_data.addition_order,
                requires_heating=ing_data.requires_heating,
                heating_temperature=ing_data.heating_temperature,
                cost_per_liter=ing_data.cost_per_liter,
                ai_adjustable=ing_data.ai_adjustable,
            )
            self.db.add(ingredient)

        await self.db.commit()
        await self.db.refresh(recipe)

        # Load ingredients for response
        return await self.get_by_id(recipe.id)  # type: ignore

    async def update(self, recipe_id: str, recipe_data: RecipeUpdate) -> Recipe | None:
        """Update an existing recipe."""
        recipe = await self.get_by_id(recipe_id)
        if not recipe:
            return None

        update_data = recipe_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(recipe, field, value)

        recipe.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(recipe)
        return recipe

    async def delete(self, recipe_id: str) -> bool:
        """Delete a recipe."""
        recipe = await self.get_by_id(recipe_id)
        if not recipe:
            return False

        await self.db.delete(recipe)
        await self.db.commit()
        return True

    async def approve(self, recipe_id: str, approved_by: str) -> Recipe | None:
        """Approve a recipe for production use."""
        recipe = await self.get_by_id(recipe_id)
        if not recipe:
            return None

        recipe.status = RecipeStatus.APPROVED
        recipe.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(recipe)
        return recipe

    async def validate_ingredients_total(self, recipe_id: str) -> tuple[bool, float]:
        """Validate that recipe ingredients sum to 100%."""
        recipe = await self.get_by_id(recipe_id)
        if not recipe:
            return False, 0.0

        total = sum(ing.target_percentage for ing in recipe.ingredients)
        return abs(total - 100.0) < 0.01, total

    async def get_approved_recipes(self) -> list[Recipe]:
        """Get all approved recipes available for blending."""
        return await self.get_all(status=RecipeStatus.APPROVED)
