"""Blend service for managing blending operations."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from lobp.core.config import settings
from lobp.models.blend import Blend, BlendIngredient, BlendPriority, BlendStatus, IngredientStatus
from lobp.models.recipe import Recipe
from lobp.schemas.blend import BlendCreate, BlendProgressUpdate, BlendUpdate


class BlendService:
    """Service for blend management operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    def _generate_batch_number(self) -> str:
        """Generate a unique batch number."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        unique_id = str(uuid.uuid4())[:8].upper()
        return f"BL-{timestamp}-{unique_id}"

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        status: BlendStatus | None = None,
        priority: BlendPriority | None = None,
    ) -> list[Blend]:
        """Get all blends with optional filtering."""
        query = select(Blend).options(selectinload(Blend.ingredients))

        if status:
            query = query.where(Blend.status == status)
        if priority:
            query = query.where(Blend.priority == priority)

        query = query.order_by(Blend.created_at.desc()).offset(skip).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_by_id(self, blend_id: str) -> Blend | None:
        """Get a blend by ID."""
        query = (
            select(Blend)
            .options(selectinload(Blend.ingredients))
            .where(Blend.id == blend_id)
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_batch_number(self, batch_number: str) -> Blend | None:
        """Get a blend by batch number."""
        query = (
            select(Blend)
            .options(selectinload(Blend.ingredients))
            .where(Blend.batch_number == batch_number)
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def create(self, blend_data: BlendCreate, created_by: str | None = None) -> Blend:
        """Create a new blend from a recipe."""
        # Create blend
        blend = Blend(
            batch_number=self._generate_batch_number(),
            recipe_id=blend_data.recipe_id,
            target_volume_liters=blend_data.target_volume_liters,
            priority=blend_data.priority,
            blend_tank_tag=blend_data.blend_tank_tag,
            destination_tank_tag=blend_data.destination_tank_tag,
            scheduled_start=blend_data.scheduled_start,
            scheduled_end=blend_data.scheduled_end,
            notes=blend_data.notes,
            created_by=created_by,
        )

        self.db.add(blend)
        await self.db.flush()

        # Add ingredients from blend data or recipe
        if blend_data.ingredients:
            for idx, ing_data in enumerate(blend_data.ingredients, 1):
                ingredient = BlendIngredient(
                    blend_id=blend.id,
                    material_code=ing_data.material_code,
                    material_name=ing_data.material_name,
                    target_volume_liters=ing_data.target_volume_liters,
                    target_percentage=ing_data.target_percentage,
                    source_tank_tag=ing_data.source_tank_tag,
                    sequence_order=ing_data.sequence_order or idx,
                )
                self.db.add(ingredient)

        blend.total_steps = len(blend_data.ingredients)
        await self.db.commit()
        await self.db.refresh(blend)

        return await self.get_by_id(blend.id)  # type: ignore

    async def create_from_recipe(
        self,
        recipe_id: str,
        target_volume: float,
        created_by: str | None = None,
    ) -> Blend | None:
        """Create a new blend using recipe defaults."""
        # Get recipe
        query = (
            select(Recipe)
            .options(selectinload(Recipe.ingredients))
            .where(Recipe.id == recipe_id)
        )
        result = await self.db.execute(query)
        recipe = result.scalar_one_or_none()

        if not recipe:
            return None

        # Create blend
        blend = Blend(
            batch_number=self._generate_batch_number(),
            recipe_id=recipe_id,
            target_volume_liters=target_volume,
            created_by=created_by,
        )

        self.db.add(blend)
        await self.db.flush()

        # Create blend ingredients from recipe
        for idx, recipe_ing in enumerate(
            sorted(recipe.ingredients, key=lambda x: x.addition_order), 1
        ):
            ingredient_volume = target_volume * (recipe_ing.target_percentage / 100)
            ingredient = BlendIngredient(
                blend_id=blend.id,
                material_code=recipe_ing.material_code,
                material_name=recipe_ing.material_name,
                target_volume_liters=ingredient_volume,
                target_percentage=recipe_ing.target_percentage,
                sequence_order=idx,
                unit_cost=recipe_ing.cost_per_liter,
            )
            self.db.add(ingredient)

        blend.total_steps = len(recipe.ingredients)
        await self.db.commit()
        await self.db.refresh(blend)

        return await self.get_by_id(blend.id)

    async def update(self, blend_id: str, blend_data: BlendUpdate) -> Blend | None:
        """Update a blend."""
        blend = await self.get_by_id(blend_id)
        if not blend:
            return None

        update_data = blend_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(blend, field, value)

        blend.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(blend)
        return blend

    async def update_status(
        self, blend_id: str, status: BlendStatus, notes: str | None = None
    ) -> Blend | None:
        """Update blend status."""
        blend = await self.get_by_id(blend_id)
        if not blend:
            return None

        blend.status = status
        if notes:
            blend.notes = notes

        # Track timing
        if status == BlendStatus.IN_PROGRESS and not blend.actual_start:
            blend.actual_start = datetime.now(timezone.utc).isoformat()
        elif status in [BlendStatus.COMPLETED, BlendStatus.TRANSFERRED]:
            blend.actual_end = datetime.now(timezone.utc).isoformat()

        blend.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(blend)
        return blend

    async def update_progress(
        self, blend_id: str, progress_data: BlendProgressUpdate
    ) -> Blend | None:
        """Update blend progress from DCS."""
        blend = await self.get_by_id(blend_id)
        if not blend:
            return None

        blend.current_step = progress_data.current_step
        blend.progress_percent = progress_data.progress_percent
        blend.actual_volume_liters = progress_data.actual_volume_liters

        if progress_data.mixing_speed_rpm is not None:
            blend.mixing_speed_rpm = progress_data.mixing_speed_rpm
        if progress_data.mixing_temperature_celsius is not None:
            blend.mixing_temperature_celsius = progress_data.mixing_temperature_celsius
        if progress_data.energy_consumed_kwh is not None:
            blend.energy_consumed_kwh = progress_data.energy_consumed_kwh

        blend.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(blend)
        return blend

    async def hold_blend(self, blend_id: str, reason: str) -> Blend | None:
        """Put a blend on hold."""
        blend = await self.get_by_id(blend_id)
        if not blend:
            return None

        blend.status = BlendStatus.ON_HOLD
        blend.hold_reason = reason
        blend.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(blend)
        return blend

    async def get_queue(self) -> list[Blend]:
        """Get blends in the queue (scheduled, queued, or in progress)."""
        query = (
            select(Blend)
            .options(selectinload(Blend.ingredients), selectinload(Blend.recipe))
            .where(
                Blend.status.in_([
                    BlendStatus.SCHEDULED,
                    BlendStatus.QUEUED,
                    BlendStatus.PREPARING,
                    BlendStatus.IN_PROGRESS,
                    BlendStatus.MIXING,
                ])
            )
            .order_by(Blend.priority.desc(), Blend.scheduled_start)
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def check_off_spec_risk(self, blend_id: str) -> bool:
        """Check if blend has high off-spec risk."""
        blend = await self.get_by_id(blend_id)
        if not blend:
            return False

        return blend.off_spec_risk_percent > (settings.ai_quality_deviation_threshold * 100)

    async def update_ingredient_status(
        self,
        blend_id: str,
        material_code: str,
        status: IngredientStatus,
        actual_volume: float | None = None,
    ) -> BlendIngredient | None:
        """Update ingredient transfer status."""
        query = (
            select(BlendIngredient)
            .where(BlendIngredient.blend_id == blend_id)
            .where(BlendIngredient.material_code == material_code)
        )
        result = await self.db.execute(query)
        ingredient = result.scalar_one_or_none()

        if not ingredient:
            return None

        ingredient.status = status
        if actual_volume is not None:
            ingredient.actual_volume_liters = actual_volume
            # Calculate actual percentage
            blend = await self.get_by_id(blend_id)
            if blend and blend.actual_volume_liters > 0:
                ingredient.actual_percentage = (
                    actual_volume / blend.actual_volume_liters
                ) * 100

        if status == IngredientStatus.TRANSFERRING:
            ingredient.transfer_start = datetime.now(timezone.utc).isoformat()
        elif status == IngredientStatus.COMPLETED:
            ingredient.transfer_end = datetime.now(timezone.utc).isoformat()

        ingredient.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(ingredient)
        return ingredient
