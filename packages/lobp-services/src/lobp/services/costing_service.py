"""
Real-time batch costing service.

Implements:
- Material cost calculation
- Energy cost tracking
- Labor cost allocation
- Overhead allocation
- Variance analysis
- Cost per liter metrics
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from lobp.models.blend import Blend, BlendIngredient, BlendStatus
from lobp.models.inventory import Material, MaterialLot
from lobp.models.recipe import Recipe


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a blend."""

    material_cost: float
    energy_cost: float
    labor_cost: float
    overhead_cost: float
    total_cost: float
    cost_per_liter: float
    margin_percent: float | None = None


@dataclass
class CostVariance:
    """Cost variance analysis."""

    category: str
    budgeted: float
    actual: float
    variance: float
    variance_percent: float
    favorable: bool


class CostingService:
    """Service for real-time batch costing and analysis."""

    def __init__(self, db: AsyncSession):
        self.db = db

        # Default cost rates (should be configurable)
        self.energy_rate_per_kwh = 0.12  # $/kWh
        self.labor_rate_per_hour = 45.0  # $/hour
        self.overhead_rate_percent = 15.0  # % of direct costs

    async def calculate_blend_cost(
        self,
        blend_id: str,
        include_variance: bool = True,
    ) -> dict[str, Any]:
        """
        Calculate complete cost breakdown for a blend.

        Args:
            blend_id: Blend to calculate costs for
            include_variance: Include variance analysis vs standard

        Returns:
            Complete cost breakdown with analysis
        """
        query = (
            select(Blend)
            .options(
                selectinload(Blend.recipe).selectinload(Recipe.ingredients),
                selectinload(Blend.ingredients),
            )
            .where(Blend.id == blend_id)
        )
        result = await self.db.execute(query)
        blend = result.scalar_one_or_none()

        if not blend:
            raise ValueError(f"Blend {blend_id} not found")

        # Calculate material costs
        material_costs = await self._calculate_material_cost(blend)

        # Calculate energy costs
        energy_cost = self._calculate_energy_cost(blend)

        # Calculate labor costs
        labor_cost = self._calculate_labor_cost(blend)

        # Calculate overhead
        direct_costs = material_costs["total"] + energy_cost + labor_cost
        overhead_cost = direct_costs * (self.overhead_rate_percent / 100)

        total_cost = direct_costs + overhead_cost
        volume = blend.actual_volume_liters or blend.target_volume_liters
        cost_per_liter = total_cost / volume if volume > 0 else 0

        result_data = {
            "blend_id": blend_id,
            "batch_number": blend.batch_number,
            "calculated_at": datetime.now(timezone.utc).isoformat(),

            # Volume
            "target_volume": blend.target_volume_liters,
            "actual_volume": blend.actual_volume_liters,

            # Cost Summary
            "cost_summary": {
                "material_cost": material_costs["total"],
                "energy_cost": energy_cost,
                "labor_cost": labor_cost,
                "overhead_cost": overhead_cost,
                "total_cost": total_cost,
                "cost_per_liter": cost_per_liter,
            },

            # Material Details
            "material_costs": material_costs["details"],

            # Energy Details
            "energy_details": {
                "kwh_consumed": blend.energy_consumed_kwh,
                "rate_per_kwh": self.energy_rate_per_kwh,
                "total_cost": energy_cost,
            },

            # Efficiency Metrics
            "efficiency": {
                "material_yield_percent": (
                    blend.actual_volume_liters /
                    sum(i.actual_volume_liters or i.target_volume_liters
                        for i in blend.ingredients) * 100
                    if blend.ingredients else 0
                ),
                "energy_per_1000l": (
                    blend.energy_consumed_kwh / (volume / 1000)
                    if volume > 0 and blend.energy_consumed_kwh else 0
                ),
            },
        }

        # Add variance analysis if requested
        if include_variance and blend.recipe:
            result_data["variance_analysis"] = await self._calculate_variance(
                blend, total_cost, material_costs["total"]
            )

        return result_data

    async def _calculate_material_cost(
        self,
        blend: Blend,
    ) -> dict[str, Any]:
        """Calculate material costs from blend ingredients."""
        total = 0.0
        details = []

        for ing in blend.ingredients:
            volume = ing.actual_volume_liters or ing.target_volume_liters
            unit_cost = ing.unit_cost

            # If no unit cost on ingredient, try to get from material
            if not unit_cost:
                material_query = select(Material).where(
                    Material.code == ing.material_code
                )
                mat_result = await self.db.execute(material_query)
                material = mat_result.scalar_one_or_none()
                if material:
                    unit_cost = material.last_purchase_cost or material.standard_cost_per_liter

            cost = volume * (unit_cost or 0)
            total += cost

            details.append({
                "material_code": ing.material_code,
                "material_name": ing.material_name,
                "volume_liters": volume,
                "unit_cost": unit_cost,
                "total_cost": cost,
                "percentage_of_total": 0,  # Will calculate after
            })

        # Calculate percentages
        if total > 0:
            for d in details:
                d["percentage_of_total"] = (d["total_cost"] / total) * 100

        return {"total": total, "details": details}

    def _calculate_energy_cost(self, blend: Blend) -> float:
        """Calculate energy cost from consumption."""
        if blend.energy_consumed_kwh:
            return blend.energy_consumed_kwh * self.energy_rate_per_kwh
        else:
            # Estimate based on volume and typical consumption
            # Assume 4 kWh per 1000L
            volume = blend.actual_volume_liters or blend.target_volume_liters
            estimated_kwh = (volume / 1000) * 4
            return estimated_kwh * self.energy_rate_per_kwh

    def _calculate_labor_cost(self, blend: Blend) -> float:
        """Calculate labor cost based on blend duration."""
        if blend.actual_start and blend.actual_end:
            start = datetime.fromisoformat(str(blend.actual_start))
            end = datetime.fromisoformat(str(blend.actual_end))
            hours = (end - start).total_seconds() / 3600
        else:
            # Estimate based on volume
            volume = blend.actual_volume_liters or blend.target_volume_liters
            hours = 0.5 + (volume / 1000) * 0.3

        # Assume 1 operator
        return hours * self.labor_rate_per_hour

    async def _calculate_variance(
        self,
        blend: Blend,
        actual_total: float,
        actual_material: float,
    ) -> list[dict[str, Any]]:
        """Calculate cost variance vs standard."""
        variances = []
        recipe = blend.recipe

        if not recipe:
            return variances

        # Calculate standard material cost
        standard_material = 0.0
        for recipe_ing in recipe.ingredients:
            # Get material standard cost
            material_query = select(Material).where(
                Material.code == recipe_ing.material_code
            )
            mat_result = await self.db.execute(material_query)
            material = mat_result.scalar_one_or_none()

            if material:
                volume = blend.target_volume_liters * (recipe_ing.target_percentage / 100)
                standard_material += volume * (material.standard_cost_per_liter or 0)

        # Material variance
        material_variance = actual_material - standard_material
        variances.append({
            "category": "Material",
            "budgeted": standard_material,
            "actual": actual_material,
            "variance": material_variance,
            "variance_percent": (
                (material_variance / standard_material * 100)
                if standard_material > 0 else 0
            ),
            "favorable": material_variance < 0,
        })

        # Volume variance
        volume_variance = blend.actual_volume_liters - blend.target_volume_liters
        variances.append({
            "category": "Volume",
            "budgeted": blend.target_volume_liters,
            "actual": blend.actual_volume_liters,
            "variance": volume_variance,
            "variance_percent": (
                (volume_variance / blend.target_volume_liters * 100)
                if blend.target_volume_liters > 0 else 0
            ),
            "favorable": volume_variance >= 0,  # More volume is favorable
        })

        return variances

    async def get_cost_summary_by_recipe(
        self,
        recipe_id: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get cost summary for recent blends of a recipe."""
        query = (
            select(Blend)
            .options(selectinload(Blend.ingredients))
            .where(Blend.recipe_id == recipe_id)
            .where(Blend.status == BlendStatus.COMPLETED)
            .order_by(Blend.created_at.desc())
            .limit(limit)
        )
        result = await self.db.execute(query)
        blends = list(result.scalars().all())

        if not blends:
            return {"recipe_id": recipe_id, "blend_count": 0}

        costs = []
        for blend in blends:
            material_cost = sum(
                (ing.actual_volume_liters or ing.target_volume_liters) * (ing.unit_cost or 0)
                for ing in blend.ingredients
            )
            volume = blend.actual_volume_liters or blend.target_volume_liters
            cost_per_liter = material_cost / volume if volume > 0 else 0

            costs.append({
                "batch_number": blend.batch_number,
                "volume": volume,
                "material_cost": material_cost,
                "cost_per_liter": cost_per_liter,
                "energy_kwh": blend.energy_consumed_kwh,
            })

        avg_cost_per_liter = sum(c["cost_per_liter"] for c in costs) / len(costs)
        avg_energy = sum(c["energy_kwh"] or 0 for c in costs) / len(costs)

        return {
            "recipe_id": recipe_id,
            "blend_count": len(blends),
            "average_cost_per_liter": avg_cost_per_liter,
            "average_energy_kwh": avg_energy,
            "min_cost_per_liter": min(c["cost_per_liter"] for c in costs),
            "max_cost_per_liter": max(c["cost_per_liter"] for c in costs),
            "recent_blends": costs,
        }

    async def get_daily_production_cost(
        self,
        date: datetime,
    ) -> dict[str, Any]:
        """Get total production cost for a day."""
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = date.replace(hour=23, minute=59, second=59, microsecond=999999)

        query = (
            select(Blend)
            .options(selectinload(Blend.ingredients))
            .where(Blend.actual_start >= start_of_day)
            .where(Blend.actual_start <= end_of_day)
        )
        result = await self.db.execute(query)
        blends = list(result.scalars().all())

        total_volume = 0
        total_material_cost = 0
        total_energy_kwh = 0

        for blend in blends:
            total_volume += blend.actual_volume_liters or 0
            total_energy_kwh += blend.energy_consumed_kwh or 0

            for ing in blend.ingredients:
                volume = ing.actual_volume_liters or ing.target_volume_liters
                total_material_cost += volume * (ing.unit_cost or 0)

        total_energy_cost = total_energy_kwh * self.energy_rate_per_kwh

        return {
            "date": date.date().isoformat(),
            "blend_count": len(blends),
            "total_volume": total_volume,
            "total_material_cost": total_material_cost,
            "total_energy_cost": total_energy_cost,
            "total_energy_kwh": total_energy_kwh,
            "cost_per_liter": (
                total_material_cost / total_volume if total_volume > 0 else 0
            ),
            "energy_per_1000l": (
                total_energy_kwh / (total_volume / 1000) if total_volume > 0 else 0
            ),
        }

    async def estimate_blend_cost(
        self,
        recipe_id: str,
        target_volume: float,
    ) -> dict[str, Any]:
        """
        Estimate cost for a planned blend.

        Uses standard costs and historical averages.
        """
        query = (
            select(Recipe)
            .options(selectinload(Recipe.ingredients))
            .where(Recipe.id == recipe_id)
        )
        result = await self.db.execute(query)
        recipe = result.scalar_one_or_none()

        if not recipe:
            raise ValueError(f"Recipe {recipe_id} not found")

        material_cost = 0.0
        material_details = []

        for ing in recipe.ingredients:
            volume = target_volume * (ing.target_percentage / 100)

            # Get material cost
            material_query = select(Material).where(
                Material.code == ing.material_code
            )
            mat_result = await self.db.execute(material_query)
            material = mat_result.scalar_one_or_none()

            unit_cost = (
                material.standard_cost_per_liter
                if material else ing.cost_per_liter
            ) or 0

            cost = volume * unit_cost
            material_cost += cost

            material_details.append({
                "material_code": ing.material_code,
                "material_name": ing.material_name,
                "volume_liters": volume,
                "unit_cost": unit_cost,
                "total_cost": cost,
            })

        # Estimate energy (4 kWh per 1000L)
        energy_kwh = (target_volume / 1000) * 4
        energy_cost = energy_kwh * self.energy_rate_per_kwh

        # Estimate labor (0.5 + 0.3 per 1000L hours)
        labor_hours = 0.5 + (target_volume / 1000) * 0.3
        labor_cost = labor_hours * self.labor_rate_per_hour

        direct_costs = material_cost + energy_cost + labor_cost
        overhead = direct_costs * (self.overhead_rate_percent / 100)
        total = direct_costs + overhead

        return {
            "recipe_id": recipe_id,
            "recipe_code": recipe.code,
            "target_volume": target_volume,
            "estimated_costs": {
                "material": material_cost,
                "energy": energy_cost,
                "labor": labor_cost,
                "overhead": overhead,
                "total": total,
                "per_liter": total / target_volume if target_volume > 0 else 0,
            },
            "material_details": material_details,
            "assumptions": {
                "energy_rate": self.energy_rate_per_kwh,
                "labor_rate": self.labor_rate_per_hour,
                "overhead_percent": self.overhead_rate_percent,
            },
        }
