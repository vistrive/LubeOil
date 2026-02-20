"""Service for generating new recipe formulations from target specs."""

from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lobp.ai.blend_optimizer import optimize_for_target
from lobp.ai.blending_calculator import BlendComponent, calculate_blend
from lobp.models.inventory import Material

logger = structlog.get_logger()


class FormulationService:
    """Generates recipe formulations from target specifications."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_available_materials(self) -> list[dict[str, Any]]:
        """Fetch all active materials with their properties."""
        result = await self.db.execute(
            select(Material).where(Material.is_active.is_(True))
        )
        materials = result.scalars().all()
        return [
            {
                "code": m.code,
                "name": m.name,
                "category": m.category.value if m.category else "",
                "standard_viscosity_40c": m.standard_viscosity_40c,
                "standard_viscosity_100c": m.standard_viscosity_100c,
                "standard_viscosity_index": m.standard_viscosity_index,
                "standard_density_15c": m.standard_density_15c,
                "standard_flash_point": m.standard_flash_point,
                "standard_pour_point": m.standard_pour_point,
                "standard_cost_per_liter": m.standard_cost_per_liter,
            }
            for m in materials
        ]

    async def formulate(
        self,
        target_viscosity_40c: float | None = None,
        target_viscosity_100c: float | None = None,
        max_components: int = 4,
        max_additive_pct: float = 25.0,
        iterations: int = 500,
    ) -> dict[str, Any]:
        """
        Generate recipe candidates for given target specifications.

        Returns top candidates with predicted properties and cost estimates.
        """
        materials = await self.get_available_materials()
        if not materials:
            return {"error": "No materials available", "candidates": []}

        candidates = optimize_for_target(
            available_materials=materials,
            target_viscosity_40c=target_viscosity_40c,
            target_viscosity_100c=target_viscosity_100c,
            max_components=max_components,
            max_additive_pct=max_additive_pct / 100.0,
            iterations=iterations,
        )

        return {
            "target": {
                "viscosity_40c": target_viscosity_40c,
                "viscosity_100c": target_viscosity_100c,
            },
            "materials_available": len(materials),
            "candidates_found": len(candidates),
            "candidates": candidates,
        }

    async def calculate_recipe_properties(
        self, ingredients: list[dict],
    ) -> dict[str, Any]:
        """
        Calculate predicted properties for a given set of ingredients.

        Args:
            ingredients: List of dicts with material_code and weight_percent.
        """
        materials = await self.get_available_materials()
        mat_map = {m["code"]: m for m in materials}

        components = []
        for ing in ingredients:
            mat = mat_map.get(ing["material_code"])
            if not mat:
                return {"error": f"Unknown material: {ing['material_code']}"}
            components.append(BlendComponent(
                material_code=mat["code"],
                name=mat["name"],
                weight_fraction=ing["weight_percent"] / 100.0,
                viscosity_40c=mat.get("standard_viscosity_40c"),
                viscosity_100c=mat.get("standard_viscosity_100c"),
                viscosity_index=mat.get("standard_viscosity_index"),
                density_15c=mat.get("standard_density_15c"),
                flash_point=mat.get("standard_flash_point"),
                pour_point=mat.get("standard_pour_point"),
            ))

        result = calculate_blend(components)
        return {
            "viscosity_40c": result.viscosity_40c,
            "viscosity_100c": result.viscosity_100c,
            "viscosity_index": result.viscosity_index,
            "density_15c": result.density_15c,
            "flash_point": result.flash_point_estimate,
            "pour_point": result.pour_point_estimate,
            "total_weight_percent": round(result.total_weight_fraction * 100, 4),
            "warnings": result.warnings,
        }
