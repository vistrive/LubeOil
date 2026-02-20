"""
Cross-recipe learning and optimization.

Implements:
- Transfer learning between similar recipes
- Knowledge sharing across product families
- Formulation pattern recognition
- New recipe suggestion based on existing knowledge
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from lobp.models.blend import Blend, BlendStatus
from lobp.models.recipe import Recipe, RecipeIngredient
from lobp.models.quality import QualityMeasurement


@dataclass
class RecipeKnowledge:
    """Learned knowledge about a recipe."""

    recipe_id: str
    recipe_code: str
    blend_count: int
    success_rate: float
    avg_quality_score: float
    optimal_conditions: dict[str, Any]
    common_issues: list[str]
    best_practices: list[str]


@dataclass
class IngredientInsight:
    """Insight about an ingredient's effect."""

    material_code: str
    effect_on_viscosity: float
    effect_on_flash_point: float
    effect_on_pour_point: float
    optimal_percentage_range: tuple[float, float]
    synergies: list[str]  # Materials it works well with
    conflicts: list[str]  # Materials to avoid combining


class CrossRecipeLearning:
    """
    AI system for learning across recipes and applying insights.

    Uses historical blend data to:
    1. Identify patterns in successful blends
    2. Transfer learnings between similar recipes
    3. Suggest optimizations based on cross-recipe analysis
    4. Recommend new formulations
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self._recipe_knowledge: dict[str, RecipeKnowledge] = {}
        self._ingredient_insights: dict[str, IngredientInsight] = {}
        self._similarity_cache: dict[tuple[str, str], float] = {}

    async def analyze_recipe_performance(
        self,
        recipe_id: str,
    ) -> RecipeKnowledge:
        """
        Analyze historical performance of a recipe.

        Extracts learnings from all blends of this recipe.
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

        # Get all completed blends
        blend_query = (
            select(Blend)
            .options(selectinload(Blend.quality_measurements))
            .where(Blend.recipe_id == recipe_id)
            .where(Blend.status == BlendStatus.COMPLETED)
        )
        blend_result = await self.db.execute(blend_query)
        blends = list(blend_result.scalars().all())

        if not blends:
            return RecipeKnowledge(
                recipe_id=recipe_id,
                recipe_code=recipe.code,
                blend_count=0,
                success_rate=0,
                avg_quality_score=0,
                optimal_conditions={},
                common_issues=[],
                best_practices=[],
            )

        # Calculate success metrics
        successful = sum(1 for b in blends if b.quality_approved)
        success_rate = successful / len(blends) if blends else 0

        # Analyze quality scores
        quality_scores = []
        for blend in blends:
            for m in blend.quality_measurements:
                if m.is_final and m.status.value == "on_spec":
                    quality_scores.append(100)
                elif m.status.value == "marginal":
                    quality_scores.append(75)
                else:
                    quality_scores.append(50)

        avg_quality = np.mean(quality_scores) if quality_scores else 0

        # Identify optimal conditions
        successful_blends = [b for b in blends if b.quality_approved]
        optimal_conditions = {}

        if successful_blends:
            temps = [b.mixing_temperature_celsius for b in successful_blends if b.mixing_temperature_celsius]
            speeds = [b.mixing_speed_rpm for b in successful_blends if b.mixing_speed_rpm]
            times = [b.mixing_duration_minutes for b in successful_blends if b.mixing_duration_minutes]

            if temps:
                optimal_conditions["mixing_temperature"] = {
                    "avg": np.mean(temps),
                    "min": np.min(temps),
                    "max": np.max(temps),
                }
            if speeds:
                optimal_conditions["mixing_speed"] = {
                    "avg": np.mean(speeds),
                    "min": np.min(speeds),
                    "max": np.max(speeds),
                }
            if times:
                optimal_conditions["mixing_time"] = {
                    "avg": np.mean(times),
                    "min": np.min(times),
                    "max": np.max(times),
                }

        # Identify common issues
        common_issues = []
        failed_blends = [b for b in blends if not b.quality_approved]
        if len(failed_blends) > len(blends) * 0.1:  # More than 10% failures
            common_issues.append("High failure rate - review formulation")
        if any(b.hold_reason for b in blends):
            hold_reasons = [b.hold_reason for b in blends if b.hold_reason]
            common_issues.extend(list(set(hold_reasons))[:3])

        # Generate best practices
        best_practices = []
        if optimal_conditions.get("mixing_temperature"):
            temp = optimal_conditions["mixing_temperature"]["avg"]
            best_practices.append(f"Maintain mixing temperature around {temp:.1f}°C")
        if optimal_conditions.get("mixing_time"):
            time = optimal_conditions["mixing_time"]["avg"]
            best_practices.append(f"Mix for approximately {time:.0f} minutes")

        knowledge = RecipeKnowledge(
            recipe_id=recipe_id,
            recipe_code=recipe.code,
            blend_count=len(blends),
            success_rate=success_rate,
            avg_quality_score=avg_quality,
            optimal_conditions=optimal_conditions,
            common_issues=common_issues,
            best_practices=best_practices,
        )

        self._recipe_knowledge[recipe_id] = knowledge
        return knowledge

    async def calculate_recipe_similarity(
        self,
        recipe1_id: str,
        recipe2_id: str,
    ) -> dict[str, Any]:
        """
        Calculate similarity between two recipes.

        Uses ingredient overlap, target specs, and performance data.
        """
        cache_key = (min(recipe1_id, recipe2_id), max(recipe1_id, recipe2_id))
        if cache_key in self._similarity_cache:
            return {"similarity": self._similarity_cache[cache_key]}

        query = select(Recipe).options(selectinload(Recipe.ingredients))

        result1 = await self.db.execute(query.where(Recipe.id == recipe1_id))
        result2 = await self.db.execute(query.where(Recipe.id == recipe2_id))

        recipe1 = result1.scalar_one_or_none()
        recipe2 = result2.scalar_one_or_none()

        if not recipe1 or not recipe2:
            return {"similarity": 0, "error": "Recipe not found"}

        # Ingredient similarity (Jaccard index)
        ing1 = {i.material_code for i in recipe1.ingredients}
        ing2 = {i.material_code for i in recipe2.ingredients}

        if ing1 or ing2:
            ingredient_sim = len(ing1 & ing2) / len(ing1 | ing2)
        else:
            ingredient_sim = 0

        # Specification similarity
        spec_diffs = []

        if recipe1.target_viscosity_40c and recipe2.target_viscosity_40c:
            diff = abs(recipe1.target_viscosity_40c - recipe2.target_viscosity_40c)
            max_val = max(recipe1.target_viscosity_40c, recipe2.target_viscosity_40c)
            spec_diffs.append(1 - (diff / max_val) if max_val > 0 else 0)

        if recipe1.target_flash_point and recipe2.target_flash_point:
            diff = abs(recipe1.target_flash_point - recipe2.target_flash_point)
            spec_diffs.append(1 - (diff / 300) if diff < 300 else 0)

        if recipe1.target_pour_point and recipe2.target_pour_point:
            diff = abs(recipe1.target_pour_point - recipe2.target_pour_point)
            spec_diffs.append(1 - (diff / 60) if diff < 60 else 0)

        spec_sim = np.mean(spec_diffs) if spec_diffs else 0

        # Overall similarity
        overall = 0.6 * ingredient_sim + 0.4 * spec_sim

        self._similarity_cache[cache_key] = overall

        return {
            "recipe1": recipe1.code,
            "recipe2": recipe2.code,
            "similarity": overall,
            "ingredient_similarity": ingredient_sim,
            "specification_similarity": spec_sim,
            "shared_ingredients": list(ing1 & ing2),
        }

    async def find_similar_recipes(
        self,
        recipe_id: str,
        min_similarity: float = 0.5,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find recipes similar to the given recipe."""
        query = select(Recipe).where(Recipe.id != recipe_id)
        result = await self.db.execute(query)
        all_recipes = list(result.scalars().all())

        similarities = []
        for recipe in all_recipes:
            sim_data = await self.calculate_recipe_similarity(recipe_id, recipe.id)
            if sim_data["similarity"] >= min_similarity:
                similarities.append({
                    "recipe_id": recipe.id,
                    "recipe_code": recipe.code,
                    "recipe_name": recipe.name,
                    **sim_data,
                })

        # Sort by similarity
        similarities.sort(key=lambda x: -x["similarity"])
        return similarities[:limit]

    async def transfer_learnings(
        self,
        source_recipe_id: str,
        target_recipe_id: str,
    ) -> dict[str, Any]:
        """
        Transfer learnings from a well-performing recipe to another.

        Returns suggestions for the target recipe.
        """
        # Get performance data
        source_knowledge = await self.analyze_recipe_performance(source_recipe_id)
        target_knowledge = await self.analyze_recipe_performance(target_recipe_id)

        # Check similarity
        similarity = await self.calculate_recipe_similarity(
            source_recipe_id, target_recipe_id
        )

        if similarity["similarity"] < 0.3:
            return {
                "success": False,
                "message": "Recipes too dissimilar for knowledge transfer",
                "similarity": similarity["similarity"],
            }

        suggestions = []

        # Transfer optimal conditions if source is better
        if source_knowledge.success_rate > target_knowledge.success_rate:
            if source_knowledge.optimal_conditions:
                suggestions.append({
                    "type": "process_conditions",
                    "description": "Apply optimal mixing conditions from source recipe",
                    "details": source_knowledge.optimal_conditions,
                    "expected_improvement": (
                        source_knowledge.success_rate - target_knowledge.success_rate
                    ) * 100,
                })

            # Transfer best practices
            for practice in source_knowledge.best_practices:
                suggestions.append({
                    "type": "best_practice",
                    "description": practice,
                    "source": source_knowledge.recipe_code,
                })

        return {
            "success": True,
            "source_recipe": source_knowledge.recipe_code,
            "target_recipe": target_knowledge.recipe_code,
            "similarity": similarity["similarity"],
            "source_success_rate": source_knowledge.success_rate,
            "target_success_rate": target_knowledge.success_rate,
            "suggestions": suggestions,
        }

    async def suggest_new_recipe(
        self,
        target_specs: dict[str, float],
        base_recipe_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Suggest a new recipe formulation based on target specifications.

        Uses knowledge from existing recipes to propose ingredients.
        """
        # Find recipes with similar specs
        query = select(Recipe).options(selectinload(Recipe.ingredients))
        result = await self.db.execute(query)
        all_recipes = list(result.scalars().all())

        # Score recipes by how close their specs are to target
        scored_recipes = []
        for recipe in all_recipes:
            score = 0
            count = 0

            if "viscosity_40c" in target_specs and recipe.target_viscosity_40c:
                diff = abs(recipe.target_viscosity_40c - target_specs["viscosity_40c"])
                score += 1 - min(1, diff / target_specs["viscosity_40c"])
                count += 1

            if "flash_point" in target_specs and recipe.target_flash_point:
                diff = abs(recipe.target_flash_point - target_specs["flash_point"])
                score += 1 - min(1, diff / 50)
                count += 1

            if "pour_point" in target_specs and recipe.target_pour_point:
                diff = abs(recipe.target_pour_point - target_specs["pour_point"])
                score += 1 - min(1, diff / 30)
                count += 1

            if count > 0:
                scored_recipes.append((recipe, score / count))

        # Sort by score
        scored_recipes.sort(key=lambda x: -x[1])

        if not scored_recipes:
            return {"success": False, "message": "No reference recipes found"}

        # Use top 3 recipes as reference
        top_recipes = scored_recipes[:3]

        # Analyze common ingredients
        ingredient_usage: dict[str, list[float]] = {}
        for recipe, _ in top_recipes:
            for ing in recipe.ingredients:
                if ing.material_code not in ingredient_usage:
                    ingredient_usage[ing.material_code] = []
                ingredient_usage[ing.material_code].append(ing.target_percentage)

        # Build suggested formulation
        suggested_ingredients = []
        for material_code, percentages in ingredient_usage.items():
            if len(percentages) >= 2:  # Used in at least 2 reference recipes
                suggested_ingredients.append({
                    "material_code": material_code,
                    "suggested_percentage": np.mean(percentages),
                    "range": (min(percentages), max(percentages)),
                    "confidence": len(percentages) / len(top_recipes),
                })

        # Sort by percentage (base oils first)
        suggested_ingredients.sort(key=lambda x: -x["suggested_percentage"])

        return {
            "success": True,
            "target_specifications": target_specs,
            "reference_recipes": [
                {"code": r.code, "score": s}
                for r, s in top_recipes
            ],
            "suggested_ingredients": suggested_ingredients,
            "notes": [
                "Formulation based on similar performing recipes",
                "Adjust percentages based on actual base oil properties",
                "Lab validation required before production",
            ],
        }

    async def get_ingredient_insights(
        self,
        material_code: str,
    ) -> IngredientInsight:
        """
        Get insights about an ingredient's effects on blend properties.

        Based on analysis of historical blend data.
        """
        # This would analyze historical data
        # Returning mock insights for demonstration

        # Default insights based on ingredient type
        if "VI" in material_code.upper():
            return IngredientInsight(
                material_code=material_code,
                effect_on_viscosity=5.0,  # Increases by 5 cSt per 1%
                effect_on_flash_point=0,
                effect_on_pour_point=0,
                optimal_percentage_range=(3.0, 8.0),
                synergies=["SN-150", "SN-500"],
                conflicts=[],
            )
        elif "PPD" in material_code.upper():
            return IngredientInsight(
                material_code=material_code,
                effect_on_viscosity=0,
                effect_on_flash_point=-2,
                effect_on_pour_point=-5.0,  # Decreases by 5°C per 0.1%
                optimal_percentage_range=(0.1, 0.5),
                synergies=[],
                conflicts=["WAX"],
            )
        else:
            return IngredientInsight(
                material_code=material_code,
                effect_on_viscosity=0,
                effect_on_flash_point=0,
                effect_on_pour_point=0,
                optimal_percentage_range=(0, 100),
                synergies=[],
                conflicts=[],
            )
